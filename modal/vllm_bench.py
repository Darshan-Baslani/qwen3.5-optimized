import json
import logging
import os
import statistics
import time
from typing import Any

import modal


APP_NAME = "qwen-vllm-bench"
CACHE_DIR = "/root/.cache/huggingface"
MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_GPU = "B200"
DEFAULT_PROMPT = "The architecture of the Blackwell B200 GPU allows for"
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
DEFAULT_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.20.1")
VLLM_PACKAGE = os.environ.get("VLLM_PACKAGE", f"vllm=={VLLM_VERSION}")
TORCH_CUDA_INDEX = os.environ.get("TORCH_CUDA_INDEX", "https://download.pytorch.org/whl/cu130")


app = modal.App(APP_NAME)
volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        VLLM_PACKAGE,
        "hf_transfer",
        extra_index_url=TORCH_CUDA_INDEX,
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_NO_USAGE_STATS": "1",
            # vLLM 0.20.1 has active startup regressions around DeepGEMM and V1
            # engine initialization on some CUDA 13 / Blackwell setups.
            "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM", "0"),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "0"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def maybe_round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def extract_ttft_seconds(metrics: Any) -> float | None:
    if metrics is None:
        return None

    if hasattr(metrics, "first_token_time") and hasattr(metrics, "arrival_time"):
        first_token_time = getattr(metrics, "first_token_time", None)
        arrival_time = getattr(metrics, "arrival_time", None)
        if first_token_time is not None and arrival_time is not None:
            return first_token_time - arrival_time

    if hasattr(metrics, "first_token_ts") and hasattr(metrics, "scheduled_ts"):
        first_token_ts = getattr(metrics, "first_token_ts", None)
        scheduled_ts = getattr(metrics, "scheduled_ts", None)
        if first_token_ts is not None and scheduled_ts is not None:
            return first_token_ts - scheduled_ts

    return None


def extract_engine_times(metrics: Any) -> dict[str, float | None]:
    if metrics is None:
        return {
            "queue_time_s": None,
            "scheduler_time_s": None,
            "model_forward_time_s": None,
            "model_execute_time_s": None,
        }

    return {
        "queue_time_s": getattr(metrics, "time_in_queue", None),
        "scheduler_time_s": getattr(metrics, "scheduler_time", None),
        "model_forward_time_s": getattr(metrics, "model_forward_time", None),
        "model_execute_time_s": getattr(metrics, "model_execute_time", None),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    generation_times = [run["generation_time_s"] for run in runs]
    output_tps = [run["output_tokens_per_second"] for run in runs]
    total_tps = [run["total_tokens_per_second"] for run in runs]
    ttfts = [run["time_to_first_token_s"] for run in runs if run["time_to_first_token_s"] is not None]

    return {
        "runs": len(runs),
        "generation_time_s": {
            "mean": maybe_round(statistics.mean(generation_times)),
            "median": maybe_round(statistics.median(generation_times)),
            "min": maybe_round(min(generation_times)),
            "max": maybe_round(max(generation_times)),
        },
        "output_tokens_per_second": {
            "mean": maybe_round(statistics.mean(output_tps), 2),
            "median": maybe_round(statistics.median(output_tps), 2),
            "min": maybe_round(min(output_tps), 2),
            "max": maybe_round(max(output_tps), 2),
        },
        "total_tokens_per_second": {
            "mean": maybe_round(statistics.mean(total_tps), 2),
            "median": maybe_round(statistics.median(total_tps), 2),
            "min": maybe_round(min(total_tps), 2),
            "max": maybe_round(max(total_tps), 2),
        },
        "time_to_first_token_s": None
        if not ttfts
        else {
            "mean": maybe_round(statistics.mean(ttfts)),
            "median": maybe_round(statistics.median(ttfts)),
            "min": maybe_round(min(ttfts)),
            "max": maybe_round(max(ttfts)),
        },
    }


def run_generation(llm: Any, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    start_time = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    generation_time_s = time.perf_counter() - start_time

    output = outputs[0]
    completion = output.outputs[0]
    prompt_tokens = len(output.prompt_token_ids or [])
    generated_tokens = len(completion.token_ids)
    total_tokens = prompt_tokens + generated_tokens
    ttft_s = extract_ttft_seconds(output.metrics)
    decode_time_s = None if ttft_s is None else max(generation_time_s - ttft_s, 0.0)
    decode_tps = None if decode_time_s is None else safe_divide(generated_tokens, decode_time_s)

    engine_times = extract_engine_times(output.metrics)

    return {
        "prompt": prompt,
        "generated_text": completion.text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "generation_time_s": maybe_round(generation_time_s),
        "time_to_first_token_s": maybe_round(ttft_s),
        "decode_time_after_first_token_s": maybe_round(decode_time_s),
        "output_tokens_per_second": maybe_round(safe_divide(generated_tokens, generation_time_s), 2),
        "total_tokens_per_second": maybe_round(safe_divide(total_tokens, generation_time_s), 2),
        "decode_tokens_per_second_after_first_token": maybe_round(decode_tps, 2),
        "num_cached_tokens": getattr(output, "num_cached_tokens", None),
        "finish_reason": completion.finish_reason,
        "engine_metrics": {key: maybe_round(value) for key, value in engine_times.items()},
    }


@app.function(
    gpu=DEFAULT_GPU,
    image=image,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
def benchmark_vllm(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 150,
    warmup_runs: int = 1,
    benchmark_runs: int = 3,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
):
    from vllm import LLM

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if benchmark_runs <= 0:
        raise ValueError("benchmark_runs must be >= 1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be >= 1")

    logger = configure_logger()
    logger.info("Initializing vLLM benchmark container")

    init_start = time.perf_counter()
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        download_dir=CACHE_DIR,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    init_time_s = time.perf_counter() - init_start
    logger.info("Loaded %s with vLLM in %.2f seconds", MODEL_ID, init_time_s)

    warmup_results = []
    for index in range(warmup_runs):
        logger.info("Warmup run %s/%s", index + 1, warmup_runs)
        warmup_results.append(run_generation(llm, prompt, max_new_tokens, temperature, top_p))

    benchmark_results = []
    for index in range(benchmark_runs):
        logger.info("Benchmark run %s/%s", index + 1, benchmark_runs)
        run_result = run_generation(llm, prompt, max_new_tokens, temperature, top_p)
        benchmark_results.append(run_result)
        logger.info(
            "Run %s: %.2f tok/s output throughput, %.4f s total generation",
            index + 1,
            run_result["output_tokens_per_second"],
            run_result["generation_time_s"],
        )

    result = {
        "app_name": APP_NAME,
        "model_id": MODEL_ID,
        "gpu": os.environ.get("MODAL_GPU", DEFAULT_GPU),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "model_init_time_s": maybe_round(init_time_s),
        "warmup_runs": warmup_results,
        "benchmark_runs": benchmark_results,
        "summary": summarize_runs(benchmark_results),
    }

    logger.info("Benchmark summary:\n%s", json.dumps(result["summary"], indent=2))
    return result


@app.local_entrypoint()
def main(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 150,
    warmup_runs: int = 1,
    benchmark_runs: int = 3,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
):
    result = benchmark_vllm.remote(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        temperature=temperature,
        top_p=top_p,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print(json.dumps(result, indent=2))

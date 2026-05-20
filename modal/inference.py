import glob
import json
import logging
import os
import time

import modal
import torch


APP_NAME = "qwen-b200-baremetal"
CACHE_DIR = "/root/.cache/huggingface"
MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_GPU = "B200"
TORCH_VERSION = "2.7.0"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
MAX_SEQUENCE_LENGTH = 8192

PREFIX_REWRITES = (
    ("model.language_model.", "model."),
    ("model.text_model.", "model."),
    ("language_model.", "model."),
    ("text_model.", "model."),
)
SKIP_WEIGHT_PATTERNS = ("visual", "audio", "vision_model", "mtp")


app = modal.App(APP_NAME)
volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang")
    .pip_install(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
        f"torch=={TORCH_VERSION}",
        "triton",
        "transformers",
        "huggingface_hub",
        "safetensors",
        extra_index_url=TORCH_CUDA_INDEX,
    )
    .env(
        {
            "MAX_JOBS": "4",
            "CACHE_BUSTER": "3",
        }
    )
    .add_local_file("./qwen.py", remote_path="/root/qwen.py")
    .add_local_dir("kernels/", remote_path="/root/kernels/")
)


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float16


def load_text_config(checkpoint_dir: str) -> dict:
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        hf_config = json.load(f)
    return hf_config.get("text_config", hf_config)


def build_model_config(text_cfg: dict):
    from qwen import Qwen3_5TextConfig

    rope_parameters = text_cfg.get("rope_parameters", {})
    return Qwen3_5TextConfig(
        vocab_size=text_cfg.get("vocab_size", 151936),
        hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg["intermediate_size"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        head_dim=text_cfg.get("head_dim", text_cfg["hidden_size"] // text_cfg["num_attention_heads"]),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=rope_parameters.get("rope_theta", text_cfg.get("rope_theta", 1000000.0)),
        partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
        mrope_section=rope_parameters.get("mrope_section", [11, 11, 10]),
        max_position_embeddings=text_cfg.get("max_position_embeddings", 32768),
        full_attention_interval=text_cfg.get("full_attention_interval", 4),
        layer_types=text_cfg.get("layer_types", []),
        hidden_act=text_cfg.get("hidden_act", "silu"),
        attention_dropout=text_cfg.get("attention_dropout", 0.0),
        attention_bias=text_cfg.get("attention_bias", False),
        pad_token_id=text_cfg.get("eos_token_id"),
        linear_conv_kernel_dim=text_cfg.get("linear_conv_kernel_dim", 4),
        linear_key_head_dim=text_cfg.get("linear_key_head_dim", 128),
        linear_value_head_dim=text_cfg.get("linear_value_head_dim", 128),
        linear_num_key_heads=text_cfg.get("linear_num_key_heads", 16),
        linear_num_value_heads=text_cfg.get("linear_num_value_heads", 32),
    )


def normalize_weight_key(key: str) -> str:
    for source_prefix, target_prefix in PREFIX_REWRITES:
        if key.startswith(source_prefix):
            return target_prefix + key[len(source_prefix):]
    return key


def load_model_weights(model, checkpoint_dir: str, logger: logging.Logger) -> None:
    from safetensors.torch import load_file

    mapped_state_dict = {}
    safetensor_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

    for filepath in safetensor_files:
        for key, tensor in load_file(filepath).items():
            if any(pattern in key for pattern in SKIP_WEIGHT_PATTERNS):
                continue
            norm_key = normalize_weight_key(key)
            
            if "input_layernorm.weight" in norm_key:
                standard_key = norm_key.replace("input_layernorm.weight", "input_layernorm_standard.weight")
                fused_key = norm_key.replace("input_layernorm.weight", "input_layernorm_fused.weight")
                
                mapped_state_dict[standard_key] = tensor
                # Clone the tensor so the two modules don't share the exact same memory address
                mapped_state_dict[fused_key] = tensor.clone() 
                
            else:
                mapped_state_dict[norm_key] = tensor

    if "lm_head.weight" not in mapped_state_dict:
        mapped_state_dict["lm_head.weight"] = mapped_state_dict["model.embed_tokens.weight"].clone()

    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)

    if unexpected_keys:
        logger.warning("Unexpected keys while loading weights: %s", unexpected_keys[:5])

    real_missing = [key for key in missing_keys if "inv_freq" not in key]
    if real_missing:
        raise RuntimeError(
            "Missing required weights after checkpoint mapping. "
            f"First missing keys: {real_missing[:20]}"
        )


@app.function(
    gpu=DEFAULT_GPU,
    image=image,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
def execute_inference(prompt: str, max_new_tokens: int = 100):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from qwen import Qwen3_5ForCausalLM, generate

    logger = configure_logger()
    device = torch.device("cuda")

    logger.info("Initializing container on %s", torch.cuda.get_device_name(device))

    checkpoint_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        allow_patterns=["*.safetensors", "*.json"],
    )

    text_cfg = load_text_config(checkpoint_dir)
    model_config = build_model_config(text_cfg)
    model_dtype = resolve_torch_dtype(text_cfg.get("dtype"))

    model = Qwen3_5ForCausalLM(model_config).to(device=device, dtype=model_dtype)
    load_model_weights(model, checkpoint_dir, logger)
    logger.info("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    input_ids = torch.tensor([tokenizer(prompt).input_ids], dtype=torch.long, device=device)

    logger.info("Generating %s tokens", max_new_tokens)
    start_time = time.time()
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_seq_len=min(model_config.max_position_embeddings, MAX_SEQUENCE_LENGTH),
    )
    total_time = time.time() - start_time

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info("Output:\n%s\n", generated_text)
    logger.info("Tokens Generated: %s", max_new_tokens)
    logger.info("Total Time: %.2f seconds", total_time)
    logger.info("Throughput: %.2f tokens/sec", max_new_tokens / total_time)


@app.local_entrypoint()
def main():
    execute_inference.remote(
        prompt="The architecture of the Blackwell B200 GPU allows for",
        max_new_tokens=150,
    )

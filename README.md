# Qwen 3.5 Bare-Metal Inference on Modal

Custom bare-metal inference stack for running `Qwen/Qwen3.5-9B` on Modal with a hand-written PyTorch model in [qwen.py](/mnt/Code/qwen_3.5/qwen.py:1) and a Modal entrypoint in [modal/inference.py](/mnt/Code/qwen_3.5/modal/inference.py:1).

This repo is currently focused on two things:

- correctness of a custom Qwen 3.5 text-model implementation
- iterative optimization of inference speed on high-end NVIDIA GPUs

## What This Repo Does

- Downloads `Qwen/Qwen3.5-9B` weights from Hugging Face into a persistent Modal volume.
- Reconstructs the text model with a custom PyTorch implementation.
- Maps checkpoint weights into the local module layout.
- Runs autoregressive generation on an NVIDIA B200.

## Repo Layout

- [qwen.py](/mnt/Code/qwen_3.5/qwen.py:1): custom Qwen 3.5 text model and generation loop.
- [modal/inference.py](/mnt/Code/qwen_3.5/modal/inference.py:1): Modal app, container image, weight loading, and remote inference entrypoint.

## Architecture

High-level execution flow:

1. Modal starts a CUDA 12.8 container with PyTorch 2.7.
2. The app downloads or reuses cached `Qwen/Qwen3.5-9B` safetensors in a persistent volume.
3. The Hugging Face config is translated into the local `Qwen3_5TextConfig`.
4. Checkpoint keys are normalized into the custom module layout and loaded into the model.
5. Generation runs through the custom decode loop with explicit KV, convolution, and recurrent caches.

Current implementation characteristics:

- custom PyTorch text-model implementation rather than `transformers` runtime inference
- explicit checkpoint mapping between HF naming and local module naming
- pure PyTorch fallback path for the DeltaNet-style linear-attention layers
- correctness-first decode path with room for kernel-level optimization

## Run

Prerequisites:

- Python with `modal` CLI installed locally.
- A Modal account.
- Optional: `HF_TOKEN` for higher Hugging Face rate limits.

Run the default prompt:

```bash
modal run modal/inference.py
```

Select a different GPU:

```bash
MODAL_GPU=H100 modal run modal/inference.py
```

## Benchmark History

The table below is intended to grow as inference performance improves.

| Date | Model | GPU | Prompt Tokens | Generated Tokens | Generation Time | Throughput | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-30 | `Qwen/Qwen3.5-9B` | `NVIDIA B200` | `13` | `150` | `9.28 s` | `16.16 tokens/s` | Current correctness baseline |

## Current Output Snapshot

Latest successful sample output:

Prompt:

```text
The architecture of the Blackwell B200 GPU allows for
```

Completion:

```text
a significant increase in the number of parameters, which is a significant improvement over the previous generation. This is achieved through a combination of architectural and architectural changes. The first of which is a significant improvement in the architecture of the model.
```

## Notes

- First-run startup is dominated by checkpoint download and model construction.
- The current implementation is correctness-first, not speed-first.
- Weights are cached in a Modal volume at `/root/.cache/huggingface`.
- The current benchmark reflects end-to-end generation time after model load, not an isolated decode-kernel benchmark.

## Optimization Backlog

Use this table to track inference-speed work as the implementation evolves.

| Idea | Why It Matters | Expected Impact | Status | Notes |
| --- | --- | --- | --- | --- |
| Add fused DeltaNet kernels | Most layers are linear-attention layers; fused kernels should reduce Python and memory overhead. | High | Planned | Replace the pure PyTorch fallback with optimized kernels. |
| Add fused causal conv1d | The linear-attention path currently uses a plain PyTorch conv fallback. | Medium | Planned | Likely pairs naturally with fused DeltaNet work. |
| Reuse initialized model across requests | Current cold-start path rebuilds and reloads the model inside the function. | High | Planned | Convert to a long-lived container lifecycle if request pattern justifies it. |
| Avoid repeated tokenizer/config network checks | Tokenizer setup still triggers extra Hub metadata calls. | Low | Planned | Cache tokenizer artifacts more aggressively. |
| Measure prefill vs decode separately | Current throughput is aggregate only. | Medium | Planned | Add separate timings for model load, prefill, and token decode loop. |
| Tune decode loop for single-token latency | Decode dominates interactive inference. | High | Planned | Focus on recurrent path, cache layout, and kernel launches. |
| Explore `torch.compile` or graph capture | May reduce Python dispatch overhead on stable decode shapes. | Medium | Planned | Needs measurement on B200 with current model structure. |
| Increase sequence/cache efficiency | Current max sequence cap is conservative and static. | Low | Planned | Review cache allocation and memory reuse strategy. |

## Known Gaps

- No apples-to-apples benchmark yet against `transformers`, vLLM, or TensorRT-LLM on the same hardware.
- No separate reporting for cold start, model load, prefill, and decode latency.
- No batching, speculative decoding, or fused-kernel path yet.

## Next Measurements

When testing a new optimization, record:

- date
- exact prompt length
- generated token count
- total generation time
- tokens/sec
- GPU type
- cold start vs warm start

import argparse
import sys
from pathlib import Path

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.triton.fused_zero_centered_rmsnorm import FusedZeroCenteredRMSNorm
from qwen import Qwen3_5RMSNorm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3_5RMSNorm against FusedZeroCenteredRMSNorm."
    )
    parser.add_argument("--rows", type=int, default=4096, help="Number of rows/tokens to benchmark.")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
        help="Input/weight dtype.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device to benchmark on.")
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon.")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Hidden sizes to benchmark.",
    )
    parser.add_argument(
        "--line-values",
        nargs="+",
        default=["qwen_rmsnorm", "fused_zero_centered_rmsnorm"],
        choices=("qwen_rmsnorm", "fused_zero_centered_rmsnorm"),
        help="Providers to include in the benchmark report.",
    )
    parser.add_argument(
        "--correctness-hidden-size",
        type=int,
        default=4096,
        help="Hidden size used for the pre-benchmark correctness check.",
    )
    return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype_name]


class QwenZeroCenteredRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = Qwen3_5RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summed = x + residual
        return self.norm(summed), summed


def _build_modules(hidden_size: int, eps: float, dtype: torch.dtype, device: str):
    qwen_module = QwenZeroCenteredRMSNorm(hidden_size, eps).to(device=device, dtype=dtype)
    fused_module = FusedZeroCenteredRMSNorm(hidden_size, eps).to(device=device, dtype=dtype)

    with torch.no_grad():
        fused_module.weights.copy_(qwen_module.norm.weight)

    return qwen_module, fused_module


def _correctness_check(
    rows: int,
    hidden_size: int,
    eps: float,
    dtype: torch.dtype,
    device: str,
) -> None:
    qwen_module, fused_module = _build_modules(hidden_size, eps, dtype, device)
    x = torch.randn((rows, hidden_size), device=device, dtype=dtype)
    residual = torch.randn((rows, hidden_size), device=device, dtype=dtype)

    with torch.no_grad():
        qwen_y, qwen_s = qwen_module(x, residual)
        fused_y, fused_s = fused_module(x, residual)

    atol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    if not torch.allclose(qwen_s, fused_s, atol=atol, rtol=rtol):
        raise AssertionError("Residual outputs differ between Qwen3_5RMSNorm baseline and fused kernel.")
    if not torch.allclose(qwen_y, fused_y, atol=atol, rtol=rtol):
        raise AssertionError("Normalized outputs differ between Qwen3_5RMSNorm baseline and fused kernel.")


def main() -> None:
    args = _parse_args()
    dtype = _torch_dtype(args.dtype)
    provider_names = {
        "qwen_rmsnorm": "Qwen3_5RMSNorm",
        "fused_zero_centered_rmsnorm": "FusedZeroCenteredRMSNorm",
    }

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark.")

    _correctness_check(
        rows=min(args.rows, 512),
        hidden_size=args.correctness_hidden_size,
        eps=args.eps,
        dtype=dtype,
        device=args.device,
    )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["hidden_size"],
            x_vals=args.hidden_sizes,
            line_arg="provider",
            line_vals=args.line_values,
            line_names=[provider_names[provider] for provider in args.line_values],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Latency (ms)",
            plot_name=f"qwen-rmsnorm-vs-fused-zero-centered-rmsnorm-rows-{args.rows}-dtype-{args.dtype}",
            args={},
        )
    )
    def benchmark(hidden_size: int, provider: str):
        qwen_module, fused_module = _build_modules(hidden_size, args.eps, dtype, args.device)
        x = torch.randn((args.rows, hidden_size), device=args.device, dtype=dtype)
        residual = torch.randn((args.rows, hidden_size), device=args.device, dtype=dtype)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "qwen_rmsnorm":
            fn = lambda: qwen_module(x, residual)
        elif provider == "fused_zero_centered_rmsnorm":
            fn = lambda: fused_module(x, residual)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        return ms, max_ms, min_ms

    benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    main()

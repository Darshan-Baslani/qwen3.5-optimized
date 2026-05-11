import triton
import triton.language as tl
import torch
from torch import nn


@triton.jit
def _fused_zero_centered_rmsnorm(
    Y_ptr,
    Y_row_stride,
    S_ptr, # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr, # input residual
    R_row_stride,
    W_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    S_ptr += row_idx * S_row_stride
    X_ptr += row_idx * X_row_stride
    R_ptr += row_idx * R_row_stride

    # residual operations
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + col_offsets, mask=mask, other=0)

    S_row = X_row + R_row

    tl.store(S_ptr + col_offsets, S_row, mask=mask)

    S_row_dtype = S_row.dtype

    # rmsnorm operations
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    S_row = S_row.to(tl.float32)
    W_row = W_row.to(tl.float32)

    mean_square = tl.sum(S_row * S_row, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)

    S_row = S_row * rstd
    Y_row = S_row * (1.0 + W_row)

    Y_row = Y_row.to(S_row_dtype)
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


def fused_zero_centered_rmsnorm(X, R, W, eps):
    shape = X.shape

    # flatning X and R to 2D
    dim = shape[-1]
    X = X.reshape(-1, dim)
    R = R.reshape(-1, dim)

    n_rows, n_cols = X.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    num_warps = 4
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    S = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)

    _fused_zero_centered_rmsnorm[(n_rows,)](
        Y,
        Y.stride(0),
        S,
        S.stride(0),
        X,
        X.stride(0),
        R,
        R.stride(0),
        W,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return Y.view(*shape), S.view(*shape)


class FusedZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, X, R):
        Y, S = fused_zero_centered_rmsnorm(X, R, self.weight , self.eps)
        return Y, S

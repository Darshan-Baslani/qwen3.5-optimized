import math

import torch
import triton
import triton.language as tl
from tvm_ffi import register_global_func as register_func

NUM_Q_HEADS: int = 4
NUM_K_HEADS: int = 4
NUM_V_HEADS: int = 8
HEAD_DIM: int = 128
GVA_RATIO: int = NUM_V_HEADS // NUM_Q_HEADS
BV: int = 8
N_V_TILES: int = HEAD_DIM // BV


def _as_torch_tensor(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    return torch.from_dlpack(tensor)


def _gdn_decode_impl(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    q = _as_torch_tensor(q)
    k = _as_torch_tensor(k)
    v = _as_torch_tensor(v)
    state = _as_torch_tensor(state)
    A_log = _as_torch_tensor(A_log)
    a = _as_torch_tensor(a)
    dt_bias = _as_torch_tensor(dt_bias)
    b = _as_torch_tensor(b)

    B = q.shape[0]
    num_k_heads = int(k.shape[1])
    num_v_heads = int(v.shape[1])
    head_dim = int(k.shape[2])
    value_dim = int(v.shape[2])
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Expected value heads to be divisible by key heads, got {num_v_heads=} {num_k_heads=}")
    if q.shape[1] != num_k_heads or q.shape[2] != head_dim:
        raise ValueError(f"Expected q and k to have matching head layout, got q={tuple(q.shape)} k={tuple(k.shape)}")
    gva_ratio = num_v_heads // num_k_heads
    bv = 8
    n_v_tiles = value_dim // bv

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_dim)

    if output is None:
        output = torch.empty(
            (B, num_v_heads, value_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        output = _as_torch_tensor(output)

    if new_state is None:
        new_state = torch.empty_like(state)
    else:
        new_state = _as_torch_tensor(new_state)

    grid = (B, num_v_heads * n_v_tiles)

    gdn_decode_kernel[grid](
        q, k, v, state, A_log, a, dt_bias, b,
        output, new_state,
        scale,
        K=head_dim, V_DIM=value_dim,
        NUM_V_HEADS=num_v_heads, NUM_K_HEADS=num_k_heads,
        GVA_RATIO=gva_ratio, BV=bv, N_V_TILES=n_v_tiles,
        num_warps=8, num_stages=4,
    )
    return output, new_state


register_func("flashinfer.gdn_decode", _gdn_decode_impl, override=True)
gdn_decode = _gdn_decode_impl


@triton.jit
def gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    out_ptr, new_state_ptr,
    scale,
    K: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    GVA_RATIO: tl.constexpr,
    BV: tl.constexpr,
    N_V_TILES: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // N_V_TILES
    pid_v = pid_hv % N_V_TILES

    qk_head = pid_h // GVA_RATIO
    i_nh = pid_b * NUM_V_HEADS + pid_h

    # Gate computation
    a_val = tl.load(a_ptr + i_nh).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    b_val = tl.load(b_ptr + i_nh).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * softplus_x)
    beta = tl.sigmoid(b_val)

    # Load q, k
    o_k = tl.arange(0, K)
    qk_base = pid_b * (NUM_K_HEADS * K) + qk_head * K
    b_q = tl.load(q_ptr + qk_base + o_k).to(tl.float32)
    b_k = tl.load(k_ptr + qk_base + o_k).to(tl.float32)

    # V-tile
    v_start = pid_v * BV
    o_v = tl.arange(0, BV)

    # Load state [BV, K] via plain pointers (row-major, K stride 1)
    s_base = state_ptr + i_nh * V_DIM * K + v_start * K
    s_ptrs = s_base + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(s_ptrs).to(tl.float32)

    # Decay
    old_state = g * b_h

    # old_v = k @ old_state per V-row
    old_v = tl.sum(old_state * b_k[None, :], axis=1)

    # Load value
    v_base = v_ptr + i_nh * V_DIM
    b_v = tl.load(v_base + v_start + o_v).to(tl.float32)

    # Compact delta (Q3: reduces peak live registers)
    delta_v = beta * (b_v - old_v)

    # Output via identity (Q7: avoids state_out live during reduction)
    # output = scale * (old_state@q + delta_v * dot(k,q))
    old_o = tl.sum(old_state * b_q[None, :], axis=1)
    kq = tl.sum(b_k * b_q)
    b_o = scale * (old_o + delta_v * kq)

    # Store output BEFORE building state_out (frees old_o registers)
    out_base = out_ptr + i_nh * V_DIM
    tl.store(out_base + v_start + o_v, b_o.to(tl.bfloat16))

    # State update (state_out only needed for store, not output)
    state_out = old_state + delta_v[:, None] * b_k[None, :]

    # Store state
    ns_base = new_state_ptr + i_nh * V_DIM * K + v_start * K
    ns_ptrs = ns_base + o_v[:, None] * K + o_k[None, :]
    tl.store(ns_ptrs, state_out)

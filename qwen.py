from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List

import torch
import torch.nn.functional as F
from torch import chunk, nn

from kernels.triton.fused_zero_centered_rmsnorm import FusedZeroCenteredRMSNorm
# from kernels.FlashQLA.flash_qla import chunk_gated_delta_rule
from kernels.linear_attention.decode import gdn_decode
from kernels.linear_attention.prefill import gdn_prefill



def _apply_activation(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return x
    if activation == "silu":
        return F.silu(x)
    raise ValueError(f"Unsupported activation: {activation}")


def pytorch_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx=None,
) -> torch.Tensor:
    del seq_idx
    conv_weight = weight.unsqueeze(1)
    conv_input = F.pad(x, (weight.shape[-1] - 1, 0))
    y = F.conv1d(conv_input, conv_weight, bias=bias, groups=x.shape[1])
    return _apply_activation(y, activation)


def pytorch_causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    if x.shape[-1] != 1:
        raise ValueError(f"Expected a single decode token, got shape {x.shape}")

    if cache is None:
        conv_input = F.pad(x, (weight.shape[-1] - 1, 0))
    else:
        conv_input = torch.cat((cache, x), dim=-1)
        cache.copy_(conv_input[:, :, -cache.shape[-1]:])

    y = F.conv1d(conv_input, weight.unsqueeze(1), bias=bias, groups=x.shape[1])
    return _apply_activation(y, activation)


def _l2_normalize_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)


def pytorch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
):
    del initial_state
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = _l2_normalize_last_dim(query, eps=1e-6)
        key = _l2_normalize_last_dim(key, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    causal_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(causal_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    core_attn_out = torch.zeros_like(value)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def pytorch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
):
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        query = _l2_normalize_last_dim(query, eps=1e-6)
        key = _l2_normalize_last_dim(key, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q and k shapes: [batch, seq_len, num_heads, head_dim]
    # cos and sin shapes: [batch, seq_len, rotary_dim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat((q_embed, q_pass), dim=-1)
    k_embed = torch.cat((k_embed, k_pass), dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seq_len, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq_len, num_key_value_heads * n_rep, head_dim)


@dataclass
class Qwen3_5TextConfig:
    # Core Dimensions
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    
    # Standard Attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    
    # Linear Attention (DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    
    # Normalization & Activations
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    
    # RoPE
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0 
    partial_rotary_factor: float = 1.0
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    
    # Hybrid Architecture Routing
    full_attention_interval: int = 4
    layer_types: List[str] = field(default_factory=list)
    pad_token_id: int | None = None

    def __post_init__(self):
        # Qwen 3.5 mixes Linear and Standard Attention.
        # This generates the array telling the model which layer is which.
        if not self.layer_types:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.full_attention_interval) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        cache_position: int,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        seq_len = hidden_states.shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # static cache update
        k_cache[:, cache_position : cache_position + seq_len, :, :] = key_states
        v_cache[:, cache_position : cache_position + seq_len, :, :] = value_states

        # slice the tensors to attend to all past + current tokens 
        keys_to_attend = k_cache[:, : cache_position + seq_len, :, :]
        values_to_attend = v_cache[:, : cache_position + seq_len, :, :]

        keys_to_attend = repeat_kv(keys_to_attend, self.num_key_value_groups)
        values_to_attend = repeat_kv(values_to_attend, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2)
        keys_to_attend = keys_to_attend.transpose(1, 2)
        values_to_attend = values_to_attend.transpose(1, 2)

        # With a KV cache, the decode step only passes past/current tokens in
        # `keys_to_attend`. For q_len=1 and kv_len>1, PyTorch's rectangular
        # `is_causal=True` mask aligns to the upper-left corner and would let
        # the token attend only to key 0. Keep causal masking for prompt
        # prefill, but disable it for cached single-token decode.
        use_causal_mask = attention_mask is None and seq_len > 1
        attn_output = F.scaled_dot_product_attention(
            query_states,
            keys_to_attend,
            values_to_attend,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=use_causal_mask,
            scale=self.scaling,
        )
        attn_weights = None
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = F.silu
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = pytorch_causal_conv1d_fn
        self.causal_conv1d_update = pytorch_causal_conv1d_update
        self.chunk_gated_delta_rule = gdn_prefill
        self.recurrent_gated_delta_rule = gdn_decode

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_conv_cache: torch.Tensor | None = None,       # Replaces k_cache
        layer_recurrent_cache: torch.Tensor | None = None,  # Replaces v_cache
        cache_position: int = 0,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Phase Detection: Are we decoding a single token, or prefilling a prompt?
        is_decode = (cache_position > 0) and (seq_len == 1)

        # 2. Input Projections
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # STEP 1: CONVOLUTION (Temporal mixing over a short window)
        if is_decode:
            # DECODE PHASE: Update the static conv cache in-place
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                layer_conv_cache,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # PREFILL PHASE: Process whole sequence, then save the tail to cache
            if layer_conv_cache is not None:
                # Pad left by kernel_size - 1 to prevent looking into the future on token 0
                conv_input = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
                
                # Save the last `kernel_size - 1` elements into the static cache
                layer_conv_cache.copy_(conv_input[:, :, -self.conv_kernel_size + 1:])
                
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )

        # STEP 2: PREPARE Q, K, V
        # FlashQLA kernels require the per-head feature dimension to be unit-stride.
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # STEP 3: RECURRENT DELTA RULE (The core SSM math)
        query = _l2_normalize_last_dim(query, eps=1e-6)
        key = _l2_normalize_last_dim(key, eps=1e-6)

        if not is_decode:
            # PREFILL: flatten packed tokens and build uniform cu_seqlens.
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                seq_len,
                dtype=torch.int32,
                device=query.device,
            )
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query.reshape(-1, query.shape[-2], query.shape[-1]).contiguous(),
                key.reshape(-1, key.shape[-2], key.shape[-1]).contiguous(),
                value.reshape(-1, value.shape[-2], value.shape[-1]).contiguous(),
                layer_recurrent_cache,
                self.A_log,
                a.reshape(-1, a.shape[-1]).contiguous(),
                self.dt_bias,
                b.reshape(-1, b.shape[-1]).contiguous(),
                cu_seqlens,
                None,
                None,
                None,
            )
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        else:
            # DECODE: process one token per sequence against the recurrent cache.
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query[:, 0].contiguous(),
                key[:, 0].contiguous(),
                value[:, 0].contiguous(),
                layer_recurrent_cache,
                self.A_log,
                a[:, 0].contiguous(),
                self.dt_bias,
                b[:, 0].contiguous(),
                None,
                None,
                None,
            )
            core_attn_out = core_attn_out.unsqueeze(1)

        # Update the static recurrent cache in-place for the next token
        if layer_recurrent_cache is not None:
            layer_recurrent_cache.copy_(last_recurrent_state)

        # STEP 4: OUTPUT PROJECTION
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out.to(dtype=self.out_proj.weight.dtype))
        return output


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm_standard = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_fused = FusedZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FusedZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache_position: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        conv_cache: torch.Tensor | None = None,
        recurrent_cache: torch.Tensor | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm_standard(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm_fused(X=hidden_states, R=residual)

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                layer_conv_cache=conv_cache,
                layer_recurrent_cache=recurrent_cache,
                cache_position=cache_position,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                k_cache=k_cache,
                v_cache=v_cache,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(X=hidden_states, R=residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base = getattr(config, "rope_theta", 1000000.0)
        self.rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        # hidden_states: [batch, seq_len, dim]
        # position_ids: [batch, seq_len]
        freqs = torch.outer(position_ids.squeeze(0).float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)

        cos = emb.cos().to(dtype=hidden_states.dtype)
        sin = emb.sin().to(dtype=hidden_states.dtype)

        return cos, sin


class Qwen3_5TextModel(nn.Module):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = FusedZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache_position: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        conv_cache: torch.Tensor | None = None,
        recurrent_cache: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
    ):
        del use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        seq_len = inputs_embeds.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                cache_position, cache_position + seq_len,
                dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        residual = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.config.layer_types[i] == "linear_attention":
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    conv_cache=conv_cache[:, i] if conv_cache is not None else None,
                    recurrent_cache=recurrent_cache[:, i] if recurrent_cache is not None else None,
                )
            else:
                hidden_states, residual = decoder_layer(
                    hidden_states,
                    residual,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    k_cache=k_cache[:, i] if k_cache is not None else None,
                    v_cache=v_cache[:, i] if v_cache is not None else None,
                )

        hidden_states, residual = self.norm(X=hidden_states, R=residual)
        
        return hidden_states

class Qwen3_5ForCausalLM(nn.Module):
    config: Qwen3_5TextConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        k_cache = None,
        v_cache = None,
        cache_position = None,
        conv_cache = None,
        recurrent_cache = None,
        inputs_embeds: torch.FloatTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        >>> model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3_5-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3_5-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_position=cache_position,
            conv_cache=conv_cache,
            recurrent_cache=recurrent_cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        logits = self.lm_head(hidden_states[:, -1:, :])

        return logits

def generate(model, input_ids, max_new_tokens, max_seq_len):
    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    config = model.config
    model_dtype = model.lm_head.weight.dtype

    # allocation phase
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // model.config.num_attention_heads

    k_cache = torch.zeros(
        batch_size, num_layers, max_seq_len, num_kv_heads, head_dim, 
        dtype=model_dtype, device=device
    )
    v_cache = torch.zeros(
        batch_size, num_layers, max_seq_len, num_kv_heads, head_dim, 
        dtype=model_dtype, device=device
    )

    # the tiny SSM Caches (for the DeltaNet layers)
    conv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2 + \
           config.linear_num_value_heads * config.linear_value_head_dim

    conv_cache = torch.zeros(
        batch_size, num_layers, conv_dim, config.linear_conv_kernel_dim - 1,
        dtype=model_dtype, device=device
    )

    recurrent_cache = torch.zeros(
        batch_size, num_layers, config.linear_num_value_heads, 
        config.linear_key_head_dim, config.linear_value_head_dim,
        dtype=torch.float32, device=device # SSM state usually needs FP32 precision
    )


    # prefill phase
    cache_position = 0
    generated_tokens = []

    if max_new_tokens <= 0:
        return input_ids[:, :0]

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_position=cache_position,
            conv_cache=conv_cache,
            recurrent_cache=recurrent_cache,
        )

        # get the logit of the last token of hidden_states ~ first newly generated token
        new_token = torch.argmax(logits[:, -1:, :], dim=-1)

        generated_tokens.append(new_token)
        cache_position += prompt_len

    for _ in range(max_new_tokens - 1):
        # OOM protection
        if cache_position >= max_seq_len:
            break

        with torch.no_grad():
            # we only pass the first token here
            logits = model(
                    input_ids=new_token,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    conv_cache=conv_cache,
                    recurrent_cache=recurrent_cache,
                    cache_position=cache_position,
            )

            new_token = torch.argmax(logits[:, -1:, :], dim=-1)

            generated_tokens.append(new_token)
            cache_position += 1

    return torch.cat(generated_tokens, dim=-1)


def generate_profiled(model, input_ids, max_new_tokens, max_seq_len, profile_decode_tokens=None, profiler=None):
    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    config = model.config
    model_dtype = model.lm_head.weight.dtype

    if profile_decode_tokens is None:
        profile_decode_tokens = max_new_tokens
    profile_decode_tokens = max(0, min(profile_decode_tokens, max_new_tokens - 1))

    # allocation phase
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // model.config.num_attention_heads

    k_cache = torch.zeros(
        batch_size, num_layers, max_seq_len, num_kv_heads, head_dim,
        dtype=model_dtype, device=device
    )
    v_cache = torch.zeros(
        batch_size, num_layers, max_seq_len, num_kv_heads, head_dim,
        dtype=model_dtype, device=device
    )

    conv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2 + \
           config.linear_num_value_heads * config.linear_value_head_dim

    conv_cache = torch.zeros(
        batch_size, num_layers, conv_dim, config.linear_conv_kernel_dim - 1,
        dtype=model_dtype, device=device
    )

    recurrent_cache = torch.zeros(
        batch_size, num_layers, config.linear_num_value_heads,
        config.linear_key_head_dim, config.linear_value_head_dim,
        dtype=torch.float32, device=device
    )

    cache_position = 0
    generated_tokens = []

    if max_new_tokens <= 0:
        return input_ids[:, :0]

    profiler_stopped = profiler is None

    with torch.no_grad():
        with torch.profiler.record_function("prefill"):
            logits = model(
                input_ids=input_ids,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_position=cache_position,
                conv_cache=conv_cache,
                recurrent_cache=recurrent_cache,
            )

            new_token = torch.argmax(logits[:, -1:, :], dim=-1)
            generated_tokens.append(new_token)
            cache_position += prompt_len

        for decode_step in range(max_new_tokens - 1):
            if cache_position >= max_seq_len:
                break

            if not profiler_stopped and decode_step >= profile_decode_tokens:
                torch.cuda.synchronize(device)
                profiler.stop()
                profiler_stopped = True

            if decode_step < profile_decode_tokens:
                record_context = torch.profiler.record_function(f"decode_step_{decode_step + 1}")
            else:
                record_context = nullcontext()

            with record_context:
                logits = model(
                    input_ids=new_token,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    conv_cache=conv_cache,
                    recurrent_cache=recurrent_cache,
                    cache_position=cache_position,
                )

                new_token = torch.argmax(logits[:, -1:, :], dim=-1)
                generated_tokens.append(new_token)
                cache_position += 1

    if not profiler_stopped:
        torch.cuda.synchronize(device)
        profiler.stop()

    return torch.cat(generated_tokens, dim=-1)

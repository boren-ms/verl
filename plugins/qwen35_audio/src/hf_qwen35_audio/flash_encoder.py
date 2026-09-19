# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Voxtral-style LLM Transformer Audio Encoder.

Architecture:
    Frontend:  LightPatchEncoder (8x subsampling via Conv2d + SwiGLU)
    Encoder:   N x Transformer Block (RMSNorm, RoPE Self-Attention, SwiGLU FFN)
    Optional:  Sliding window attention for streaming

The encoder uses the same building blocks as modern LLMs (RoPE, RMSNorm, SwiGLU,
GQA), making it natively compatible with flash attention and packing SFT.

References:
    - Voxtral Realtime (arxiv:2602.11298): causal audio encoder with sliding window
    - Qwen3-ASR: transformer encoder with flash attention + cu_seqlens packing
"""

import math
import inspect
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint

from collections import OrderedDict

# Try to import flash_attn for optimal performance
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    _flash_attn_available = True
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
except ImportError:
    _flash_attn_available = False
    _flash_supports_window_size = False

# Fused kernels (optional, toggled by use_fused_kernels config)
try:
    from flash_attn.layers.rotary import apply_rotary_emb as _fused_apply_rotary_emb
    _fused_rotary_available = True
except ImportError:
    _fused_rotary_available = False

try:
    from flash_attn.ops.activations import swiglu as _fused_swiglu
    _fused_swiglu_available = True
except ImportError:
    _fused_swiglu_available = False

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn as _fused_rms_norm_fn
    _fused_rms_norm_available = True
except ImportError:
    _fused_rms_norm_available = False

try:
    from flash_attn.ops.fused_dense import FusedDense as _FusedDense
    _fused_dense_available = True
except ImportError:
    _fused_dense_available = False


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────

ACT2FN = {
    "silu": F.silu,
}


def _swiglu(x, activation="silu"):
    x, gate = x.chunk(2, dim=-1)
    return x * ACT2FN[activation](gate)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ──────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding (RoPE)
# ──────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Standard RoPE with configurable base frequency."""

    def __init__(self, dim: int, max_position_embeddings: int = 131072, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor):
        """
        Args:
            x: input tensor, used only for dtype/device.
            position_ids: (batch_size, seq_len) or (seq_len,)
        Returns:
            cos, sin: (batch_size, seq_len, dim) or (1, seq_len, dim)
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        # (batch, seq_len, 1) @ (1, 1, dim//2) -> (batch, seq_len, dim//2)
        inv_freq = self.inv_freq[None, None, :].float().expand(position_ids.shape[0], -1, -1)
        position_ids = position_ids[:, :, None].float()
        freqs = (position_ids @ inv_freq)  # (batch, seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (batch, seq_len, dim)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin

    @torch.no_grad()
    def forward_fused(self, x: Tensor, position_ids: Tensor):
        """Return cos/sin in half-dim format for fused RoPE kernel.

        Returns:
            cos, sin: (seqlen, dim//2) — no batch dim, half-dim (not duplicated).
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        inv_freq = self.inv_freq[None, None, :].float().expand(position_ids.shape[0], -1, -1)
        position_ids = position_ids[:, :, None].float()
        freqs = (position_ids @ inv_freq)  # (1, seq_len, dim//2)
        cos = freqs.squeeze(0).cos().to(dtype=x.dtype)  # (seq_len, dim//2)
        sin = freqs.squeeze(0).sin().to(dtype=x.dtype)
        return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1) -> tuple:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (batch, num_heads, seq_len, head_dim) or (batch, seq_len, num_heads, head_dim)
        cos, sin: (batch, seq_len, head_dim) or (1, seq_len, head_dim)
        unsqueeze_dim: 1 for (batch, heads, seq, dim), 2 for (batch, seq, heads, dim)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ──────────────────────────────────────────────────────────────────────────────
# LightPatch Frontend (8x subsampling)
# ──────────────────────────────────────────────────────────────────────────────

class LightPatchEncoder(nn.Module):
    """Light Patch Encoder for Speech Input — 8x temporal subsampling via Conv2d + SwiGLU."""

    def __init__(
        self,
        input_dim: int = 80,
        window_size: int = 8,
        activation: str = "swiglu",
        layers: list = None,
    ):
        super().__init__()
        assert activation == "swiglu", "Only 'swiglu' activation is supported for LightPatchEncoder"
        self.input_dim = input_dim
        self.window_size = window_size

        if layers is None:
            layers = []

        conv_layers = []
        ln_layers = []

        in_dim = self.input_dim
        for i, layer in enumerate(layers):
            out_dim = layer.get("out_dim", in_dim) * 2

            kernel_size = tuple(layer.get("kernel_size", (3, 3)))
            stride = tuple(layer.get("stride", (1, 1)))
            padding = tuple(layer.get("padding", (1, 1)))
            conv_layer = nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            conv_layers.append((f"conv_{i}", conv_layer))

            ln_layer = RMSNorm(out_dim)
            ln_layers.append((f"ln_{i}", ln_layer))
            in_dim = layer.get("out_dim", in_dim)

        self.conv_layers = nn.ModuleDict(OrderedDict(conv_layers))
        self.ln_layers = nn.ModuleDict(OrderedDict(ln_layers))
        # output_dim accounts for any remaining spatial dimensions that aren't collapsed
        self.output_dim = in_dim

    def chunk_input(self, x: Tensor, chunk_size: int) -> Tensor:
        batch_size, seq_len, _ = x.size()
        num_chunks = math.ceil(seq_len / chunk_size)
        padded_seq_len = num_chunks * chunk_size
        padding = padded_seq_len - seq_len
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding), value=0)
        x = x.view(batch_size, num_chunks, chunk_size, -1)
        return x

    def forward(self, mel: Tensor, mask: Optional[Tensor] = None):
        """
        Args:
            mel: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) or (batch_size, 1, seq_len)
        Returns:
            x: (batch_size, num_chunks, out_dim)
            mask: updated mask (batch_size, 1, num_chunks) or None
        """
        x = self.chunk_input(mel, self.window_size)
        batch_size, num_chunks, chunk_size, _ = x.size()
        x = x.view(batch_size * num_chunks, chunk_size, -1).unsqueeze(1)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[f"conv_{i}"](x.permute(0, 3, 1, 2))
            x = self.ln_layers[f"ln_{i}"](x.permute(0, 2, 3, 1))
            x = _swiglu(x)

        # x shape: (batch*chunks, 1, remaining_time, out_dim) or (batch*chunks, 1, 1, out_dim)
        # Flatten all spatial dims into feature dim
        x = x.reshape(batch_size * num_chunks, -1)
        x = x.view(batch_size, num_chunks, -1)  # (batch, num_chunks, flattened_dim)

        # Update mask for subsampled sequence
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            # Each chunk of window_size frames -> 1 output frame
            # A chunk is valid if any frame in it is unmasked
            mask_chunked = self.chunk_input(mask.unsqueeze(-1).float(), self.window_size)
            mask = (mask_chunked.squeeze(-1).sum(-1) > 0).unsqueeze(1)  # (batch, 1, num_chunks)

        return x, mask


class TemporalConvFrontend(nn.Module):
    """Full-sequence temporal Conv1D frontend for speech input.

    Unlike LightPatchEncoder, this applies convolution over the entire time
    sequence before subsampling, which avoids hard non-overlapping chunk
    boundaries. This is closer to Whisper/Voxtral-style audio frontends.
    """

    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 1024,
        factors: Optional[List[int]] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if factors is None:
            factors = [2]
        assert all(factor > 0 for factor in factors), "frontend_downsample_factors must be positive"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factors = factors
        self.downsample_rate = math.prod(factors)

        layers = []
        norms = []
        in_dim = input_dim
        for i, factor in enumerate(factors):
            out_dim = output_dim
            padding = (kernel_size - 1) // 2
            layers.append((
                f"conv_{i}",
                nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=factor, padding=padding),
            ))
            norms.append((f"ln_{i}", RMSNorm(out_dim)))
            in_dim = out_dim

        self.conv_layers = nn.ModuleDict(OrderedDict(layers))
        self.ln_layers = nn.ModuleDict(OrderedDict(norms))
        self.kernel_size = kernel_size

    @staticmethod
    def _factorize_window_size(window_size: int) -> List[int]:
        factors = []
        remaining = window_size
        while remaining > 1 and remaining % 2 == 0:
            factors.append(2)
            remaining //= 2
        if remaining > 1:
            factors.append(remaining)
        return factors or [1]

    def forward(self, mel: Tensor, mask: Optional[Tensor] = None):
        x = mel.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            mask = mask.float().unsqueeze(1)

        for i in range(len(self.conv_layers)):
            conv = self.conv_layers[f"conv_{i}"]
            x = conv(x)
            x = self.ln_layers[f"ln_{i}"](x.transpose(1, 2)).transpose(1, 2)
            x = F.gelu(x)

            if mask is not None:
                mask = F.max_pool1d(
                    mask,
                    kernel_size=conv.kernel_size[0],
                    stride=conv.stride[0],
                    padding=conv.padding[0],
                )

        x = x.transpose(1, 2)
        if mask is not None:
            mask = (mask > 0).unsqueeze(1).squeeze(2)
        return x, mask


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Layer
# ──────────────────────────────────────────────────────────────────────────────

class LocalConvModule(nn.Module):
    """Same-length local Conv1D module for acoustic detail modeling."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 31,
        causal: bool = False,
        use_glu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size > 0, "local_conv_kernel_size must be positive"
        self.kernel_size = kernel_size
        self.causal = causal

        self.in_proj = nn.Linear(hidden_size, hidden_size * 2) if use_glu else None
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=True,
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size) if use_glu else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def _pad(self, hidden_states: Tensor) -> Tensor:
        if self.causal:
            return F.pad(hidden_states, (self.kernel_size - 1, 0))
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size // 2
        return F.pad(hidden_states, (left, right))

    def _forward_padded(self, hidden_states: Tensor) -> Tensor:
        if self.in_proj is not None:
            hidden_states, gate = self.in_proj(hidden_states).chunk(2, dim=-1)
            hidden_states = hidden_states * F.silu(gate)

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.depthwise_conv(self._pad(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.out_proj(hidden_states)
        return self.dropout(hidden_states)

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        if cu_seqlens is None:
            return self._forward_padded(hidden_states)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        num_seqs = lengths.shape[0]
        if max_seqlen is None:
            max_seqlen = lengths.max().item()

        positions = torch.arange(max_seqlen, device=hidden_states.device)
        mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        padded = hidden_states.new_zeros(num_seqs, max_seqlen, hidden_states.shape[-1])
        padded[mask] = hidden_states
        padded = self._forward_padded(padded)
        return padded[mask]

class TransformerEncoderLayer(nn.Module):
    """Single LLM-style transformer encoder layer.

    Pre-norm architecture:
        x -> RMSNorm -> Self-Attention (RoPE + optional GQA + optional sliding window) -> + residual
        x -> RMSNorm -> FFN -> + residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_key_value_heads: Optional[int] = None,
        sliding_window: Optional[int] = None,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_fused_kernels: bool = False,
        use_local_conv: bool = False,
        local_conv_kernel_size: int = 31,
        local_conv_use_glu: bool = False,
        local_conv_dropout: float = 0.0,
        ffn_activation: str = "swiglu",
        causal: bool = False,
    ):
        super().__init__()
        assert ffn_activation in ("swiglu", "gelu"), f"Unsupported ffn_activation: {ffn_activation}"
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self._causal = causal
        self._use_fused_kernels = use_fused_kernels
        self.use_local_conv = use_local_conv
        self.ffn_activation = ffn_activation

        # Choose linear layer class: FusedDense overlaps dX and dW GEMMs in backward
        _Linear = _FusedDense if (use_fused_kernels and _fused_dense_available) else nn.Linear

        # Attention projections — fused QKV when use_fused_kernels=True
        if use_fused_kernels:
            qkv_out_dim = (num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim
            self.qkv_proj = _Linear(hidden_size, qkv_out_dim, bias=False)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.qkv_proj = None
        self.o_proj = _Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # QK-Norm (Voxtral / Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        if use_local_conv:
            self.local_conv_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.local_conv = LocalConvModule(
                hidden_size=hidden_size,
                kernel_size=local_conv_kernel_size,
                causal=causal,
                use_glu=local_conv_use_glu,
                dropout=local_conv_dropout,
            )
        else:
            self.local_conv_layernorm = None
            self.local_conv = None
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # FFN — SwiGLU by default, or Whisper/Voxtral-style GELU for audio encoder ablations.
        if ffn_activation == "swiglu" and use_fused_kernels and _fused_swiglu_available:
            self.gate_up_proj = _Linear(hidden_size, intermediate_size * 2, bias=False)
            self.gate_proj = None
            self.up_proj = None
        elif ffn_activation == "swiglu":
            self.gate_proj = _Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = _Linear(hidden_size, intermediate_size, bias=False)
            self.gate_up_proj = None
        else:
            self.gate_up_proj = None
            self.gate_proj = None
            self.up_proj = _Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = _Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if ffn_activation == "swiglu" else F.gelu

    def _project_qkv(self, hidden_states: Tensor):
        """Project to Q, K, V — fused or separate."""
        shape = hidden_states.shape[:-1]
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(hidden_states)
            # Reshape to (*, total_heads, head_dim) then slice heads — zero-copy
            # flash_attn only requires stride(-1)==1, which head-dim slices preserve
            nq = self.num_attention_heads
            nkv = self.num_key_value_heads
            qkv = qkv.view(*shape, nq + 2 * nkv, self.head_dim)
            q = qkv[..., :nq, :]
            k = qkv[..., nq:nq + nkv, :]
            v = qkv[..., nq + nkv:, :]
        else:
            q = self.q_proj(hidden_states).view(*shape, self.num_attention_heads, self.head_dim)
            k = self.k_proj(hidden_states).view(*shape, self.num_key_value_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(*shape, self.num_key_value_heads, self.head_dim)
        return q, k, v

    def _rms_norm(self, norm_module, x: Tensor) -> Tensor:
        """Apply RMSNorm — fused Triton kernel or fallback."""
        if self._use_fused_kernels and _fused_rms_norm_available and not self.use_local_conv:
            return _fused_rms_norm_fn(x, norm_module.weight, None, eps=norm_module.variance_epsilon)
        return norm_module(x)

    def _attention(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        """Multi-head attention with flash_attn (preferred) or SDPA fallback.

        Args:
            hidden_states: (batch, seq_len, hidden_size) for padded mode,
                           or (total_tokens, hidden_size) for packed mode.
            cos, sin: RoPE embeddings.
            attention_mask: Only used in SDPA fallback path.
            cu_seqlens: Cumulative sequence lengths for packed mode (flash_attn_varlen_func).
            max_seqlen: Max sequence length in batch for packed mode.
        """
        is_packed = cu_seqlens is not None

        if is_packed:
            # Packed mode: hidden_states is (total_tokens, hidden_size)
            total_tokens = hidden_states.shape[0]
            query_states, key_states, value_states = self._project_qkv(hidden_states)

            # QK-Norm
            query_states = self._rms_norm(self.q_norm, query_states)
            key_states = self._rms_norm(self.k_norm, key_states)

            # Apply RoPE
            if self._use_fused_kernels and _fused_rotary_available:
                if query_states.dtype != cos.dtype:
                    query_states = query_states.to(cos.dtype)
                    key_states = key_states.to(cos.dtype)
                    value_states = value_states.to(cos.dtype)
                # Fused RoPE kernel with native cu_seqlens support — no Python loop
                # cos/sin are (max_seqlen, dim//2) from forward_fused()
                query_states = _fused_apply_rotary_emb(
                    query_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    inplace=True,
                )
                key_states = _fused_apply_rotary_emb(
                    key_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    inplace=True,
                )
            else:
                # Original: expand RoPE per-sequence using Python loop
                cos_flat = cos.squeeze(0)  # (max_seqlen, head_dim)
                sin_flat = sin.squeeze(0)
                num_seqs = cu_seqlens.shape[0] - 1
                position_cos = torch.empty_like(query_states[..., :cos_flat.shape[-1]])
                position_sin = torch.empty_like(position_cos)
                for i in range(num_seqs):
                    start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                    slen = end - start
                    position_cos[start:end] = cos_flat[:slen].unsqueeze(1)
                    position_sin[start:end] = sin_flat[:slen].unsqueeze(1)
                query_states = (query_states * position_cos) + (_rotate_half(query_states) * position_sin)
                key_states = (key_states * position_cos[:, :self.num_key_value_heads]) + (
                    _rotate_half(key_states) * position_sin[:, :self.num_key_value_heads]
                )

            # Compute window_size for flash_attn
            window_size = (-1, -1)
            if self.sliding_window is not None and _flash_supports_window_size:
                if self._causal:
                    window_size = (self.sliding_window - 1, 0)
                else:
                    half_w = self.sliding_window // 2
                    window_size = (half_w, half_w)

            dropout_p = self.attention_dropout if self.training else 0.0
            # flash_attn only supports fp16/bf16 — cast if needed
            input_dtype = query_states.dtype
            if input_dtype not in (torch.float16, torch.bfloat16):
                query_states = query_states.to(torch.bfloat16)
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=self.scaling,
                causal=self._causal,
                window_size=window_size,
            )
            attn_output = attn_output.to(input_dtype)
            attn_output = attn_output.reshape(total_tokens, -1)
            return self.o_proj(attn_output)

        # Padded mode
        batch_size, seq_len, _ = hidden_states.shape
        query_states, key_states, value_states = self._project_qkv(hidden_states)

        # QK-Norm
        query_states = self._rms_norm(self.q_norm, query_states)
        key_states = self._rms_norm(self.k_norm, key_states)

        if _flash_attn_available:
            # flash_attn_func expects (batch, seq, heads, dim) — no transpose needed
            # Apply RoPE
            if self._use_fused_kernels and _fused_rotary_available:
                if query_states.dtype != cos.dtype:
                    query_states = query_states.to(cos.dtype)
                    key_states = key_states.to(cos.dtype)
                    value_states = value_states.to(cos.dtype)
                # Fused kernel: cos/sin are (seqlen, dim//2)
                query_states = _fused_apply_rotary_emb(query_states, cos, sin, inplace=True)
                key_states = _fused_apply_rotary_emb(key_states, cos, sin, inplace=True)
            else:
                # Original: unsqueeze_dim=2 for (batch, seq, heads, dim) layout
                query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

            # flash_attn natively supports GQA — no repeat_kv needed
            window_size = (-1, -1)
            if self.sliding_window is not None and _flash_supports_window_size:
                if self._causal:
                    window_size = (self.sliding_window - 1, 0)
                else:
                    half_w = self.sliding_window // 2
                    window_size = (half_w, half_w)

            dropout_p = self.attention_dropout if self.training else 0.0
            # flash_attn only supports fp16/bf16 — cast if needed
            input_dtype = query_states.dtype
            if input_dtype not in (torch.float16, torch.bfloat16):
                query_states = query_states.to(torch.bfloat16)
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_p,
                softmax_scale=self.scaling,
                causal=self._causal,
                window_size=window_size,
            )
            # attn_output: (batch, seq, heads, dim)
            attn_output = attn_output.to(input_dtype).reshape(batch_size, seq_len, -1)
        else:
            # SDPA fallback: needs (batch, heads, seq, dim) layout
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

            # GQA: expand KV heads for SDPA (doesn't support native GQA)
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

            dropout_p = self.attention_dropout if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=self._causal and attention_mask is None,
                scale=self.scaling,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)

        return self.o_proj(attn_output)

    @staticmethod
    def _repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads for SDPA fallback (doesn't support native GQA)."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def _ffn(self, hidden_states: Tensor) -> Tensor:
        if self.ffn_activation == "gelu":
            return self.down_proj(self.act_fn(self.up_proj(hidden_states)))

        if self.gate_up_proj is not None:
            # Fused: single matmul then split for fused swiglu activation
            gate_up = self.gate_up_proj(hidden_states)
            gate, up = gate_up.chunk(2, dim=-1)
            return self.down_proj(_fused_swiglu(gate, up))
        else:
            # Original: separate gate_proj + up_proj
            return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        residual: Optional[Tensor] = None,
    ) -> Tensor:
        if self._use_fused_kernels and _fused_rms_norm_available and not self.use_local_conv:
            # ── Fused path: residual-add + RMSNorm in one kernel ──
            eps_in = self.input_layernorm.variance_epsilon
            eps_post = self.post_attention_layernorm.variance_epsilon

            # input layernorm (+ residual add from previous layer if provided)
            if residual is None:
                normed = _fused_rms_norm_fn(
                    hidden_states, self.input_layernorm.weight, None, eps=eps_in,
                )
                residual = hidden_states
            else:
                normed, residual = _fused_rms_norm_fn(
                    hidden_states, self.input_layernorm.weight, None,
                    residual=residual, prenorm=True, eps=eps_in,
                )

            # Self-Attention
            hidden_states = self._attention(normed, cos, sin, attention_mask, cu_seqlens, max_seqlen)

            # post-attention layernorm + fused residual add
            normed, residual = _fused_rms_norm_fn(
                hidden_states, self.post_attention_layernorm.weight, None,
                residual=residual, prenorm=True, eps=eps_post,
            )

            # FFN
            hidden_states = self._ffn(normed)

            # Return hidden_states (un-added FFN output) + residual for next layer
            return hidden_states, residual

        # ── Original path: separate residual-add and norm ──
        hidden_states, residual = self._forward_attn(hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen)
        hidden_states = self._forward_ffn(hidden_states, residual, cu_seqlens, max_seqlen)
        return hidden_states

    def _forward_attn(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """Attention sub-block (for selective gradient checkpointing)."""
        residual = hidden_states
        hidden_states = self._rms_norm(self.input_layernorm, hidden_states)
        hidden_states = self._attention(hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen)
        return hidden_states, residual

    def _forward_ffn(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """FFN sub-block (for selective gradient checkpointing)."""
        hidden_states = residual + hidden_states
        if self.local_conv is not None:
            residual = hidden_states
            hidden_states = self._rms_norm(self.local_conv_layernorm, hidden_states)
            hidden_states = self.local_conv(hidden_states, cu_seqlens, max_seqlen)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self._rms_norm(self.post_attention_layernorm, hidden_states)
        hidden_states = self._ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Full Transformer Audio Encoder
# ──────────────────────────────────────────────────────────────────────────────

class FlashEncoder(nn.Module):
    """Voxtral-style LLM Transformer Audio Encoder.

    Architecture:
        LightPatchEncoder (8x subsample) -> N x TransformerEncoderLayer -> LN

    Compatible interface with ConformerEncoder:
        - forward(xs_pad, masks) -> (output, masks)
        - post_init(init_model_config) for weight loading
        - gradient_checkpointing_enable()

    Args:
        input_size: Input feature dimension (e.g., 80 for mel spectrogram).
        hidden_size: Hidden dimension of the transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (for GQA). Defaults to num_attention_heads (MHA).
        intermediate_size: FFN intermediate size. Defaults to 4 * hidden_size.
        num_hidden_layers: Number of transformer layers.
        sliding_window: Sliding window size for attention (None = full attention).
        attention_dropout: Dropout rate for attention weights.
        rms_norm_eps: Epsilon for RMSNorm.
        rope_theta: Base frequency for RoPE.
        max_position_embeddings: Maximum sequence length for RoPE.
        frontend_window_size: Window size for LightPatchEncoder (subsampling factor).
        frontend_layers: Layer configs for LightPatchEncoder.
        causal: Whether to use causal attention (for streaming).
        use_fused_kernels: Use fused CUDA kernels for RoPE and SwiGLU (requires flash_attn).
        use_local_conv: Add a same-length local Conv1D block after attention in each layer.
        local_conv_kernel_size: Kernel size for the optional local Conv1D block.
        local_conv_use_glu: Whether to use a pointwise GLU before the depthwise Conv1D.
        local_conv_dropout: Dropout rate inside the optional local Conv1D block.
    """

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        ffn_activation: str = "swiglu",
        sliding_window: Optional[int] = None,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 131072,
        frontend_type: str = "lightpatch",
        frontend_window_size: int = 8,
        frontend_downsample_factors: Optional[List[int]] = None,
        frontend_layers: Optional[list] = None,
        causal: bool = False,
        use_fused_kernels: bool = False,
        use_local_conv: bool = False,
        local_conv_kernel_size: int = 31,
        local_conv_use_glu: bool = False,
        local_conv_dropout: float = 0.0,
        # Ignored kwargs for compatibility with ConformerEncoder config pattern
        **kwargs,
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.causal = causal
        self.sliding_window = sliding_window
        self.use_fused_kernels = use_fused_kernels and _flash_attn_available
        self.use_local_conv = use_local_conv
        self.ffn_activation = ffn_activation

        # Frontend: default chunked LightPatch frontend; optionally use full-sequence temporal conv.
        if frontend_layers is None:
            frontend_layers = [
                {"out_dim": hidden_size, "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1]},
            ]

        if frontend_type == "lightpatch":
            self.frontend = LightPatchEncoder(
                input_dim=input_size,
                window_size=frontend_window_size,
                activation="swiglu",
                layers=frontend_layers,
            )
            # Compute actual frontend output dim by tracing through conv layers.
            frontend_out_dim = self._compute_frontend_out_dim(input_size, frontend_window_size, frontend_layers)
        elif frontend_type == "temporal_conv":
            if frontend_downsample_factors is None:
                frontend_downsample_factors = TemporalConvFrontend._factorize_window_size(frontend_window_size)
            assert math.prod(frontend_downsample_factors) == frontend_window_size, \
                "frontend_downsample_factors product must equal frontend_window_size"
            self.frontend = TemporalConvFrontend(
                input_dim=input_size,
                output_dim=hidden_size,
                factors=frontend_downsample_factors,
            )
            frontend_out_dim = self.frontend.output_dim
        else:
            raise ValueError(f"Unsupported frontend_type: {frontend_type}")

        # Input projection: frontend output -> hidden_size
        _Linear = _FusedDense if (self.use_fused_kernels and _fused_dense_available) else nn.Linear
        self.input_proj = _Linear(frontend_out_dim, hidden_size, bias=False) if frontend_out_dim != hidden_size else nn.Identity()

        # RoPE
        head_dim = hidden_size // num_attention_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                sliding_window=sliding_window,
                attention_dropout=attention_dropout,
                rms_norm_eps=rms_norm_eps,
                use_fused_kernels=self.use_fused_kernels,
                use_local_conv=use_local_conv,
                local_conv_kernel_size=local_conv_kernel_size,
                local_conv_use_glu=local_conv_use_glu,
                local_conv_dropout=local_conv_dropout,
                ffn_activation=ffn_activation,
                causal=causal,
            )
            for _ in range(num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self._gradient_checkpointing = False
        self._gc_mode = "none"  # "none", "full", "attention_only"

    @staticmethod
    def _compute_frontend_out_dim(input_size, window_size, frontend_layers):
        """Trace through conv layer configs to compute the flattened output dim."""
        # Start with input: (1, window_size, input_size) -> Conv2d expects (N, C, H, W)
        # After permute: (N, input_size, 1, window_size)
        h, w = 1, window_size
        feat_dim = input_size
        for layer in frontend_layers:
            out_dim = layer.get("out_dim", feat_dim) * 2  # conv output channels (before SwiGLU)
            kernel_size = tuple(layer.get("kernel_size", (3, 3)))
            stride = tuple(layer.get("stride", (1, 1)))
            padding = tuple(layer.get("padding", (1, 1)))
            # Conv2d output spatial size
            h = (h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            w = (w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            feat_dim = layer.get("out_dim", feat_dim)  # after SwiGLU halving
        return h * w * feat_dim

    def post_init(self, init_model_config: dict):
        """Load pretrained weights.

        Compatible with ConformerEncoder.post_init() interface.
        """
        pretrained_path = init_model_config.get("pretrained_speech_encoder_path", None)
        if pretrained_path:
            if pretrained_path.startswith("az://"):
                import blobfile as bf
                with bf.BlobFile(pretrained_path, "rb") as f:
                    model_state = torch.load(f, map_location="cpu")
            else:
                model_state = torch.load(pretrained_path, map_location="cpu")

            if "module" in model_state:
                model_state = model_state["module"]

            encoder_state_dict = {}
            for k, v in model_state.items():
                if "encoder." in k:
                    tmp_k = k.split("encoder.", 1)[1]
                    encoder_state_dict[tmp_k] = v

            self.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")

    def gradient_checkpointing_enable(self, mode: str = "full"):
        """Enable gradient checkpointing.

        Args:
            mode: Checkpointing granularity.
                - "full": Checkpoint entire layer (max memory savings, recomputes everything).
                - "attention_only": Checkpoint only attention sub-block, FFN runs without
                  recompute. Saves most of the memory (attention activations dominate)
                  while avoiding FFN recompute cost (~40% cheaper than full).
        """
        assert mode in ("full", "attention_only"), f"Invalid gc mode: {mode}"
        self._gradient_checkpointing = True
        self._gc_mode = mode

    @staticmethod
    def _get_unpad_data(attention_mask: Tensor):
        """Compute cu_seqlens and indices for unpadding.

        Args:
            attention_mask: (batch_size, seq_len) bool mask, True = valid.

        Returns:
            indices: (total_valid_tokens,) indices of valid tokens in flattened batch.
            cu_seqlens: (batch_size + 1,) cumulative sequence lengths.
            max_seqlen: maximum sequence length in the batch.
        """
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    def _run_layers(self, hidden_states, cos, sin, attention_mask=None, cu_seqlens=None, max_seqlen=None):
        """Run all transformer layers + final norm. Handles fused residual+norm path."""
        _gc = self._gradient_checkpointing and self.training
        _cp = torch.utils.checkpoint.checkpoint

        if self.use_fused_kernels and _fused_rms_norm_available and not self.use_local_conv:
            # Fused path: layers return (hidden_states, residual)
            residual = None
            for layer in self.layers:
                if _gc:
                    if self._gc_mode == "attention_only":
                        # Checkpoint attention only — split into attn + ffn
                        # Attention sub-block (checkpointed)
                        normed, residual_new = _cp(
                            self._fused_attn_block, layer, hidden_states, cos, sin,
                            attention_mask, cu_seqlens, max_seqlen, residual,
                            use_reentrant=False,
                        )
                        # FFN sub-block (not checkpointed)
                        hidden_states = layer._ffn(normed)
                        residual = residual_new
                    else:
                        # Full layer checkpoint
                        hidden_states, residual = _cp(
                            layer, hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen, residual,
                            use_reentrant=False,
                        )
                else:
                    hidden_states, residual = layer(
                        hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen, residual,
                    )
            # Final norm: fuse last residual add
            hidden_states = _fused_rms_norm_fn(
                hidden_states, self.norm.weight, None,
                residual=residual, eps=self.norm.variance_epsilon,
            )
        else:
            # Original path: layers return hidden_states only
            for layer in self.layers:
                if _gc:
                    if self._gc_mode == "attention_only":
                        # Checkpoint attention only
                        hidden_states_attn, attn_residual = _cp(
                            layer._forward_attn, hidden_states, cos, sin,
                            attention_mask, cu_seqlens, max_seqlen,
                            use_reentrant=False,
                        )
                        # FFN not checkpointed
                        hidden_states = layer._forward_ffn(
                            hidden_states_attn, attn_residual, cu_seqlens, max_seqlen
                        )
                    else:
                        hidden_states = _cp(
                            layer, hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen,
                            use_reentrant=False,
                        )
                else:
                    hidden_states = layer(hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen)
            hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _fused_attn_block(layer, hidden_states, cos, sin, attention_mask, cu_seqlens, max_seqlen, residual):
        """Fused-path attention sub-block for selective gradient checkpointing."""
        eps_in = layer.input_layernorm.variance_epsilon
        eps_post = layer.post_attention_layernorm.variance_epsilon

        if residual is None:
            normed = _fused_rms_norm_fn(
                hidden_states, layer.input_layernorm.weight, None, eps=eps_in,
            )
            residual = hidden_states
        else:
            normed, residual = _fused_rms_norm_fn(
                hidden_states, layer.input_layernorm.weight, None,
                residual=residual, prenorm=True, eps=eps_in,
            )

        hidden_states = layer._attention(normed, cos, sin, attention_mask, cu_seqlens, max_seqlen)

        normed, residual = _fused_rms_norm_fn(
            hidden_states, layer.post_attention_layernorm.weight, None,
            residual=residual, prenorm=True, eps=eps_post,
        )
        return normed, residual

    def forward(self, xs_pad: Tensor, masks: Tensor) -> tuple:
        """Forward pass compatible with ConformerEncoder interface.

        Args:
            xs_pad: (batch_size, seq_len, input_size) — raw mel features.
            masks: (batch_size, seq_len) or (batch_size, 1, seq_len) — padding mask (True=valid).

        Returns:
            output: (batch_size, subsampled_seq_len, hidden_size)
            masks: updated mask after subsampling
        """
        # Frontend: 8x subsampling
        hidden_states, masks = self.frontend(xs_pad, masks)

        # Project to hidden_size
        hidden_states = self.input_proj(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape

        # Position IDs
        position_ids = torch.arange(seq_len, device=hidden_states.device)

        # RoPE embeddings — use fused format when fused kernels enabled
        if self.use_fused_kernels and _fused_rotary_available:
            cos, sin = self.rotary_emb.forward_fused(hidden_states, position_ids)
        else:
            cos, sin = self.rotary_emb(hidden_states, position_ids)

        if _flash_attn_available:
            # Check if we need unpadding (variable-length sequences in batch)
            has_padding = False
            if masks is not None:
                pad_mask = masks.squeeze(1) if masks.dim() == 3 else masks
                has_padding = not pad_mask.all()

            if has_padding:
                # Unpad for efficient variable-length attention
                indices, cu_seqlens, max_seqlen = self._get_unpad_data(pad_mask)
                hidden_states = hidden_states.reshape(-1, self.hidden_size)[indices]

                hidden_states = self._run_layers(hidden_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

                # Re-pad to original batch layout
                hidden_states = pad_input(hidden_states, indices, batch_size, seq_len)
            else:
                # No padding — use flash_attn_func (fastest path)
                hidden_states = self._run_layers(hidden_states, cos, sin)
        else:
            # SDPA fallback: build explicit attention mask if needed
            attention_mask = None

            if self.causal or self.sliding_window is not None:
                attention_mask = self._build_sdpa_mask(seq_len, hidden_states.device, hidden_states.dtype)

            # Combine with padding mask
            if masks is not None:
                pad_mask = masks.squeeze(1) if masks.dim() == 3 else masks
                if not pad_mask.all():
                    pad_mask_float = pad_mask.float()
                    pad_mask_expanded = pad_mask_float.unsqueeze(1).unsqueeze(2)
                    pad_attn_mask = (1.0 - pad_mask_expanded) * torch.finfo(hidden_states.dtype).min
                    if attention_mask is not None:
                        attention_mask = attention_mask + pad_attn_mask
                    else:
                        attention_mask = pad_attn_mask

            for layer in self.layers:
                if self._gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer, hidden_states, cos, sin, attention_mask, None, None,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = layer(hidden_states, cos, sin, attention_mask=attention_mask)

            hidden_states = self.norm(hidden_states)

        return hidden_states, masks

    def forward_packed(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        """Forward pass for pre-packed sequences (SFT packing).

        This bypasses the frontend — input should already be subsampled and projected.
        Use this when sequences are packed externally (e.g., by a data collator).

        Args:
            hidden_states: (total_tokens, hidden_size) — concatenated token embeddings.
            cu_seqlens: (num_sequences + 1,) — cumulative sequence lengths, int32.
            max_seqlen: maximum sequence length in the batch.

        Returns:
            output: (total_tokens, hidden_size)
        """
        assert _flash_attn_available, "forward_packed requires flash_attn"

        # RoPE: compute for max_seqlen positions
        position_ids = torch.arange(max_seqlen, device=hidden_states.device)
        if self.use_fused_kernels and _fused_rotary_available:
            cos, sin = self.rotary_emb.forward_fused(hidden_states, position_ids)
        else:
            cos, sin = self.rotary_emb(hidden_states, position_ids)

        return self._run_layers(hidden_states, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def _build_sdpa_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        """Build attention mask for SDPA fallback (causal / sliding window).

        Only used when flash_attn is not available.
        """
        if not self.causal and self.sliding_window is None:
            return None

        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

        if self.causal:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            mask = mask + causal_mask

        if self.sliding_window is not None:
            if self.causal:
                for i in range(seq_len):
                    start = max(0, i - self.sliding_window + 1)
                    if start > 0:
                        mask[i, :start] = float("-inf")
            else:
                half_w = self.sliding_window // 2
                for i in range(seq_len):
                    start = max(0, i - half_w)
                    end = min(seq_len, i + half_w + 1)
                    if start > 0:
                        mask[i, :start] = float("-inf")
                    if end < seq_len:
                        mask[i, end:] = float("-inf")

        return mask.unsqueeze(0).unsqueeze(0)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Self-contained Conformer encoder + utils for HF trust_remote_code deployment."""

from __future__ import annotations

import abc
import math
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.act_fn(x)


def get_activation(name="relu"):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    return nn.Identity()


# ---------------------------------------------------------------------------
# GLU modules
# ---------------------------------------------------------------------------


class GLU(nn.Module):
    def __init__(self, dim: int = -1, act_name: str = "sigmoid"):
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()
        if self.act_name == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = Swish()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        half_x, gate = x.chunk(2, dim=self.dim)
        return half_x * self.act_fn(gate)


class GLUPointWiseConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, glu_type="sigmoid", bias_in_glu=True, causal=False):
        super().__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        if causal:
            self.ext_pw_conv_1d = nn.Conv1d(input_dim, output_dim * 2, kernel_size, 1, padding=(kernel_size - 1))
        else:
            self.ext_pw_conv_1d = nn.Conv1d(input_dim, output_dim * 2, kernel_size, 1, padding=(kernel_size - 1) // 2)
        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = Swish()
        else:
            raise ValueError(f"Unsupported activation type {glu_type}")
        if bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, output_dim, 1))
            self.b2 = nn.Parameter(torch.zeros(1, output_dim, 1))

    def forward(self, x):
        x = x.permute([0, 2, 1])
        x = self.ext_pw_conv_1d(x)
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                x = (x[:, 0:self.output_dim, :] + self.b1) * (x[:, self.output_dim:self.output_dim * 2, :] + self.b2)
            else:
                x = (x[:, 0:self.output_dim, :]) * (x[:, self.output_dim:self.output_dim * 2, :])
        else:
            if self.bias_in_glu:
                x = (x[:, 0:self.output_dim, :] + self.b1) * self.glu_act(x[:, self.output_dim:self.output_dim * 2, :] + self.b2)
            else:
                x = (x[:, 0:self.output_dim, :]) * self.glu_act(x[:, self.output_dim:self.output_dim * 2, :])
        x = x.permute([0, 2, 1])
        return x


class GLULinear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type)

    def forward(self, x):
        x = self.linear(x)
        return self.glu_act(x)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout_rate, activation="sigmoid", bias_in_glu=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, activation, bias_in_glu)
        self.net = nn.Sequential(module, nn.Dropout(dropout_rate), nn.Linear(d_inner, d_model), nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.net(self.layer_norm(x))


# ---------------------------------------------------------------------------
# Depthwise Separable Conv
# ---------------------------------------------------------------------------


class DepthWiseSeperableConv1d(nn.Module):
    def __init__(self, input_dim, depthwise_seperable_out_channel, kernel_size, depthwise_multiplier, padding=0):
        super().__init__()
        self.dw_conv = nn.Conv1d(input_dim, input_dim * depthwise_multiplier, kernel_size, 1, padding=padding, groups=input_dim)
        if depthwise_seperable_out_channel != 0:
            self.pw_conv = nn.Conv1d(input_dim * depthwise_multiplier, depthwise_seperable_out_channel, 1, 1, 0)
        else:
            self.pw_conv = nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def forward(self, x):
        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x


# ---------------------------------------------------------------------------
# ConvModule
# ---------------------------------------------------------------------------


class ConvModule(nn.Module):
    def __init__(self, input_dim, ext_pw_out_channel, depthwise_seperable_out_channel,
                 ext_pw_kernel_size, kernel_size, depthwise_multiplier, dropout_rate,
                 causal=False, batch_norm=False, chunk_se=0, chunk_size=18,
                 activation="relu", glu_type="sigmoid", bias_in_glu=True,
                 linear_glu_in_convm=False, export=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal
        self._add_ext_pw_layer()
        self.batch_norm = batch_norm
        self.kernel_size = kernel_size
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(input_dim)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        self.export = export
        if causal:
            padding = 0 if export else kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2
        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(input_dim, depthwise_seperable_out_channel, kernel_size, depthwise_multiplier, padding=padding)
        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = nn.Linear(depthwise_seperable_out_channel, input_dim)
        else:
            if depthwise_multiplier != 1:
                self.ln2 = nn.Linear(input_dim * depthwise_multiplier, input_dim)

    def _add_ext_pw_layer(self):
        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = nn.Identity()
        self.squeeze_excitation = nn.Identity()
        self.apply_ln1 = self.fix_len1 = False
        if self.ext_pw_out_channel != 0:
            if self.causal:
                self.ext_pw_conv_1d = nn.Conv1d(self.input_dim, self.ext_pw_out_channel, self.ext_pw_kernel_size, 1, padding=(self.ext_pw_kernel_size - 1))
                self.fix_len1 = self.ext_pw_kernel_size > 1
            else:
                self.ext_pw_conv_1d = nn.Conv1d(self.input_dim, self.ext_pw_out_channel, self.ext_pw_kernel_size, 1, padding=(self.ext_pw_kernel_size - 1) // 2)
                self.fix_len1 = False
            if self.linear_glu_in_convm:
                self.glu = GLULinear(self.input_dim, self.ext_pw_out_channel, self.glu_type, self.bias_in_glu)
            else:
                self.glu = GLUPointWiseConv(self.input_dim, self.ext_pw_out_channel, self.ext_pw_kernel_size, self.glu_type, self.bias_in_glu, self.causal)
            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = torch.nn.Parameter(torch.ones(3))
            self.pw_conv_simplify_b = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x):
        x = self.layer_norm(x)
        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, :-(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1
        x = x.permute([0, 2, 1])
        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, :-(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.permute([0, 2, 1])
            x = self.ln2(x)
            x = x.permute([0, 2, 1])
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)
        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, :-(self.ext_pw_kernel_size - 1)]
            if self.apply_ln1:
                x = x.permute([0, 2, 1])
                x = self.ln1(x)
                x = x.permute([0, 2, 1])
            x = x.permute([0, 2, 1])
        else:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = x.squeeze(1)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------


def _pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super().__init__()
        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2 * maxlen, d_model)
        self.pe_v = torch.nn.Embedding(2 * maxlen, d_model) if embed_v else None

    def forward(self, pos_seq: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        pos_seq = pos_seq.masked_fill(pos_seq < -self.maxlen, -self.maxlen)
        pos_seq = pos_seq.masked_fill(pos_seq > self.maxlen - 1, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.pe_v is not None:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        return self.pe_k(pos_seq), None


class T5RelativeAttentionLogitBias(nn.Module):
    def __init__(self, num_heads, num_buckets=-1, max_distance=1000, symmetric=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.symmetric = symmetric
        self._skip_bucketing = self.num_buckets < 0
        if self._skip_bucketing:
            self.num_buckets = max_distance
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, x):
        maxpos = x.size(1)
        context_position = torch.arange(maxpos, device=x.device, dtype=torch.long)[:, None]
        memory_position = torch.arange(maxpos, device=x.device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.masked_fill(relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(relative_position > self.max_distance - 1, self.max_distance - 1)
        if self._skip_bucketing:
            bias_idx = relative_position
        else:
            bias_idx = relative_position
        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx += self.num_buckets // 2
        t5_rel_att_bias = self.bias_values(bias_idx)
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(0)
        return t5_rel_att_bias


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# MeanVarianceNormLayer
# ---------------------------------------------------------------------------


class MeanVarianceNormLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.global_mean = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.global_invstd = nn.Parameter(torch.ones(input_size), requires_grad=False)

    def forward(self, input_: Tensor) -> Tensor:
        return (input_ - self.global_mean) * self.global_invstd


# ---------------------------------------------------------------------------
# Causal Conv helpers
# ---------------------------------------------------------------------------


class CausalConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x):
        if self.training:
            x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        else:
            x = F.pad(x, pad=(self._left_padding, self._right_padding, 0, 0))
        x = super().forward(x)
        return x


# ---------------------------------------------------------------------------
# NemoConvSubsampling
# ---------------------------------------------------------------------------


class NemoConvSubsampling(torch.nn.Module):
    def __init__(self, feat_in, feat_out, subsampling_factor=4, subsampling="dw_striding",
                 conv_channels=256, subsampling_conv_chunking_factor=1, activation=nn.ReLU(), is_causal=False):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_causal_cond = subsampling in ("dw_striding", "striding", "striding_conv1d")
        if subsampling_conv_chunking_factor != -1 and subsampling_conv_chunking_factor != 1 and subsampling_conv_chunking_factor % 2 != 0:
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor
        in_channels = 1
        layers = []
        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False
            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0
            if self.is_causal:
                layers.append(CausalConv2D(in_channels=in_channels, out_channels=conv_channels, kernel_size=self._kernel_size, stride=self._stride, padding=None))
            else:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=self._kernel_size, stride=self._stride, padding=self._left_padding))
            in_channels = conv_channels
            layers.append(activation)
            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(CausalConv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=self._kernel_size, stride=self._stride, padding=None, groups=in_channels))
                else:
                    layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self._kernel_size, stride=self._stride, padding=self._left_padding, groups=in_channels))
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=1, stride=1, padding=0, groups=1))
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")
        # Compute output length
        out_length = float(feat_in)
        add_pad = float((self._left_padding + self._right_padding) - self._kernel_size)
        for _ in range(self._sampling_num):
            out_length = (out_length + add_pad) / self._stride + 1.0
            out_length = float(math.floor(out_length))
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, mask):
        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        if mask is None:
            return x, None
        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        if self.is_causal and self.subsampling_causal_cond:
            feature_lens_remainder = feature_lens % self.subsampling_factor
            padding_length[feature_lens_remainder != 1] += 1
        pad_mask = (torch.arange(0, max_audio_length, device=x.device).expand(padding_length.size(0), -1) < padding_length.unsqueeze(1))
        return x, pad_mask.unsqueeze(1)


# ---------------------------------------------------------------------------
# MultiHeadedAttention
# ---------------------------------------------------------------------------


def masked_softmax(scores, mask: Optional[Tensor]):
    if mask is not None:
        mask = mask.unsqueeze(1).eq(0)
        scores = scores.masked_fill(mask, -torch.inf)
        attn = torch.softmax(scores.float(), dim=-1).masked_fill(mask, 0.0).to(scores.dtype)
    else:
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    return attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate, attention_inner_dim=-1, glu_type="swish",
                 bias_in_glu=True, use_pt_scaled_dot_product_attention=False, n_value=-1, group_size=1):
        super().__init__()
        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0
        self.d_k = attention_inner_dim // n_head
        self.inv_sqrt_d_k = 1.0 / math.sqrt(self.d_k)
        self.h = n_head
        assert n_head % group_size == 0
        self.g = group_size
        self.h_k = n_head // group_size
        self.linear_q = nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = nn.Linear(n_value, attention_inner_dim // group_size)
        self.linear_out = nn.Linear(attention_inner_dim // group_size, n_value)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.use_pt_scaled_dot_product_attention = use_pt_scaled_dot_product_attention

    def forward(self, query, key, value, pos_k, pos_v, mask, relative_attention_bias=None):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h_k, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h_k, self.d_k)
        if self.use_pt_scaled_dot_product_attention and not torch.jit.is_scripting():
            q = q.transpose(1, 2)
        else:
            q = q.transpose(1, 2) * self.inv_sqrt_d_k
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_pt_scaled_dot_product_attention and not torch.jit.is_scripting():
            attn_mask = None
            if mask is not None:
                mask_unsq = mask.unsqueeze(1)
                attn_mask = mask_unsq + relative_attention_bias if relative_attention_bias is not None else mask_unsq
                if mask_unsq.dtype != q.dtype:
                    attn_mask = attn_mask.to(q.dtype)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_rate)
        else:
            if self.h != self.h_k:
                q = q.reshape(n_batch, self.g, self.h_k, -1, self.d_k)
                A = torch.einsum("b g h t d, b h s d -> b h t s", q, k)
            else:
                A = torch.matmul(q, k.transpose(-2, -1))
            if pos_k is not None:
                if self.h != self.h_k:
                    B = torch.einsum("b g h t d, t s d -> b h t s", q, pos_k)
                else:
                    reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0, 1)
                    B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
                    B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
                scores = A + B
            else:
                scores = A
            if relative_attention_bias is not None:
                scores = scores + relative_attention_bias
            attn = masked_softmax(scores, mask)
            self.attn = attn
            p_attn = self.dropout(attn)
            x = torch.matmul(p_attn.to(v.dtype), v)
            if pos_v is not None:
                reshape_attn = p_attn.contiguous().view(n_batch * self.h, pos_v.size(0), pos_v.size(1)).transpose(0, 1)
                attn_v = torch.matmul(reshape_attn, pos_v).transpose(0, 1).contiguous().view(n_batch, self.h, pos_v.size(0), self.d_k)
                x = x + attn_v
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h_k * self.d_k)
        return self.linear_out(x)


# ---------------------------------------------------------------------------
# unfold_tensor
# ---------------------------------------------------------------------------


def unfold_tensor(xs_pad, max_seq_len):
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)
    xs_pad = F.unfold(xs_pad[..., None, :], kernel_size=(1, max_seq_len), stride=(1, max_seq_len))
    new_bsz, _, slen = xs_pad.shape
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad


# ---------------------------------------------------------------------------
# adaptive_enc_mask
# ---------------------------------------------------------------------------


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()
    start_pad = torch.nn.functional.pad(chunk_start_idx, (1, 0))
    end_pad = torch.nn.functional.pad(chunk_start_idx, (0, 1), value=x_len)
    seq_range = torch.arange(0, x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
    boundary = end_pad[idx]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


# ---------------------------------------------------------------------------
# Activation checkpointing (simplified for inference — no-ops)
# ---------------------------------------------------------------------------


def _identity_wrapper(x):
    return x


def embedding_checkpoint_wrapper(activation_checkpointing):
    return lambda x: x


def encoder_checkpoint_wrapper(activation_checkpointing, layer_cls=None, idx=0):
    return lambda x: x


def attn_checkpointing(activation_checkpointing, i):
    return ""


# ---------------------------------------------------------------------------
# ConformerEncoderLayer
# ---------------------------------------------------------------------------


class MultiSequential(torch.nn.Sequential):
    @torch.jit.ignore
    def forward(self, *args):
        for m in self:
            args = m(*args)
        return args


def repeat(repeat_num, module_gen_fn):
    return MultiSequential(*[module_gen_fn(i) for i in range(repeat_num)])


class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, ext_pw_out_channel=0, depthwise_seperable_out_channel=256,
                 depthwise_multiplier=1, n_head=4, d_ffn=2048, ext_pw_kernel_size=1,
                 kernel_size=3, dropout_rate=0.1, causal=False, batch_norm=False,
                 activation="relu", chunk_se=0, chunk_size=18, conv_activation="relu",
                 conv_glu_type="sigmoid", bias_in_glu=True, linear_glu_in_convm=False,
                 attention_innner_dim=-1, attention_glu_type="swish",
                 activation_checkpointing="", export=False,
                 use_pt_scaled_dot_product_attention=False, attn_group_sizes=1,
                 conv_threshold=8, th=0):
        super().__init__()
        self.feed_forward_in = FeedForward(d_model=d_model, d_inner=d_ffn, dropout_rate=dropout_rate, activation=activation, bias_in_glu=bias_in_glu)
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout_rate, attention_innner_dim, attention_glu_type, bias_in_glu, use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention, group_size=attn_group_sizes)
        self.th = th
        self.use_conv = (self.th < conv_threshold)
        if self.use_conv:
            self.conv = ConvModule(d_model, ext_pw_out_channel, depthwise_seperable_out_channel, ext_pw_kernel_size, kernel_size, depthwise_multiplier, dropout_rate, causal, batch_norm, chunk_se, chunk_size, conv_activation, conv_glu_type, bias_in_glu, linear_glu_in_convm, export=export)
        else:
            self.conv = None
        self.feed_forward_out = FeedForward(d_model=d_model, d_inner=d_ffn, dropout_rate=dropout_rate, activation=activation, bias_in_glu=bias_in_glu)
        self.layer_norm_att = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_k, pos_v, mask, relative_attention_bias=None):
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)
        x = x + self.self_attn(norm_x, norm_x, norm_x, pos_k, pos_v, mask, relative_attention_bias=relative_attention_bias)
        if self.use_conv and self.conv is not None:
            x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)
        out = self.layer_norm(x)
        return out, pos_k, pos_v, mask


# ---------------------------------------------------------------------------
# TransformerEncoderBase
# ---------------------------------------------------------------------------


class TransformerEncoderBase(abc.ABC, nn.Module):
    def __init__(self, input_size, chunk_size, left_chunk, attention_dim=256, attention_heads=4,
                 input_layer="nemo_conv", cnn_out=-1, cnn_layer_norm=False, time_reduction=4,
                 dropout_rate=0.0, padding_idx=-1, relative_k=False, relative_v=False,
                 relative_attention_bias_args=None, positional_dropout_rate=0.0,
                 nemo_conv_settings=None, conv2d_extra_padding="none", attention_group_size=1,
                 encoder_embedding_config=None):
        super().__init__()
        self.input_size = input_size
        self.input_layer = input_layer
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.attention_dim = attention_dim
        self.num_heads = attention_heads
        self.attention_group_size = attention_group_size
        self.time_reduction = time_reduction
        self.nemo_conv_settings = nemo_conv_settings
        self.encoder_embedding_config = encoder_embedding_config

        if self.input_layer == "nemo_conv":
            default_nemo_conv_settings = {
                "subsampling": "dw_striding", "subsampling_factor": self.time_reduction,
                "feat_in": input_size, "feat_out": attention_dim, "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1, "activation": nn.ReLU(), "is_causal": False,
            }
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
            self.embed = NemoConvSubsampling(**default_nemo_conv_settings)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.relative_k = relative_k
        if relative_k:
            self.pos_emb = RelativePositionalEncoding(attention_dim // attention_heads, 1000, relative_v)
            self.dropout_layer = torch.nn.Dropout(p=positional_dropout_rate)
        else:
            self.pos_emb = AbsolutePositionalEncoding(attention_dim, positional_dropout_rate)

        self.relative_attention_bias_type = relative_attention_bias_args.get("type") if relative_attention_bias_args else None
        if self.relative_attention_bias_type == "t5":
            assert self.num_heads % self.attention_group_size == 0
            self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(
                self.num_heads // self.attention_group_size,
                max_distance=relative_attention_bias_args.get("t5_bias_max_distance", 1000),
                symmetric=relative_attention_bias_args.get("t5_bias_symmetric", False),
            )
        else:
            self.relative_attention_bias_layer = None

    def post_init(self, init_model_config):
        pretrained_speech_encoder_path = init_model_config.get("pretrained_speech_encoder_path", None)
        if pretrained_speech_encoder_path:
            if pretrained_speech_encoder_path.startswith("az://"):
                import blobfile as bf
                with bf.BlobFile(pretrained_speech_encoder_path, "rb") as f:
                    model_state = torch.load(f, map_location="cpu")
            else:
                model_state = torch.load(pretrained_speech_encoder_path, map_location="cpu")
            if "module" in model_state:
                model_state = model_state["module"]
                is_ds = True
            else:
                is_ds = False
            encoder_state_dict = {}
            for k, v in model_state.items():
                if is_ds and "encoder." in k:
                    encoder_state_dict[k.replace("model.embed_tokens_extend.encoder.", "")] = v
                elif "encoder." in k:
                    encoder_state_dict[k.replace("encoder.", "")] = v
            if hasattr(self, "encoder_embedding"):
                del self.encoder_embedding
            self.load_state_dict(encoder_state_dict, strict=False)
        if not hasattr(self, "encoder_embedding"):
            self.encoder_embedding = MeanVarianceNormLayer(self.encoder_embedding_config["input_size"])

    def compute_lens_change(self, feature_lens):
        if self.input_layer == "nemo_conv":
            ceil_func = math.ceil if isinstance(feature_lens, int) else torch.ceil
            return ceil_func(feature_lens / self.time_reduction)
        return feature_lens

    @abc.abstractmethod
    def forward(self):
        pass

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk
        return chunk_size_train_eff, left_chunk_train_eff

    def _forward_embeddings_core(self, input_tensor, masks):
        input_tensor, masks = self.embed(input_tensor, masks)
        return input_tensor, masks

    def _position_embedding(self, input_tensor):
        if self.relative_k:
            x_len = input_tensor.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(input_tensor.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, pos_v = self.pos_emb(pos_seq)
            input_tensor = self.dropout_layer(input_tensor)
        else:
            pos_k = None
            pos_v = None
            if self.relative_attention_bias_layer is None:
                input_tensor = self.pos_emb(input_tensor)
        return pos_k, pos_v

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        if self.training and np.random.rand() > 0.5:
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)
        enc_streaming_mask = adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk_train_eff).unsqueeze(0).expand([batch_size, -1, -1])
        return enc_streaming_mask

    def forward_embeddings(self, xs_pad, masks, chunk_size_nc=None, left_chunk_nc=None):
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))
        if seq_len <= 0:
            raise ValueError(f"Sequence length after time reduction is invalid: {seq_len}.")
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.chunk_size, self.left_chunk)
        if xs_pad.is_cuda:
            enc_streaming_mask = enc_streaming_mask.cuda()
            xs_pad = xs_pad.cuda()
        input_tensor = xs_pad
        input_tensor, masks = self._forward_embeddings_core(input_tensor, masks)
        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask
        pos_k, pos_v = self._position_embedding(input_tensor)
        return input_tensor, pos_k, pos_v, hs_mask, masks

    def get_offset(self):
        return get_offset(self.input_layer, self.time_reduction)


def get_offset(input_layer, time_reduction):
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 4:
        return 3
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 8:
        return 7
    return 0


# ---------------------------------------------------------------------------
# ConformerEncoder
# ---------------------------------------------------------------------------


class ConformerEncoder(TransformerEncoderBase):
    extra_multi_layer_output_idxs: List[int]

    def __init__(self, input_size, chunk_size=-1, left_chunk=0, num_lang=None,
                 attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6,
                 dropout_rate=0.1, input_layer="nemo_conv", causal=True, batch_norm=False,
                 cnn_out=-1, cnn_layer_norm=False, ext_pw_out_channel=0, ext_pw_kernel_size=1,
                 depthwise_seperable_out_channel=256, depthwise_multiplier=1, chunk_se=0,
                 kernel_size=3, activation="relu", conv_activation="relu",
                 conv_glu_type="sigmoid", bias_in_glu=True, linear_glu_in_convm=False,
                 attention_glu_type="swish", export=False, extra_layer_output_idx=-1,
                 extra_multi_layer_output_idxs=[], activation_checkpointing="",
                 relative_k=False, relative_v=False, relative_attention_bias_args=None,
                 time_reduction=4, use_pt_scaled_dot_product_attention=False,
                 nemo_conv_settings=None, conv_threshold=8,
                 conv2d_extra_padding="none", replication_pad_for_subsample_embedding=False,
                 attention_group_size=1, encoder_embedding_config=None):
        super().__init__(
            input_size, chunk_size, left_chunk, attention_dim, attention_heads,
            input_layer, cnn_out, cnn_layer_norm, time_reduction,
            dropout_rate=dropout_rate, relative_k=relative_k, relative_v=relative_v,
            relative_attention_bias_args=relative_attention_bias_args,
            positional_dropout_rate=0.0, nemo_conv_settings=nemo_conv_settings,
            conv2d_extra_padding=conv2d_extra_padding,
            attention_group_size=attention_group_size,
            encoder_embedding_config=encoder_embedding_config,
        )
        self.num_blocks = num_blocks
        self.num_lang = num_lang
        self.kernel_size = kernel_size
        self.replication_pad_for_subsample_embedding = replication_pad_for_subsample_embedding
        assert self.num_heads % attention_group_size == 0
        self.num_heads_k = self.num_heads // attention_group_size
        self.is_streaming_encoder = (chunk_size > -1)

        self.encoders = repeat(
            num_blocks,
            lambda i: ConformerEncoderLayer(
                d_model=attention_dim, ext_pw_out_channel=ext_pw_out_channel,
                depthwise_seperable_out_channel=depthwise_seperable_out_channel,
                depthwise_multiplier=depthwise_multiplier, n_head=attention_heads,
                d_ffn=linear_units, ext_pw_kernel_size=ext_pw_kernel_size,
                kernel_size=kernel_size, dropout_rate=dropout_rate, causal=causal,
                batch_norm=batch_norm, activation=activation, chunk_se=chunk_se,
                chunk_size=chunk_size, conv_activation=conv_activation,
                conv_glu_type=conv_glu_type, bias_in_glu=bias_in_glu,
                linear_glu_in_convm=linear_glu_in_convm,
                attention_glu_type=attention_glu_type,
                activation_checkpointing="", export=export,
                use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
                attn_group_sizes=attention_group_size,
                conv_threshold=conv_threshold,
                th=i if self.is_streaming_encoder else 0,
            ),
        )
        self.extra_layer_output_idx = extra_layer_output_idx
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs
        self.register_buffer("dev_type", torch.zeros(()), persistent=False)

    def init_relative_attention_bias(self, input_tensor):
        if self.relative_attention_bias_layer:
            return self.relative_attention_bias_layer(input_tensor)

    def calculate_hs_mask(self, xs_pad, device, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask
        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = (torch.arange(0, max_audio_length, device=device).expand(padding_length.size(0), -1) < padding_length.unsqueeze(1))
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    @torch.jit.ignore
    def forward(self, xs_pad, masks):
        xs_pad = self.encoder_embedding(xs_pad)
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(xs_pad, masks)
        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500
        if seq_len > max_seq_len and not self.is_streaming_encoder:
            unfolded = True
            chunk_pad_size = max_seq_len - (seq_len % max_seq_len) if seq_len % max_seq_len > 0 else 0
            if chunk_pad_size > 0:
                input_tensor_pad = F.pad(input_tensor, (0, 0, 0, chunk_pad_size), "constant", 0)
                input_tensor = input_tensor_pad.to(input_tensor.device)
            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                subsampled_pad_mask = masks.squeeze(1)
                extra_padded_subsamlped_pad_mask = F.pad(subsampled_pad_mask, (0, chunk_pad_size), "constant", False)
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(extra_padded_subsamlped_pad_mask, max_seq_len)
                masks_unfold = masks_unfold.squeeze(-1).bool()
            else:
                masks_unfold = None
            hs_mask = self.calculate_hs_mask(input_tensor, input_tensor.device, masks_unfold)

        relative_attention_bias = self.init_relative_attention_bias(input_tensor)
        _simplified_path = (self.extra_layer_output_idx == -1 and relative_attention_bias is None)
        if _simplified_path:
            input_tensor, *_ = self.encoders(input_tensor, pos_k, pos_v, hs_mask)
        else:
            for i, layer in enumerate(self.encoders):
                input_tensor, _, _, _ = layer(input_tensor, pos_k, pos_v, hs_mask, relative_attention_bias=relative_attention_bias)

        if unfolded:
            embed_dim = input_tensor.shape[-1]
            input_tensor = input_tensor.reshape(ori_bz, -1, embed_dim)
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]
        return input_tensor, masks

    def gradient_checkpointing_enable(self):
        pass

# Copyright 2026 Microsoft Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keep Qwen3.5's convolution and delta-rule states separate in packed batches."""

import inspect
import sys
from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F


_PACKING_KWARG = "_verl_qwen3_5_packing"


def _packing_info(position_ids):
    if position_ids is None:
        return None
    if position_ids.ndim == 3:
        if position_ids.shape[1] != 1:
            return None
        if position_ids.shape[0] == 3:
            # Spatial/temporal RoPE axes can repeat within a single sample. Do
            # not interpret them as text boundaries during ordinary inference.
            # A suspected packed reset needs the separate text-position axis.
            if bool(position_ids[:, :, 1:].eq(0).all(dim=0).any()):
                raise ValueError("Packed Qwen3.5 requires text position IDs (2D or the first of four position axes).")
            return None
        if position_ids.shape[0] != 4:
            raise ValueError("Packed Qwen3.5 requires text position IDs (2D or the first of four position axes).")
        position_ids = position_ids[0]
    if position_ids.ndim != 2 or position_ids.shape[0] != 1 or position_ids.shape[-1] == 0:
        return None

    # Each unpadded sample starts at zero, including adjacent single-token samples.
    starts = position_ids.eq(0)
    starts[:, 0] = True
    offsets = starts[0].nonzero(as_tuple=True)[0]
    if offsets.numel() <= 1:
        return None
    cu_seqlens = torch.cat([offsets, offsets.new_tensor([position_ids.shape[-1]])]).to(torch.int32)
    seq_idx = starts.cumsum(dim=-1, dtype=torch.int32) - 1
    # Compute host boundaries once, rather than synchronizing again in every layer.
    return cu_seqlens, seq_idx, cu_seqlens.tolist()


def _packed_linear_forward(self, hidden_states, cu_seqlens, seq_idx):
    """Run the existing projections with both fast kernels respecting boundaries."""
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    mixed_qkv = self.causal_conv1d_fn(
        x=mixed_qkv,
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,
        seq_idx=seq_idx,
    ).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    core_attn_out, _ = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z)
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def _patch_linear_attention(module):
    original_forward = module.forward.__func__
    source_module = sys.modules[type(module).__module__]
    # The torch fallback can accept cu_seqlens without actually using it. Only
    # the imported FLA kernel provides the required variable-length semantics.
    fast_delta_rule = getattr(source_module, "chunk_gated_delta_rule", None)
    use_fast_kernels = (
        module.causal_conv1d_fn is not None
        and fast_delta_rule is not None
        and module.chunk_gated_delta_rule is fast_delta_rule
    )

    @wraps(original_forward)
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        packing = kwargs.pop(_PACKING_KWARG, None)
        if packing is None:
            return original_forward(
                self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
            )
        cu_seqlens, seq_idx, bounds = packing
        if use_fast_kernels:
            return _packed_linear_forward(self, hidden_states, cu_seqlens, seq_idx)
        # Isolate both states even when either fast kernel is unavailable. One
        # batched call keeps the number/order of FSDP collectives independent of
        # the number of packed samples on each rank. Right padding cannot affect
        # the valid prefix of a causal convolution or delta-rule recurrence.
        segments = [hidden_states[0, start:end] for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
        padded_states = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)
        output = original_forward(self, padded_states)
        return torch.cat(
            [output[index : index + 1, : segment.shape[0]] for index, segment in enumerate(segments)],
            dim=1,
        )

    module.forward = MethodType(forward, module)


def _patch_decoder(module):
    original_forward = module.forward.__func__

    @wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        packing = kwargs.pop(_PACKING_KWARG, None)
        if packing is None or self.layer_type != "linear_attention":
            return original_forward(self, hidden_states, *args, **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(hidden_states, **{_PACKING_KWARG: packing})
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)

    module.forward = MethodType(forward, module)


def _patch_text_model(model):
    original_forward = model.forward.__func__
    signature = inspect.signature(original_forward)
    extra_kwargs = next(
        name for name, parameter in signature.parameters.items() if parameter.kind == parameter.VAR_KEYWORD
    )

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        arguments = bound.arguments
        attention_mask = arguments.get("attention_mask")
        unmasked = attention_mask is None or (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.shape[0] == 1
            and bool(torch.all(attention_mask == 1))
        )
        if unmasked:
            packing = _packing_info(arguments.get("position_ids"))
            if packing is not None:
                use_cache = arguments.get("use_cache")
                if use_cache is None:
                    use_cache = self.config.use_cache
                if use_cache or arguments.get("past_key_values") is not None:
                    raise ValueError("Qwen3.5 packed samples require use_cache=False and no past_key_values.")
                # HF only constructs a packed full-attention mask when the
                # redundant all-ones mask is absent. Linear and full layers
                # must use the same sample boundaries.
                arguments["attention_mask"] = None
                arguments.setdefault(extra_kwargs, {})[_PACKING_KWARG] = packing
        # HF's configuration-default decorator injects keyword arguments such
        # as use_cache. Forward named parameters by keyword even when the caller
        # supplied them positionally, avoiding duplicate argument values.
        arguments.pop("self")
        model_kwargs = arguments.pop(extra_kwargs, {})
        return original_forward(self, **arguments, **model_kwargs)

    model.forward = MethodType(forward, model)


def patch_qwen3_5_packed_sequences(model, ulysses_sp_size=1):
    """Patch one native/audio Qwen3.5 model before FSDP wrapping.

    Packing metadata travels with each forward call, including checkpoint
    recomputation. Ordinary padded batches and cache decoding use the original
    implementation. No parameters or checkpoint keys are added.
    """
    layers = [module for module in model.modules() if type(module).__name__ == "Qwen3_5DecoderLayer"]
    if not any(layer.layer_type == "linear_attention" for layer in layers):
        return
    if ulysses_sp_size > 1:
        raise ValueError("Qwen3.5 packed linear attention does not support Ulysses sequence parallelism (SP > 1).")
    if getattr(model, "_verl_qwen3_5_packing_patched", False):
        return

    text_models = [module for module in model.modules() if type(module).__name__ == "Qwen3_5TextModel"]
    if not text_models:
        raise ValueError("Cannot locate the Qwen3.5 text model to pass packed sequence boundaries to its layers.")
    for layer in layers:
        if layer.layer_type == "linear_attention":
            _patch_linear_attention(layer.linear_attn)
        _patch_decoder(layer)
    # Both ordinary and fused CausalLM forwards call the text model. Keep this
    # below the audio embedding path, without changing any multimodal inputs.
    for text_model in text_models:
        _patch_text_model(text_model)
    model._verl_qwen3_5_packing_patched = True

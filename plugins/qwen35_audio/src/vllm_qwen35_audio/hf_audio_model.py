"""HF-compatible Qwen3.5 Audio CausalLM for FSDP actor logprob computation.

Uses the ConformerEncoder audio tower from vLLM's phi4mm_audio plus the
HF Qwen3.5 text backbone to compute logprobs conditioned on audio.

Audio features (mel filterbank, shape T×80) are passed in multi_modal_inputs
as "input_audio_embeds". They are processed through the ConformerEncoder +
MLP projection, then mean-pooled and injected at the <|audio_start|> token
position so the sequence length stays constant (compatible with remove_padding
and FSDP batching).

Key design choices:
- mean-pool audio embeddings → single vector per audio → no sequence expansion
- Works with use_remove_padding=True and all FSDP sharding modes
- Key mapping: model.embed_tokens_extend.* → embed_tokens_extend.*
  (applied in fsdp_workers.py at load time)
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# <|audio_start|> token ID in the Qwen3.5 tokenizer
_AUDIO_START_TOKEN_ID = 248070
# <|audio_pad|> token ID (used by vLLM internally, kept for reference)
_AUDIO_PAD_TOKEN_ID = 248076


class Qwen3_5AudioForCausalLMHF(PreTrainedModel):
    """Qwen3.5 text model + ConformerEncoder audio tower, HF-compatible.

    Loading:
        Use AutoModelForCausalLM or directly with from_pretrained().
        Set key_mapping = {r"^model\\.embed_tokens_extend\\.": "embed_tokens_extend."}
        in the load kwargs so the audio tower weights are routed correctly.

    Forward extra kwargs (from multi_modal_inputs):
        input_audio_embeds: (B, T, 80) mel filterbank features, or (T, 80) single
        audio_attention_mask: optional mask for mel frames
    """

    _keys_to_ignore_on_load_unexpected = []
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        # Text backbone (embed_tokens, layers, norm)
        self.model = Qwen3_5TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Audio tower: ConformerEncoder + MLP projection.
        # Attach to self.model so the checkpoint key prefix "model.embed_tokens_extend.*"
        # maps directly to self.model.embed_tokens_extend.* — no key remapping needed.
        self.model.embed_tokens_extend = self._build_audio_embedding(config)

        self.post_init()

    # ------------------------------------------------------------------
    # HF boilerplate
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ------------------------------------------------------------------
    # Audio tower construction
    # ------------------------------------------------------------------

    def _build_audio_embedding(self, config):
        """Build AudioEmbedding from vLLM's phi4mm_audio with correct compression."""
        from vllm.model_executor.models.phi4mm_audio import AudioEmbedding

        embd_layer = config.embd_layer if isinstance(config.embd_layer, dict) else {}
        embd_kwargs = {k: v for k, v in embd_layer.items() if k not in ("embedding_cls",)}

        # Map compression_rate → downsample_rate so the MLP projection uses
        # the correct input dimension: audio_dim_out * compression_rate
        if "compression_rate" in embd_kwargs and "downsample_rate" not in embd_kwargs:
            embd_kwargs["downsample_rate"] = embd_kwargs.pop("compression_rate")
        embd_kwargs.pop("enable_gradient_checkpointing", None)

        ae = AudioEmbedding(config, **embd_kwargs)
        # Remove vision projection (not needed for audio-only models)
        if hasattr(ae, "audio_projection_for_vision"):
            del ae.audio_projection_for_vision
        return ae

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        # Audio kwargs (passed via **multi_modal_inputs from dp_actor.py)
        input_audio_embeds=None,
        audio_attention_mask=None,
        **kwargs,
    ):
        # 1. Text token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # 2. Inject audio embeddings at <|audio_start|> positions (in-place)
        if input_audio_embeds is not None:
            inputs_embeds = self._inject_audio(
                input_ids, inputs_embeds, input_audio_embeds, audio_attention_mask
            )

        # 3. Text model forward (pass inputs_embeds, not input_ids)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        # 4. LM head
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _inject_audio(self, input_ids, inputs_embeds, audio_features, audio_attention_mask):
        """Process mel features through audio tower and inject at <|audio_start|>.

        Strategy: mean-pool the ConformerEncoder output so audio is represented
        as a single token vector. This keeps sequence length unchanged, which is
        required for remove_padding and standard FSDP batching.

        Args:
            input_ids: (B, S) or (1, S) if remove_padding packed
            inputs_embeds: (B, S, H) — modified in-place
            audio_features: (B, T, 80), (T, 80), or list of (T_i, 80)
            audio_attention_mask: optional mel mask (ignored — mel is already cropped)
        """
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        B = inputs_embeds.shape[0]

        # Normalise to list of 2-D mel tensors (one per batch item)
        mels = _normalise_audio_list(audio_features, B)

        for b in range(B):
            mel = mels[b]  # (T, 80) or None
            if mel is None:
                continue

            mel = mel.to(device=device, dtype=dtype)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # → (1, T, 80) for get_audio_features

            # ConformerEncoder + MLP projection → (N_tokens, hidden_size)
            # embed_tokens_extend.forward takes (T, 80) and returns (N, hidden)
            mel_2d = mel.squeeze(0)  # (T, 80)
            audio_embs = self.model.embed_tokens_extend(
                mel_2d,
                audio_attention_mask=None,
                audio_projection_mode="speech",
            )  # (N, H)

            # Mean-pool across N audio tokens → (H,) single vector
            audio_vec = audio_embs.mean(dim=0)  # (H,)

            # Find <|audio_start|> positions in this batch item
            if input_ids.dim() == 2:
                ids_b = input_ids[b]  # (S,)
            else:
                ids_b = input_ids[0]  # packed remove_padding case: scan all

            audio_positions = (ids_b == _AUDIO_START_TOKEN_ID).nonzero(as_tuple=False)
            if audio_positions.numel() == 0:
                continue

            # Replace each <|audio_start|> embedding with the audio vector
            for pos in audio_positions.squeeze(-1).tolist():
                if isinstance(pos, int):
                    inputs_embeds[b, pos] = audio_vec

        return inputs_embeds


def _normalise_audio_list(audio_features, batch_size: int):
    """Return a list of length batch_size of (T, 80) mel tensors or None."""
    if audio_features is None:
        return [None] * batch_size

    if isinstance(audio_features, torch.Tensor):
        if audio_features.dim() == 2:
            # (T, 80) — same audio for all batch items (broadcast)
            return [audio_features] * batch_size
        elif audio_features.dim() == 3:
            # (B, T, 80) — one per batch item
            return [audio_features[i] for i in range(min(audio_features.shape[0], batch_size))]
        else:
            return [None] * batch_size

    if isinstance(audio_features, (list, tuple)):
        result = list(audio_features)
        # Pad to batch_size if needed
        while len(result) < batch_size:
            result.append(None)
        return result[:batch_size]

    return [None] * batch_size

"""Qwen3.5-Audio embedding adapter for the bundled HF text model.

This module provides:
- ``Qwen3_5AudioEmbeddingHF``: A torch.nn.Module that wraps vLLM's AudioEmbedding
  and adapts it to the calling convention expected by phyagi's Qwen3_5TextModel.
- ``get_embedding``: A factory used by the bundled text model to create the
  embed_tokens_extend module.

The phyagi TextModel calls:
    outputs, ctc_loss = self.embed_tokens_extend(
        input_ids,
        input_embeds=mel_features,
        wte=self.embed_tokens,
        audio_attention_mask=...,
        ...
    )

Where:
- input_ids: (B, S) with <|audio_start|> (248070) marking audio injection sites
- input_embeds: mel filterbank features (B_audio, T, 80) or None
- wte: the token embedding layer

Our adapter:
1. Embeds text: text_embeds = wte(input_ids)  → (B, S, H)
2. Processes mel through the ConformerEncoder+MLP audio tower → (N, H) per clip
3. Mean-pools each clip's encoder output → single (H,) vector per clip
4. Injects it at the <|audio_start|> position for that batch item
5. Returns (text_embeds, None)
"""

import torch
import torch.nn as nn

# <|audio_start|> token ID in the Qwen3.5-Audio tokenizer
_AUDIO_START_TOKEN_ID = 248070


class Qwen3_5AudioEmbeddingHF(nn.Module):
    """Audio tower + text injection adapter for the bundled HF Qwen3.5 text model.

    Weight keys match the checkpoint's ``model.embed_tokens_extend.*`` hierarchy
    because we delegate to vLLM's ``AudioEmbedding`` which uses the same
    ConformerEncoder + MLP structure as the phyagi checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self._audio_tower = self._build_audio_tower(config)

    def _build_audio_tower(self, config):
        """Create vLLM's AudioEmbedding with config from embd_layer dict."""
        from vllm.model_executor.models.phi4mm_audio import AudioEmbedding

        embd_layer = config.embd_layer if isinstance(config.embd_layer, dict) else {}
        embd_kwargs = {k: v for k, v in embd_layer.items() if k not in ("embedding_cls",)}

        # phyagi uses "compression_rate"; vLLM uses "downsample_rate"
        if "compression_rate" in embd_kwargs and "downsample_rate" not in embd_kwargs:
            embd_kwargs["downsample_rate"] = embd_kwargs.pop("compression_rate")
        embd_kwargs.pop("enable_gradient_checkpointing", None)

        ae = AudioEmbedding(config, **embd_kwargs)
        if hasattr(ae, "audio_projection_for_vision"):
            del ae.audio_projection_for_vision
        return ae

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeds: "torch.FloatTensor | None" = None,
        wte: "nn.Embedding | None" = None,
        audio_attention_mask=None,
        **kwargs,
    ):
        """Embed tokens and inject audio at <|audio_start|> positions.

        Args:
            input_ids: (B, S) — text token IDs with <|audio_start|> at audio sites.
            input_embeds: Mel filterbank features — (B, T, 80), (T, 80), or list. May be None.
            wte: Token embedding layer. If None, raises RuntimeError.
            audio_attention_mask: Ignored (mel is already padded/cropped).

        Returns:
            Tuple of (hidden_states, ctc_loss) where ctc_loss is always None.
        """
        if wte is None:
            raise RuntimeError("Qwen3_5AudioEmbeddingHF.forward requires wte (token embedding layer)")

        # 1. Text embeddings
        text_embeds = wte(input_ids)  # (B, S, H)

        # 2. Early exit if no audio
        if input_embeds is None:
            return text_embeds, None

        B = text_embeds.shape[0]
        device = text_embeds.device
        dtype = text_embeds.dtype

        # 3. Normalise mel features to a list of per-batch-item tensors
        mels = _normalise_mel_list(input_embeds, B)

        # 4. Process each audio clip and inject
        for b in range(B):
            mel = mels[b]
            if mel is None:
                continue

            mel = mel.to(device=device, dtype=dtype)
            if mel.dim() == 3:
                mel = mel.squeeze(0)  # (T, 80)
            assert mel.dim() == 2, f"Expected 2-D mel, got {mel.shape}"

            # ConformerEncoder + MLP → (N, H)
            audio_tokens = self._audio_tower(mel, audio_projection_mode="speech")

            # Mean-pool to single vector
            audio_vec = audio_tokens.mean(dim=0).to(dtype=dtype)  # (H,)

            # Find <|audio_start|> positions in this batch item
            audio_positions = (input_ids[b] == _AUDIO_START_TOKEN_ID).nonzero(as_tuple=False)
            for pos_t in audio_positions.squeeze(-1).tolist():
                pos = int(pos_t)
                text_embeds[b, pos] = audio_vec

        return text_embeds, None


def _normalise_mel_list(audio_features, batch_size: int) -> list:
    """Convert various mel representations to a list of (T, 80) tensors or None."""
    if audio_features is None:
        return [None] * batch_size

    if isinstance(audio_features, torch.Tensor):
        if audio_features.dim() == 2:
            return [audio_features] * batch_size  # broadcast
        if audio_features.dim() == 3:
            n = audio_features.shape[0]
            result = [audio_features[i] for i in range(min(n, batch_size))]
            result += [None] * max(0, batch_size - n)
            return result
        return [None] * batch_size

    if isinstance(audio_features, (list, tuple)):
        result = list(audio_features)
        while len(result) < batch_size:
            result.append(None)
        return result[:batch_size]

    return [None] * batch_size


def get_embedding(config, embedding_config=None):
    """Factory matching phyagi's get_embedding signature.

    Called by bundled Qwen3_5TextModel.__init__ to create embed_tokens_extend.
    Only 'audio' embedding class is supported.
    """
    cls = "default"
    if isinstance(embedding_config, dict):
        cls = embedding_config.get("embedding_cls", "default")

    if cls in ("audio",):
        return Qwen3_5AudioEmbeddingHF(config)

    # Fallback: identity — just does text embedding and returns zeros for ctc_loss.
    # This should not be reached for Qwen3.5-Audio checkpoints.
    import warnings
    warnings.warn(
        f"Unknown embedding_cls={cls!r}; using identity fallback for embed_tokens_extend.",
        stacklevel=2,
    )
    return _IdentityEmbedding()


class _IdentityEmbedding(nn.Module):
    """Fallback: only text embedding, no audio injection."""

    def forward(self, input_ids, input_embeds=None, wte=None, **kwargs):
        if wte is not None:
            return wte(input_ids), None
        raise RuntimeError("_IdentityEmbedding requires wte")

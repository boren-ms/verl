# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Self-contained AudioEmbedding for HF trust_remote_code deployment."""

from __future__ import annotations

import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cascade_encoder import ConformerEncoder, NemoConvSubsampling
from .processing_qwen3_5_audio import AUDIO_PAD_TOKEN_ID

logger = logging.getLogger(__name__)


def _match_audio_positions(positions, audio_embed_sizes):
    expected = int(audio_embed_sizes.sum().item())
    if expected == len(positions):
        return positions

    matched = []
    cursor = 0
    for size in audio_embed_sizes.tolist():
        size = int(size)
        if size == 0:
            continue

        while cursor + size <= len(positions):
            candidate = positions[cursor:cursor + size]
            same_row = bool(torch.all(candidate[:, 0] == candidate[0, 0]))
            contiguous = size == 1 or bool(torch.all(candidate[1:, 1] == candidate[:-1, 1] + 1))
            if same_row and contiguous:
                matched.append(candidate)
                cursor += size
                break
            cursor += 1
        else:
            raise ValueError(
                f"Could not match an audio placeholder block of size {size}: "
                f"expected {expected} positions, found {len(positions)}."
            )

    return torch.cat(matched, dim=0) if matched else positions[:0]


class AudioEmbedding(nn.Module):
    """Audio embedding layer — encodes audio features and splices them into text embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        if hasattr(config, "embd_pdrop"):
            self.drop = nn.Dropout(config.embd_pdrop) if config.embd_pdrop > 0 else None
        else:
            self.drop = None

        audio_dim_out = None
        self.layer_idx = -2

        if isinstance(config.audio_processor, dict) and config.audio_processor.get("name") == "cascades":
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None
            self.encoder = ConformerEncoder(**encoder_config)
            self.encoder.post_init({})
            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        elif isinstance(config.audio_processor, dict) and config.audio_processor.get("name") == "whisper":
            from transformers import WhisperModel as HFWhisperModel
            model_path = config.audio_processor.get("pretrained_model_path")
            whisper_model = HFWhisperModel.from_pretrained(model_path)
            self.encoder = whisper_model.encoder
            n_mels = self.encoder.num_mel_bins
            audio_dim_out = self.encoder.layers[0].embed_dim
        else:
            raise NotImplementedError(f"Unsupported audio_processor: {config.audio_processor}")

        assert audio_dim_out is not None
        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        self.freeze_audio_processor = kwargs.get("freeze_audio_processor", False)
        self.zero_audio_projection = kwargs.get("zero_audio_projection", False)
        self.downsample_rate = kwargs.get("downsample_rate", 1)

        self.ctc_weight = kwargs.get("ctc_weight", 0.0)
        self.use_ctc = self.ctc_weight > 0.0
        if self.use_ctc:
            self.ctc_linear = nn.Linear(audio_dim_out, config.vocab_size)
            self.blank_id = kwargs.get("blank_id", config.vocab_size - 1)
        else:
            self.ctc_linear = None
            self.blank_id = None

        self.qformer = None
        self.conv_ds = None

        if kwargs.get("use_conv_downsample", False):
            nemo_conv_settings = kwargs.get("nemo_conv_settings", {})
            default_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.downsample_rate,
                "feat_in": audio_dim_out,
                "feat_out": audio_dim_out,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            if nemo_conv_settings:
                default_settings.update(nemo_conv_settings)
            self.conv_ds = NemoConvSubsampling(**default_settings)

        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
            self.linear_downsample_rate = 1
        elif projection_cls == "mlp":
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = 1 if (self.qformer or self.conv_ds) else self.downsample_rate
            layers = [nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection)])
            self.audio_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}")

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        if audio_config.get("name") == "cascades":
            init_model_config = audio_config.get("init_model", {})
            self.encoder.post_init(init_model_config)
            if "init_model" in audio_config:
                audio_config.pop("init_model")

        if self.zero_audio_projection:
            if isinstance(self.audio_projection, nn.Sequential):
                for layer in self.audio_projection:
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            else:
                nn.init.zeros_(self.audio_projection.weight)
                if self.audio_projection.bias is not None:
                    nn.init.zeros_(self.audio_projection.bias)

    def set_audio_embeds(self, input_embeds, audio_attention_mask):
        self.input_embeds = input_embeds
        self.audio_attention_mask = audio_attention_mask

    def set_audio_embed_sizes(self, audio_embed_sizes):
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(self, input_embeds, audio_attention_mask, audio_embed_sizes=None, **kwargs):
        if self.freeze_audio_processor:
            with torch.no_grad():
                audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        if self.ctc_weight > 0.0 and self.training:
            ctc_loss_fct = nn.CTCLoss(reduction="mean", blank=self.blank_id, zero_infinity=True)
            ctc_logits = F.log_softmax(self.ctc_linear(audio_features), dim=-1)
            input_lengths = audio_embed_sizes
            label_lengths = kwargs.get("ctc_label_lens")
            ctc_loss = ctc_loss_fct(ctc_logits.to(dtype=torch.float32).transpose(0, 1), kwargs.get("ctc_labels"), input_lengths, label_lengths)
            ctc_loss = ctc_loss.to(dtype=ctc_logits.dtype, device=ctc_logits.device)
        else:
            ctc_loss = None

        if self.qformer is not None:
            audio_features, _ = self.qformer(audio_features, mask=None)

        if self.conv_ds is not None:
            if masks is not None:
                masks = masks.squeeze(1)
            audio_features, masks = self.conv_ds(audio_features, mask=masks)

        if self.linear_downsample_rate != 1:
            bs, seq_len, feat_dim = audio_features.size()
            padding = seq_len % self.linear_downsample_rate
            if padding > 0:
                audio_features = F.pad(audio_features, (0, 0, 0, self.linear_downsample_rate - padding), "constant", 0)
            seq_len = audio_features.size(1)
            audio_features = audio_features.view(bs, seq_len // self.linear_downsample_rate, feat_dim * self.linear_downsample_rate)

        audio_set_tensor = self.audio_projection(audio_features)
        return audio_set_tensor, ctc_loss

    def forward(self, input_ids, input_embeds=None, audio_embed_sizes=None, audio_attention_mask=None, **kwargs):
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.clone()
            audio_attention_mask = self.audio_attention_mask.clone()
        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.clone()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        with torch.no_grad():
            positions = torch.nonzero(input_ids == AUDIO_PAD_TOKEN_ID, as_tuple=False)

        if isinstance(self.audio_projection, nn.Sequential):
            target_device = self.audio_projection[0].weight.device
            target_dtype = self.audio_projection[0].weight.dtype
        else:
            target_device = self.audio_projection.weight.device
            target_dtype = self.audio_projection.weight.dtype

        if input_embeds is not None:
            input_embeds = input_embeds.to(target_device).to(target_dtype)

        if len(positions.tolist()) > 0:
            audio_set_tensor, ctc_loss = self.get_audio_features(input_embeds, audio_attention_mask, audio_embed_sizes=audio_embed_sizes, **kwargs)
        else:
            ctc_loss = None
            if self.training:
                input_embeds = torch.zeros(1, 8, self.audio_dim_in).to(target_device).to(target_dtype)
                audio_attention_mask = input_embeds.new_ones(input_embeds.size()[:2]).long()
                audio_set_tensor, ctc_loss = self.get_audio_features(input_embeds, audio_attention_mask, audio_embed_sizes=audio_embed_sizes, **kwargs)

        with torch.no_grad():
            input_ids = input_ids.clone()
            input_ids[input_ids == AUDIO_PAD_TOKEN_ID] = 0
            input_ids.clamp_max_(self.vocab_size)

        if "wte" in kwargs:
            hidden_states = kwargs["wte"](input_ids)
        else:
            hidden_states = self.wte(input_ids)

        hidden_states = hidden_states.clone()
        if len(positions.tolist()) > 0:
            matched_positions = _match_audio_positions(positions, audio_embed_sizes)
            if len(matched_positions) != len(positions):
                logger.warning(
                    "Ignoring %s stray AUDIO_PAD token position(s) outside the expected audio placeholder blocks.",
                    len(positions) - len(matched_positions),
                )
            positions = matched_positions
            idx = 0
            audio_idx = 0
            for i in range(len(audio_embed_sizes)):
                cnt = audio_embed_sizes[i]
                if cnt == 0:
                    continue
                hidden_states[positions[idx, 0], positions[idx, 1]:positions[idx, 1] + cnt] = (
                    audio_set_tensor[audio_idx, :cnt, :].to(hidden_states.dtype).to(hidden_states.device)
                )
                idx += cnt
                audio_idx += 1
        else:
            if self.training:
                hidden_states[:, 0:1] = hidden_states[:, 0:1] + 0 * audio_set_tensor[:, 0:1].to(hidden_states.dtype).to(hidden_states.device)

        if not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states, ctc_loss

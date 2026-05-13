# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.5 + Audio (Conformer) model for vLLM.

This model uses the Qwen3.5 hybrid LLM backbone (full attention + GatedDeltaNet
linear attention) with a Conformer-based audio encoder and MLP projection,
similar to the Phi4-MM audio architecture but with a different LLM backbone.
"""

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.phi4mm_audio import (
    AudioEmbedding as _AudioEmbeddingBase,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    ResolvedPromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape


class AudioEmbedding(_AudioEmbeddingBase):
    """AudioEmbedding without the vision projection layer.

    Subclasses Phi4-MM's AudioEmbedding and removes the
    ``audio_projection_for_vision`` parameter after init, since
    audio-only models don't need it and have no checkpoint weights for it.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Remove vision projection (not used for audio-to-text,
        # and no weights exist in checkpoint)
        if hasattr(self, "audio_projection_for_vision"):
            del self.audio_projection_for_vision


# Audio placeholder token ID used for embedding replacement.
# <|audio_pad|> token in the Qwen3.5 tokenizer.
# Shared definition lives in hf_model/configuration_qwen3_5_audio.py (AUDIO_PAD_TOKEN_ID).
_AUDIO_PLACEHOLDER_TOKEN_ID = 248076

_AUDIO_MAX_SOUNDFILE_SIZE = 241_000

# SpeechLib-compatible log filterbank feature extraction parameters
_AUDIO_SAMPLING_RATE = 16000
_AUDIO_N_MELS = 80
_AUDIO_HOP_LENGTH = 160
_AUDIO_WIN_LENGTH = 400
_AUDIO_N_FFT = 512
_AUDIO_PREEMPHASIS = 0.97


def _speechlib_mel(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float | None = None,
    fmax: float | None = 7690.0,
) -> np.ndarray:
    """Create a Mel filter-bank matching SpeechLib FbankFC."""
    bank_width = n_fft // 2 + 1
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0.0

    def mel(f: float) -> float:
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin: int) -> float:
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f: float) -> int:
        return int((f * n_fft / sample_rate) + 0.5)

    klo = f2bin(fmin) + 1
    khi = max(f2bin(fmax), klo)

    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = m_centers[m], m_centers[m + 1], m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms
    return matrix


# Pre-computed mel filter bank (transposed for dot product: [n_fft//2+1, 80])
_MEL_FILTER = _speechlib_mel(_AUDIO_SAMPLING_RATE, _AUDIO_N_FFT, _AUDIO_N_MELS)
_MEL_FILTER_T = _MEL_FILTER.T
_HAMMING_400 = np.hamming(_AUDIO_WIN_LENGTH).astype(np.float32)


def extract_logfbank(wav: np.ndarray, fs: int) -> np.ndarray:
    """Extract log mel filterbank features (SpeechLib-compatible).

    Args:
        wav: 1-D float32 waveform.
        fs: Sample rate (Hz). Resampled to 16 kHz if needed.

    Returns:
        log_fbank: (T, 80) float32 array.
    """
    if wav.ndim > 1:
        wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)
    wav = wav.astype(np.float32)

    if fs > 16000:
        wav = scipy.signal.resample_poly(wav, 16000, fs).astype(np.float32)
    elif 8000 <= fs < 16000:
        wav = scipy.signal.resample_poly(wav, 2, 1).astype(np.float32)
    elif fs < 8000:
        raise RuntimeError(f"Unsupported sample rate {fs}")

    n_batch = (wav.shape[0] - _AUDIO_WIN_LENGTH) // _AUDIO_HOP_LENGTH + 1
    y_frames = np.array(
        [wav[s : s + _AUDIO_WIN_LENGTH] for s in range(0, _AUDIO_HOP_LENGTH * n_batch, _AUDIO_HOP_LENGTH)],
        dtype=np.float32,
    )

    # Preemphasis within each frame
    y_prev = np.roll(y_frames, 1, axis=1)
    y_prev[:, 0] = y_prev[:, 1]
    y_frames = (y_frames - _AUDIO_PREEMPHASIS * y_prev) * 32768

    S = np.fft.rfft(_HAMMING_400 * y_frames, n=_AUDIO_N_FFT, axis=1)
    spec_power = np.abs(S).astype(np.float32) ** 2

    fbank_power = np.clip(spec_power.dot(_MEL_FILTER_T), 1.0, None)
    return np.log(fbank_power).astype(np.float32)


class Qwen3_5AudioFeatureInputs(TensorSchema):
    """Audio feature inputs (mel spectrograms) for Qwen3.5 audio model."""

    type: Literal["audio_features"]

    audio_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "t", _AUDIO_N_MELS, dynamic_dims={"t"}),
    ]
    audio_attention_mask: torch.Tensor | list[torch.Tensor] | None
    audio_embed_sizes: torch.Tensor | list[int] | None


class Qwen3_5AudioEmbeddingInputs(TensorSchema):
    """Pre-computed audio embedding inputs."""

    type: Literal["audio_embeds"]
    data: Annotated[
        NestedTensors,
        TensorShape("b", "n", "f", "h"),
    ]


Qwen3_5AudioInputs: TypeAlias = Qwen3_5AudioFeatureInputs | Qwen3_5AudioEmbeddingInputs


class Qwen3_5AudioProcessingInfo(BaseProcessingInfo):
    @property
    def audio_tokens(self) -> list[str]:
        # Use <audio> as user-facing audio placeholder.
        # Each audio input maps to one occurrence of this token in the prompt.
        return ["<audio>"] * 100

    def get_feature_extractor(self, **kwargs: object) -> None:
        return None

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=_AUDIO_SAMPLING_RATE,
            audio_resample_method="scipy",
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_audio_num_frames(self, audio_len: int, sr: float) -> int:
        """Compute the number of spectrogram time frames (SpeechLib-style)."""
        if sr > 16000:
            audio_len = int(audio_len * 16000 / sr)
        elif 8000 <= sr < 16000:
            audio_len *= 2
        elif sr < 8000:
            raise RuntimeError(f"Unsupported sample rate {sr}")

        num_frames = (audio_len - _AUDIO_WIN_LENGTH) // _AUDIO_HOP_LENGTH + 1
        if num_frames < 1:
            raise ValueError("Waveform too short for given parameters.")

        return num_frames

    def _compute_audio_embed_size(self, audio_frames: int) -> int:
        """Compute token count after Conformer encoder compression."""
        hf_config = self.get_hf_config()
        compression_rate = hf_config.embd_layer.get("compression_rate", 8)
        integer = audio_frames // compression_rate
        remainder = audio_frames % compression_rate
        return integer if remainder == 0 else integer + 1


class Qwen3_5AudioDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3_5AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        audio_tokens: list[str] = self.info.audio_tokens[:num_audios]
        return "".join(audio_tokens)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio")

        return {
            "audio": self._get_dummy_audios(
                length=_AUDIO_MAX_SOUNDFILE_SIZE,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


class Qwen3_5AudioMultiModalProcessor(BaseMultiModalProcessor[Qwen3_5AudioProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process text and audio data into model inputs.

        Since this model does not have a registered HuggingFace processor,
        we handle tokenization and log-filterbank feature extraction directly.
        Feature extraction uses the SpeechLib-compatible LogFbank pipeline
        (same as Phi4-MM).
        """
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer.encode(prompt)

        if not mm_data or not mm_data.get("audios"):
            return BatchFeature(dict(input_ids=[input_ids]), tensor_type="pt")

        audio_data = mm_data.get("audios", [])

        audio_features_list = []
        audio_embed_sizes = []
        audio_frames = []
        for audio in audio_data:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio_np = np.asarray(audio, dtype=np.float32)

            # Extract SpeechLib-style log mel filterbank features (T, 80)
            log_fbank = extract_logfbank(audio_np, _AUDIO_SAMPLING_RATE)
            num_frames = log_fbank.shape[0]
            audio_features_list.append(torch.from_numpy(log_fbank).float())
            audio_frames.append(num_frames)
            audio_embed_sizes.append(self.info._compute_audio_embed_size(num_frames))

        input_audio_embeds = torch.nn.utils.rnn.pad_sequence(audio_features_list, batch_first=True)
        max_frames = input_audio_embeds.shape[1]
        audio_attention_mask = torch.zeros(len(audio_features_list), max_frames, dtype=torch.long)
        for idx, num_frames in enumerate(audio_frames):
            audio_attention_mask[idx, :num_frames] = 1

        return BatchFeature(
            dict(
                input_ids=[input_ids],
                input_audio_embeds=input_audio_embeds,
                audio_embed_sizes=torch.tensor(audio_embed_sizes, dtype=torch.long),
                audio_attention_mask=audio_attention_mask,
            ),
            tensor_type="pt",
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # We handle prompt updates ourselves via _get_prompt_updates
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_audio_embeds=MultiModalFieldConfig.batched("audio"),
            audio_embed_sizes=MultiModalFieldConfig.batched("audio"),
            audio_attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audio_tokens: list[str] = self.info.audio_tokens

        def get_audio_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            audio_frames = self.info.get_audio_num_frames(audio_len, _AUDIO_SAMPLING_RATE)
            audio_embed_size = self.info._compute_audio_embed_size(audio_frames)
            return [_AUDIO_PLACEHOLDER_TOKEN_ID] * audio_embed_size

        return [
            PromptReplacement(
                modality="audio",
                target=audio_tokens.__getitem__,
                replacement=get_audio_replacement,
            ),
        ]

    def _recompute_cached_prompt_update(
        self,
        cached_update: ResolvedPromptUpdate,
        new_item_idx: int,
    ) -> ResolvedPromptUpdate:
        new_update = super()._recompute_cached_prompt_update(
            cached_update,
            new_item_idx,
        )

        if cached_update.modality == "audio":
            audio_tokens: list[str] = self.info.audio_tokens
            new_update = new_update.with_target(audio_tokens[new_item_idx])

        return new_update


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3_5AudioMultiModalProcessor,
    info=Qwen3_5AudioProcessingInfo,
    dummy_inputs=Qwen3_5AudioDummyInputsBuilder,
)
class Qwen3_5AudioForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsMRoPE):
    """Qwen3.5 + Audio Encoder model for speech-to-text tasks.

    Uses the Qwen3.5 hybrid backbone (full attention + GatedDeltaNet) with a
    Conformer-based audio encoder for multimodal audio understanding.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "base_layer.": "",
        },
        orig_to_new_prefix={
            "model.embed_tokens_extend.": "embed_tokens_extend.",
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.layers.": "language_model.model.layers.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
        },
    )

    supports_mrope: ClassVar[Literal[True]] = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<audio>"
        raise ValueError("Only audio modality is supported")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        # Audio is 1D (temporal only) — treat all tokens (text and audio)
        # as sequential positions across all 3 M-RoPE sections.
        n = len(input_tokens)
        positions = np.arange(n, dtype=np.int64)
        llm_positions = np.broadcast_to(positions, (3, n))
        return torch.from_numpy(np.array(llm_positions)), 0

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config, "multimodal_config is required"
        quant_config = vllm_config.quant_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.vllm_config = vllm_config

        assert get_pp_group().world_size == 1, "pipeline parallel is not supported"

        # Build audio embedding (Conformer encoder + MLP projection)
        if isinstance(config.embd_layer, dict):
            embedding_config = {
                "embedding_cls": config.embd_layer["embedding_cls"],
                **config.embd_layer,
            }
        else:
            embedding_config = {
                "embedding_cls": config.embd_layer,
            }

        mark_tower_model = getattr(self, "_mark_tower_model", lambda *_args, **_kwargs: nullcontext())
        with mark_tower_model(vllm_config, "audio"):
            self.embed_tokens_extend = AudioEmbedding(config, **embedding_config)

        config.model_type = "qwen3_5_text"

        # Build language model (Qwen3.5 hybrid backbone)
        mark_language_model = getattr(self, "_mark_language_model", lambda *_args, **_kwargs: nullcontext())
        with mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(vllm_config=vllm_config, prefix=maybe_prefix(prefix, ""))

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Qwen3_5AudioInputs | None:
        audio_features = kwargs.pop("input_audio_embeds", None)
        audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        audio_embed_sizes = kwargs.pop("audio_embed_sizes", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            return Qwen3_5AudioFeatureInputs(
                type="audio_features",
                audio_features=audio_features,
                audio_attention_mask=audio_attention_mask,
                audio_embed_sizes=audio_embed_sizes,
            )

        if audio_embeds is not None:
            return Qwen3_5AudioEmbeddingInputs(type="audio_embeds", data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(self, audio_input: Qwen3_5AudioInputs) -> NestedTensors:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["audio_features"]
        audio_attention_mask = audio_input.get("audio_attention_mask")
        audio_embed_sizes = audio_input.get("audio_embed_sizes")
        dtype = next(self.embed_tokens_extend.parameters()).dtype
        if isinstance(audio_features, torch.Tensor):
            audio_features = list(audio_features)
        if isinstance(audio_attention_mask, torch.Tensor):
            audio_attention_mask = list(audio_attention_mask)
        if isinstance(audio_embed_sizes, torch.Tensor):
            audio_embed_sizes = audio_embed_sizes.flatten().tolist()

        def get_audio_attention_mask(idx: int, features: torch.Tensor):
            if audio_attention_mask is None:
                return None
            mask = audio_attention_mask[idx].to(features.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            return mask

        audio_embeds = [
            self.embed_tokens_extend(
                features.to(dtype),
                audio_attention_mask=get_audio_attention_mask(idx, features),
                audio_projection_mode="speech",
            )[: int(audio_embed_sizes[idx])]
            if audio_embed_sizes is not None
            else self.embed_tokens_extend(
                features.to(dtype),
                audio_attention_mask=get_audio_attention_mask(idx, features),
                audio_projection_mode="speech",
            )
            for idx, features in enumerate(audio_features)
        ]
        return audio_embeds

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_embeddings = self._process_audio_input(audio_input)
        return tuple(audio_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp.", "lora_A.", "lora_B."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector=["embed_tokens_extend.audio_projection"],
            tower_model=["embed_tokens_extend.encoder"],
        )

    # Required by HasInnerState / IsHybrid for mamba cache
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig") -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig") -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()

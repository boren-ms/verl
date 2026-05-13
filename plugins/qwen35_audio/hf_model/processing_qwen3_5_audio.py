# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Qwen3.5 Audio Processor for HF AutoProcessor with trust_remote_code.

Usage:
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    inputs = processor(text=prompt, audios=[(wav, sr)], return_tensors="pt")
    outputs = model.generate(**inputs.to("cuda"))
    text = processor.batch_decode(outputs, skip_special_tokens=True)
"""

from __future__ import annotations

import re
from itertools import chain, zip_longest
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from transformers import AutoTokenizer
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


# ---------------------------------------------------------------------------
# Audio feature extraction (log-fbank, matching LogFbankProcessor)
# ---------------------------------------------------------------------------

def _speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)
    khi = max(khi, klo)
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)
    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms
    return matrix


class Qwen3_5AudioFeatureExtractor(SequenceFeatureExtractor):
    """Extract 80-dim log-mel filterbank features from raw waveforms."""

    model_input_names = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask", "audio_frames"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        compression_rate=8,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.compression_rate = compression_rate
        self._mel_filter = _speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T
        self._hamming400 = np.hamming(400)

    def _extract_logfbank(self, wav: np.ndarray, fs: int) -> np.ndarray:
        import scipy.signal

        if wav.ndim > 1:
            wav = np.squeeze(wav)
        if len(wav.shape) == 2:
            wav = wav.mean(1)
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 16000, fs)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 8000, fs)
            fs = 8000
        if fs == 8000:
            wav = scipy.signal.resample_poly(wav, 2, 1)
            fs = 16000

        preemphasis = 0.97
        n_fft, win_length, hop_length = 512, 400, 160

        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        y_frames = np.array(
            [wav[s: s + win_length] for s in range(0, hop_length * n_batch, hop_length)],
            dtype=np.float32,
        )
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        S = np.fft.rfft(self._hamming400 * y_frames, n=n_fft, axis=1).astype(np.complex64)
        spec = np.abs(S).astype(np.float32)
        spec_power = spec ** 2
        fbank_power = np.clip(spec_power.dot(self._mel_filter), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)
        return log_fbank

    def _compute_audio_embed_size(self, audio_frames: int) -> int:
        integer = audio_frames // self.compression_rate
        remainder = audio_frames % self.compression_rate
        return integer if remainder == 0 else integer + 1

    def __call__(
        self,
        audios: List[Tuple[np.ndarray, int]],
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """Process a list of (waveform, sample_rate) tuples.

        Returns:
            BatchFeature with input_audio_embeds, audio_embed_sizes, audio_attention_mask, audio_frames.
        """
        all_features = []
        embed_sizes = []
        frame_counts = []

        for wav, sr in audios:
            if isinstance(wav, torch.Tensor):
                wav = wav.numpy()
            wav = wav.astype(np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            feat = self._extract_logfbank(wav, sr)
            all_features.append(feat)
            n_frames = len(feat)
            frame_counts.append(n_frames)
            embed_sizes.append(self._compute_audio_embed_size(n_frames))

        # Pad to same length
        max_frames = max(f.shape[0] for f in all_features)
        padded = []
        masks = []
        for feat in all_features:
            pad_len = max_frames - feat.shape[0]
            if pad_len > 0:
                feat = np.pad(feat, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            padded.append(feat)
            mask = np.ones(max_frames, dtype=np.int64)
            if pad_len > 0:
                mask[-pad_len:] = 0
            masks.append(mask)

        data = {
            "input_audio_embeds": np.stack(padded),
            "audio_embed_sizes": np.array(embed_sizes, dtype=np.int64),
            "audio_attention_mask": np.stack(masks),
            "audio_frames": np.array(frame_counts, dtype=np.int64),
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class Qwen3_5AudioProcessor(ProcessorMixin):
    """HF-compatible processor for Qwen3.5 Audio models.

    Combines tokenizer + audio feature extraction into a single `__call__`.
    Compatible with `AutoProcessor.from_pretrained(..., trust_remote_code=True)`.
    """

    attributes = ["audio_feature_extractor", "tokenizer"]
    audio_feature_extractor_class = "Qwen3_5AudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    AUDIO_TAG = "<audio>"

    def __init__(self, audio_feature_extractor, tokenizer, **kwargs):
        super().__init__(audio_feature_extractor, tokenizer, **kwargs)
        self.audio_feature_extractor = audio_feature_extractor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load feature extractor config
        import json, os
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        fe_kwargs = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            fe_kwargs = {k: v for k, v in cfg.items() if k not in ("feature_extractor_type", "processor_class", "auto_map")}

        audio_feature_extractor = Qwen3_5AudioFeatureExtractor(**fe_kwargs)
        return cls(audio_feature_extractor=audio_feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audios: Optional[List[Tuple[np.ndarray, int]]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """Process text + audio inputs for the model.

        Args:
            text: Prompt string(s) containing ``<audio>`` placeholder(s).
            audios: List of ``(waveform_ndarray, sample_rate)`` tuples.
            return_tensors: ``"pt"`` for PyTorch tensors.

        Returns:
            BatchFeature with input_ids, attention_mask, and audio fields.
        """
        if text is None:
            raise ValueError("text is required")

        if isinstance(text, str):
            text = [text]

        # --- Extract audio features ---
        audio_data = None
        if audios is not None and len(audios) > 0:
            audio_data = self.audio_feature_extractor(audios, return_tensors=return_tensors)

        # --- Tokenize with audio placeholder insertion ---
        all_input_ids = []
        all_embed_sizes = []
        audio_idx = 0

        for prompt in text:
            text_parts = re.split(r"<audio>", prompt)
            text_chunks = [self.tokenizer(part).input_ids for part in text_parts]

            n_audios_in_prompt = len(text_parts) - 1
            pad_chunks = []
            for _ in range(n_audios_in_prompt):
                if audio_data is not None and audio_idx < len(audio_data["audio_embed_sizes"]):
                    sz = int(audio_data["audio_embed_sizes"][audio_idx])
                    audio_idx += 1
                else:
                    sz = 0
                pad_chunks.append([248076] * sz)  # <|audio_pad|> token id
                all_embed_sizes.append(sz)

            interleaved = list(filter(None, chain.from_iterable(zip_longest(text_chunks, pad_chunks))))
            ids = list(chain(*interleaved))
            all_input_ids.append(ids)

        # --- Pad to same length (left-pad for generation) ---
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        padded_ids = []
        attn_masks = []
        for ids in all_input_ids:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attn_masks.append([0] * pad_len + [1] * len(ids))

        result = {
            "input_ids": padded_ids,
            "attention_mask": attn_masks,
        }

        if audio_data is not None:
            result["input_audio_embeds"] = audio_data["input_audio_embeds"]
            result["audio_embed_sizes"] = audio_data["audio_embed_sizes"]
            result["audio_attention_mask"] = audio_data["audio_attention_mask"]
            result["audio_frames"] = audio_data["audio_frames"]

        batch = BatchFeature(data=result, tensor_type=return_tensors)
        return batch

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

"""Exercise vLLM's normalized audio callback without loading model weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from vllm.multimodal.parse import MultiModalDataParser

def test_vllm_normalized_audio_processing_preserves_ragged_features():
    from vllm_qwen35_audio.plugin import register

    register()
    from vllm_qwen35_audio.qwen3_5_audio import Qwen3_5AudioMultiModalProcessor

    processor = Qwen3_5AudioMultiModalProcessor.__new__(Qwen3_5AudioMultiModalProcessor)
    # No tokenizer is provided: vLLM already tokenized the original prompt.
    processor.info = SimpleNamespace(_compute_audio_embed_size=lambda frames: (frames + 7) // 8)
    parser = MultiModalDataParser(target_sr=16000, audio_resample_method="scipy")
    audio = [np.zeros(16000, dtype=np.float32), np.zeros(8000, dtype=np.float32)]
    items = parser.parse_mm_data({"audio": audio})

    actual = processor._apply_hf_processor_main(items, {})

    assert "input_ids" not in actual
    assert actual["input_audio_embeds"].shape[0] == 2
    assert torch.isfinite(actual["input_audio_embeds"]).all()
    for index, waveform in enumerate(audio):
        independent = processor._apply_hf_processor_main(parser.parse_mm_data({"audio": [waveform]}), {})
        frames = independent["input_audio_embeds"].shape[1]
        torch.testing.assert_close(actual["input_audio_embeds"][index, :frames], independent["input_audio_embeds"][0])
        assert actual["audio_attention_mask"][index].sum().item() == frames
        assert actual["audio_embed_sizes"][index] == independent["audio_embed_sizes"][0]
    assert actual["audio_attention_mask"][0].sum() > actual["audio_attention_mask"][1].sum()


def test_vllm_prompt_replacement_targets_token_ids():
    from vllm_qwen35_audio.plugin import register

    register()
    from vllm_qwen35_audio.qwen3_5_audio import AUDIO_PAD_TOKEN_ID, Qwen3_5AudioMultiModalProcessor

    tokenizer = Mock()
    tokenizer.encode.return_value = [123]
    processor = Qwen3_5AudioMultiModalProcessor.__new__(Qwen3_5AudioMultiModalProcessor)
    processor.info = SimpleNamespace(
        audio_tokens=["<audio>"],
        get_tokenizer=lambda: tokenizer,
        get_audio_num_frames=lambda length, sr: 98,
        _compute_audio_embed_size=lambda frames: 13,
    )
    items = MultiModalDataParser(target_sr=16000).parse_mm_data({"audio": [np.zeros(16000, dtype=np.float32)]})

    update = processor._get_prompt_updates(items, {}, {})[0].resolve(0)

    assert update.target == [123]
    assert update.content.full == [AUDIO_PAD_TOKEN_ID] * 13
    tokenizer.encode.assert_called_once_with("<audio>", add_special_tokens=False)

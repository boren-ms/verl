"""Exercise native request budgets without loading model weights."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("vllm")

from vibevoice_asr import agent_loop  # noqa: E402


def test_offline_checkpoint_processor_preserves_tokenizer_assets(tmp_path):
    from hf_vibevoice_asr.rl import VibeVoiceASRProcessor

    processor = VibeVoiceASRProcessor.__new__(VibeVoiceASRProcessor)
    processor.speech_tok_compress_ratio = 3200
    processor.target_sample_rate = 24000
    processor.normalize_audio = True
    processor.tokenizer = Mock()
    processor.save_pretrained(tmp_path)
    assert (tmp_path / "preprocessor_config.json").is_file()
    processor.tokenizer.save_pretrained.assert_called_once_with(tmp_path)


@pytest.mark.parametrize(
    ("source", "validate", "expected"),
    [("ami_train_ihm", True, 2048), ("ls", True, 2048), ("ami_train_ihm", False, 2048), ("davidai_test", True, 8192)],
)
def test_short_greedy_baseline_keeps_policy_cap_while_full_validation_has_long_budget(
    monkeypatch, source, validate, expected
):
    loop = agent_loop.VibeVoiceAgentLoop.__new__(agent_loop.VibeVoiceAgentLoop)
    loop.config = OmegaConf.create({"data": {"max_audio_dur": 40}, "asr": {"validation_max_audio_seconds": 180}})
    loop.rollout_config = OmegaConf.create(
        {
            "prompt_length": 2048,
            "response_length": 2048,
            "custom": {
                "validation_response_length": 8192,
                "no_repeat_ngram_size": 0,
                "validation_no_repeat_ngram_size": 0,
                "stop": [],
                "stop_token_ids": [],
            },
        }
    )
    loop.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 9)
    loop.processor = Mock(
        speech_start_id=1,
        speech_end_id=3,
        return_value={
            "input_ids": torch.tensor([[10, 1, 2, 2, 3, 198, 11]]),
            "attention_mask": torch.ones(1, 7),
            "acoustic_input_mask": torch.tensor([[False, False, True, True, False, False, False]]),
            "speech_tensors": torch.full((1, 24000), 0.125),
        },
    )
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                token_ids=[12, 13],
                log_probs=[-0.1, -0.2],
                extra_fields={},
            )
        )
    )
    monkeypatch.setattr(agent_loop, "load_waveform", lambda row: np.zeros(24000, dtype=np.float32))

    output = asyncio.run(loop.generate_row({"data_source": source}, {"logprobs": True}, validate=validate))

    assert loop.server_manager.generate.call_args.kwargs["sampling_params"]["max_tokens"] == expected
    assert output.response_ids == [12, 13]
    assert output.response_logprobs == [-0.1, -0.2]
    audio, sample_rate = loop.server_manager.generate.call_args.kwargs["audio_data"][0]
    assert sample_rate == 24000
    np.testing.assert_array_equal(audio, np.full(24000, 0.125, dtype=np.float32))

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_qwen35_audio_hf as hf  # noqa: E402
import run_qwen35_audio_vllm as vllm  # noqa: E402


def test_decoders_share_prompt_defaults(monkeypatch):
    expected_prompt = (
        "<|im_start|>user\n"
        "Detect the language and transcribe the audio clip into text.<audio>"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": hf.DEFAULT_INSTRUCTION}]
            assert add_generation_prompt is True
            assert tokenize is False
            return expected_prompt

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, *, trust_remote_code):
            assert model_path == "/model"
            assert trust_remote_code is False
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert hf.DEFAULT_INSTRUCTION == vllm.DEFAULT_INSTRUCTION
    assert hf.DEFAULT_STOP_TOKEN_IDS == vllm.DEFAULT_STOP_TOKEN_IDS
    assert hf.build_prompt("/model", hf.DEFAULT_INSTRUCTION, "") == expected_prompt
    assert vllm.build_prompt("/model", vllm.DEFAULT_INSTRUCTION, "") == expected_prompt


def test_hf_generation_defaults_match_vllm():
    args = argparse.Namespace(
        max_tokens=512,
        temperature=0.0,
        stop_token_id=None,
    )
    tokenizer = SimpleNamespace(eos_token_id=248044)

    assert hf.build_generation_kwargs(args, tokenizer) == {
        "max_new_tokens": 512,
        "do_sample": False,
        "eos_token_id": [248044, 248046],
        "pad_token_id": 248044,
        "repetition_penalty": 1.0,
    }


def test_hf_sampling_and_stop_overrides():
    args = argparse.Namespace(
        max_tokens=64,
        temperature=0.7,
        stop_token_id=[7, 8],
    )
    tokenizer = SimpleNamespace(eos_token_id=9)

    assert hf.build_generation_kwargs(args, tokenizer) == {
        "max_new_tokens": 64,
        "do_sample": True,
        "eos_token_id": [7, 8],
        "pad_token_id": 9,
        "repetition_penalty": 1.0,
        "temperature": 0.7,
    }


def test_hf_rejects_negative_temperature():
    args = argparse.Namespace(
        max_tokens=64,
        temperature=-0.1,
        stop_token_id=None,
    )

    with pytest.raises(ValueError, match="non-negative"):
        hf.build_generation_kwargs(args, SimpleNamespace(eos_token_id=248044))


def test_hf_model_uses_native_qwen35_backbone_when_available():
    native_module = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5",
        exc_type=ImportError,
    )
    plugin_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(plugin_src))

    from hf_qwen35_audio.modeling_qwen3_5_audio import Qwen3_5AudioForCausalLM

    assert issubclass(Qwen3_5AudioForCausalLM, native_module.Qwen3_5ForCausalLM)

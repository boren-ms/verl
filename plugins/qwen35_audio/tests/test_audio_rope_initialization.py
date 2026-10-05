import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5", exc_type=ImportError)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hf_qwen35_audio import Qwen3_5AudioConfig, Qwen3_5AudioForCausalLM


def make_model():
    return Qwen3_5AudioForCausalLM(
        Qwen3_5AudioConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
            audio_processor={
                "name": "flash",
                "config": {
                    "input_size": 8,
                    "hidden_size": 16,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "intermediate_size": 32,
                    "rope_theta": 10000.0,
                    "frontend_type": "temporal_conv",
                    "frontend_window_size": 2,
                },
            },
        )
    )


def expected_frequencies(rotary):
    return 1.0 / (rotary.base ** (torch.arange(0, rotary.dim, 2).float() / rotary.dim))


def test_audio_rope_reinitialized_after_meta_materialization():
    model = make_model()
    rotary = model.model.embed_tokens_extend.encoder.rotary_emb
    rotary.to("meta").to_empty(device="cpu")
    rotary.inv_freq.fill_(float("nan"))
    model.initialize_weights()
    torch.testing.assert_close(rotary.inv_freq, expected_frequencies(rotary))
    assert "inv_freq" not in rotary.state_dict()


def test_audio_rope_restored_by_from_pretrained(tmp_path, monkeypatch):
    monkeypatch.setattr("hf_qwen35_audio.flash_encoder._flash_attn_available", False)
    model = make_model()
    model.save_pretrained(tmp_path)
    restored = Qwen3_5AudioForCausalLM.from_pretrained(tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    rotary = restored.model.embed_tokens_extend.encoder.rotary_emb
    torch.testing.assert_close(rotary.inv_freq, expected_frequencies(rotary))
    encoder = restored.model.embed_tokens_extend.encoder.eval()
    with torch.no_grad():
        encoded, _ = encoder(torch.randn(1, 16, 8), torch.ones(1, 1, 16, dtype=torch.bool))
    assert torch.isfinite(encoded).all()

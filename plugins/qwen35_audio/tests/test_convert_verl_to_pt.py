"""Standalone checkpoint layout and precision checks, without model runtime imports."""

import importlib.util
import sys
from pathlib import Path

import blobfile as bf
import pytest
import torch

# The converter is a standalone CLI; its tests should not import the audio model
# package, whose optional runtime dependencies are unrelated to checkpoint math.
_path = Path(__file__).parents[1] / "src/hf_qwen35_audio/convert_verl_to_pt.py"
_spec = importlib.util.spec_from_file_location("qwen_audio_checkpoint_converter", _path)
converter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(converter)
_resolve_actor_dir = converter._resolve_actor_dir


@pytest.mark.parametrize("nested_actor", [False, True])
def test_remote_sft_and_rl_checkpoint_layouts(monkeypatch, nested_actor):
    root = "az://account/container/global_step_20"
    actor = root + "/actor"
    monkeypatch.setattr(bf, "isdir", lambda path: path == actor and nested_actor)

    def listdir(path):
        if not nested_actor:
            raise FileNotFoundError(path)
        assert path == actor
        return ["model_world_size_16_rank_0.pt", "optim_world_size_16_rank_0.pt"]

    monkeypatch.setattr(bf, "listdir", listdir)
    assert _resolve_actor_dir(root + "/") == (actor if nested_actor else root)


def test_remote_unrelated_actor_directory_does_not_hide_sft_shards(monkeypatch):
    root = "az://account/container/global_step_20"
    monkeypatch.setattr(bf, "isdir", lambda path: True)
    monkeypatch.setattr(bf, "listdir", lambda path: ["effective_config.yaml"])
    assert _resolve_actor_dir(root) == root


@pytest.fixture
def tiny_checkpoint(tmp_path):
    # BF16 rounds 1.003 to 1.0. Rounding A/B before their product changes the
    # final BF16 result, so this catches accidental early adapter quantization.
    a = torch.eye(2, dtype=torch.float32) * 1.003
    b = a.clone()
    base = torch.zeros((2, 2), dtype=torch.float32)
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    state = {
        prefix + ".lora_A.default.weight": a,
        prefix + ".lora_B.default.weight": b,
        prefix + ".base_layer.weight": base,
        "base_model.model.unadapted.weight": torch.full((2, 2), 1.003),
    }
    for rank in range(2):
        torch.save(
            {key: value.chunk(2, dim=0)[rank].clone() for key, value in state.items()},
            tmp_path / f"model_world_size_2_rank_{rank}.pt",
        )
    return tmp_path, prefix, a, b, base


@pytest.mark.parametrize("preserve", [False, True])
def test_source_adapter_precision_survives_shard_reconstruction_only_when_requested(tiny_checkpoint, preserve):
    directory, prefix, a, _, _ = tiny_checkpoint
    paths = [str(directory / f"model_world_size_2_rank_{rank}.pt") for rank in range(2)]
    merged = converter._load_and_merge(paths, workers=1, preserve_lora_precision=preserve)
    adapter = merged[prefix + ".lora_A.default.weight"]
    assert adapter.shape == a.shape
    assert adapter.dtype == (torch.float32 if preserve else torch.bfloat16)
    torch.testing.assert_close(adapter, a if preserve else a.bfloat16(), rtol=0, atol=0)
    assert merged[prefix + ".base_layer.weight"].dtype == torch.bfloat16
    assert merged["base_model.model.unadapted.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("preserve", [False, True])
def test_merged_cli_preserves_source_adapter_math_without_leaving_adapter_keys(tiny_checkpoint, monkeypatch, preserve):
    directory, _, a, b, base = tiny_checkpoint
    output = directory / "merged.pt"
    argv = [
        "convert_verl_to_pt.py",
        "--input",
        str(directory),
        "--output",
        str(output),
        "--match-lora-merged",
        "--lora-alpha",
        "4",
        "--lora-rank",
        "2",
        "--workers",
        "1",
    ]
    if preserve:
        argv.append("--preserve-lora-precision")
    monkeypatch.setattr(sys, "argv", argv)
    converter.main()
    result = torch.load(output, map_location="cpu", weights_only=False)
    full_precision = (base + (b @ a) * 2).bfloat16()
    historical = (base.bfloat16().float() + (b.bfloat16().float() @ a.bfloat16().float()) * 2).bfloat16()
    assert not torch.equal(full_precision, historical)
    torch.testing.assert_close(
        result["model.layers.0.self_attn.q_proj.base_layer.weight"],
        full_precision if preserve else historical,
        rtol=0,
        atol=0,
    )
    assert not any("lora_A" in key or "lora_B" in key for key in result)
    assert all(value.dtype == torch.bfloat16 for value in result.values())


def test_preserve_precision_requires_merged_mode_before_reading_input(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["convert_verl_to_pt.py", "--input", "/missing", "--output", "/unused", "--preserve-lora-precision"],
    )
    with pytest.raises(SystemExit) as error:
        converter.main()
    assert error.value.code == 2
    assert "--preserve-lora-precision requires --match-lora-merged" in capsys.readouterr().err

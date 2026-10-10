import hashlib
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[2] / ".github/skills/lora-weight-transfer/scripts/merge_lora_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("merge_lora_checkpoint", SCRIPT)
merger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merger)


def adapter(prefix="model.layers.0.mlp.up_proj"):
    return {
        f"base_model.model.{prefix}.lora_A.speech.weight": torch.tensor([[1.0, 2.0]]),
        f"base_model.model.{prefix}.lora_B.speech.weight": torch.tensor([[3.0], [4.0]]),
    }


def baseline():
    return OrderedDict([
        ("model.layers.0.mlp.up_proj.base_layer.weight", torch.ones(2, 2, dtype=torch.bfloat16)),
        ("model.norm.weight", torch.ones(2)),
    ])


def test_preserves_full_baseline_mtp_and_unrelated_tensors():
    state = baseline()
    state["mtp.layers.0.self_attn.q_proj.weight"] = torch.ones(2, 2)
    merged, updated = merger.merge_lora(state, adapter(), 2)
    assert list(merged) == [key.replace(".base_layer.", ".") for key in state]
    assert updated == ["model.layers.0.mlp.up_proj.weight"]
    expected = torch.tensor([[7, 13], [9, 17]], dtype=torch.bfloat16)
    assert torch.equal(merged[updated[0]], expected)
    for key in ("model.norm.weight", "mtp.layers.0.self_attn.q_proj.weight"):
        assert merged[key] is state[key]
    assert torch.equal(state["model.layers.0.mlp.up_proj.base_layer.weight"], torch.ones(2, 2))


def test_restores_only_mtp_from_full_training_checkpoint(tmp_path):
    full = baseline()
    full.update(adapter())
    full["mtp.layers.0.self_attn.q_proj.weight"] = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    path = tmp_path / "full.pt"
    torch.save({"module": full, "optimizer": {"unrelated": 1}}, path)
    loaded, wrapped = merger.load_tensor_dict(path)
    assert wrapped
    merged, updated = merger.merge_lora(baseline(), adapter(), 2, loaded)
    assert len(merged) == 3
    assert len(updated) == 1
    assert not any(merger.LORA_KEY_PATTERN.match(key) for key in merged)
    assert merged["mtp.layers.0.self_attn.q_proj.weight"] is loaded["mtp.layers.0.self_attn.q_proj.weight"]
    output = tmp_path / "merged.pt"
    checksum = merger.write_checkpoint(merged, output)
    recovered, wrapped = merger.load_tensor_dict(output)
    assert wrapped
    assert list(recovered) == list(merged)
    for key in merged:
        assert recovered[key].dtype == merged[key].dtype
        assert torch.equal(recovered[key], merged[key])
    assert checksum == hashlib.md5(output.read_bytes()).hexdigest()
    assert output.with_suffix(".md5").read_text() == f"{checksum}  merged.pt\n"


@pytest.mark.parametrize("source", [
    {},
    {"model.norm.weight": torch.ones(2)},
    {"mtp.layers.0.q_proj.lora_A.default.weight": torch.ones(2, 2)},
])
def test_rejects_missing_or_unmerged_mtp_source(source):
    with pytest.raises(ValueError, match="MTP baseline"):
        merger.merge_lora(baseline(), adapter(), 2, source)


def test_rejects_conflicting_mtp_source():
    state = baseline()
    state["mtp.fc.weight"] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="conflicts"):
        merger.merge_lora(state, adapter(), 2, {"mtp.fc.weight": torch.zeros(2, 2)})


def test_rejects_mtp_adapter_updates():
    state = baseline()
    state["mtp.fc.weight"] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="must not modify"):
        merger.merge_lora(state, adapter("mtp.fc"), 2)


def test_checks_configured_mtp_layers(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mtp_num_hidden_layers": 2}))
    state = {"mtp.layers.0.self_attn.q_proj.weight": torch.ones(2, 2)}
    with pytest.raises(ValueError, match=r"missing configured MTP layers \[1\].*--mtp-baseline"):
        merger.validate_mtp_layers(state, config)
    state["mtp.layers.1.self_attn.q_proj.weight"] = torch.ones(2, 2)
    merger.validate_mtp_layers(state, config)
    config.write_text("{}")
    merger.validate_mtp_layers(baseline(), config)
    merger.validate_mtp_layers(baseline(), tmp_path / "absent.json")


def test_mtp_serialization_excludes_unrelated_flat_storage(tmp_path):
    flat = torch.arange(10000, dtype=torch.float32)
    state = OrderedDict([("mtp.fc.weight", flat[100:104].reshape(2, 2))])
    output = tmp_path / "compact.pt"
    merger.write_checkpoint(state, output)
    recovered, wrapped = merger.load_tensor_dict(output)
    assert wrapped
    tensor = recovered["mtp.fc.weight"]
    assert torch.equal(tensor, state["mtp.fc.weight"])
    assert tensor.dtype == state["mtp.fc.weight"].dtype
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert output.stat().st_size < flat.untyped_storage().nbytes()

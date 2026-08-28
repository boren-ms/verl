import importlib.util
from pathlib import Path

import torch


def _load_converter_module():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py"
    spec = importlib.util.spec_from_file_location("convert_verl_to_pt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_representative_mesh_ranks_selects_one_ddp_replica():
    converter = _load_converter_module()

    ranks = converter._representative_mesh_ranks(
        ("ddp", "fsdp"), torch.tensor([[0, 1], [2, 3], [4, 5]])
    )

    assert ranks == [0, 1]


def test_representative_mesh_ranks_keeps_plain_fsdp_mesh():
    converter = _load_converter_module()

    ranks = converter._representative_mesh_ranks(
        ("fsdp",), torch.tensor([0, 1, 2, 3])
    )

    assert ranks == [0, 1, 2, 3]

import ast
import logging
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
import soundfile as sf
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).parents[3]


@pytest.fixture
def dataset_namespace():
    # Follow the dataset CPU tests' AST loading pattern to avoid model/GPU imports.
    path = PROJECT_ROOT / "recipe/phimm/data/rl_dataset.py"
    module = ast.parse(path.read_text())
    definitions = [
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"
    ]
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + definitions,
        type_ignores=[],
    )
    sources = {
        "short": datasets.Dataset.from_dict({"audio_path": ["bad.wav"]}),
        "long": datasets.Dataset.from_dict({"audio_path": ["good0.wav", "good1.wav", "good2.wav"]}),
    }
    namespace = {
        "Dataset": object,
        "Sequence": Sequence,
        "datasets": datasets,
        "sf": sf,
        "logger": logging.getLogger(__name__),
        "get_num_proc": lambda value: value,
        "create_audio_dataset": lambda **kwargs: sources[kwargs["name"]],
        "ds_conf": [{"name": "short"}, {"name": "long"}],
        "config": SimpleNamespace(data=OmegaConf.create({"num_proc": None})),
        "tokenizer": None,
        "processor": None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _generation_dataset(namespace):
    path = PROJECT_ROOT / "recipe/phimm/main_asr_gen.py"
    module = ast.parse(path.read_text())
    main_task = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main_task")
    call = next(
        node
        for node in ast.walk(main_task)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RLHFDataset"
    )
    return eval(compile(ast.Expression(call), str(path), "eval"), namespace)


def test_generation_preserves_all_examples_from_unequal_sources(dataset_namespace):
    dataset = _generation_dataset(dataset_namespace)

    assert len(dataset) == 4
    assert list(dataset.ds["audio_path"]) == ["bad.wav", "good0.wav", "good1.wav", "good2.wav"]


def test_generation_does_not_substitute_unreadable_audio(dataset_namespace):
    attempted = []

    def load_audio(row, max_dur):
        attempted.append(row["audio_path"])
        if row["audio_path"] == "bad.wav":
            raise FileNotFoundError("bad.wav")
        raise AssertionError("Generation must not substitute another sample")

    dataset_namespace["load_audio"] = load_audio
    dataset = _generation_dataset(dataset_namespace)

    with pytest.raises(RuntimeError, match="Unable to load audio after 1 attempt"):
        dataset[0]
    assert attempted == ["bad.wav"]


def test_training_still_uses_interleaving_by_default(dataset_namespace):
    dataset = dataset_namespace["RLHFDataset"](dataset_namespace["ds_conf"], None, dataset_namespace["config"].data)

    assert dataset.is_training
    assert list(dataset.ds["audio_path"]) == ["bad.wav", "good0.wav"]

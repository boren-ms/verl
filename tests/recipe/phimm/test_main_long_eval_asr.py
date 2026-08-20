import ast
from pathlib import Path

import yaml


def test_build_dataloader_disables_training_interleave():
    module_path = Path(__file__).parents[3] / "recipe/phimm/main_long_eval_asr.py"
    module = ast.parse(module_path.read_text())
    build_dataloader = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "build_dataloader"
    )
    dataset_call = next(
        node
        for node in ast.walk(build_dataloader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RLHFDataset"
    )
    is_train = next(keyword.value for keyword in dataset_call.keywords if keyword.arg == "is_train")

    assert isinstance(is_train, ast.Constant)
    assert is_train.value is False


def test_openasr_long_eval_config_uses_custom_score_function():
    config_path = (
        Path(__file__).parents[3]
        / "recipe/phimm/config/eval/long_eval_AA_2607v1_seg30_openasr.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["custom_reward_function"] == {
        "path": "recipe/phimm/reward/asr_edge.py",
        "name": "openasr_eval",
    }


def test_long_eval_passes_custom_score_function_to_segment_scoring():
    module_path = Path(__file__).parents[3] / "recipe/phimm/main_long_eval_asr.py"
    module = ast.parse(module_path.read_text())
    main_task = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main_task")
    score_call = next(
        node
        for node in ast.walk(main_task)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "score_segments"
    )
    score_fn = next(keyword.value for keyword in score_call.keywords if keyword.arg == "score_fn")

    assert isinstance(score_fn, ast.Name)
    assert score_fn.id == "score_fn"

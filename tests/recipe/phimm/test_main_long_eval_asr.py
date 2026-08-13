import ast
from pathlib import Path


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

import ast
from pathlib import Path


def test_dapo_inherits_ppo_validation_generation_logging():
    repo_root = Path(__file__).parents[3]
    module = ast.parse((repo_root / "recipe/dapo/dapo_ray_trainer.py").read_text())
    trainer_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RayDAPOTrainer"
    )

    assert [base.id for base in trainer_class.bases if isinstance(base, ast.Name)] == ["RayPPOTrainer"]

    overridden_methods = {node.name for node in trainer_class.body if isinstance(node, ast.FunctionDef)}
    assert "_validate" not in overridden_methods
    assert "_maybe_log_val_generations" not in overridden_methods

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[3]


def test_long_eval_uses_trainer_v1_pipeline():
    module = ast.parse((REPO_ROOT / "recipe/phimm/main_long_eval_asr.py").read_text())
    run_eval = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run_eval")
    calls = {
        node.func.id
        for node in ast.walk(run_eval)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    classes = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    assert "run_ppo" in calls
    assert {"LongASREvalTrainer", "LongASREvalTaskRunner"}.issubset(classes)


def test_long_eval_base_uses_trainer_v1_validation():
    config = yaml.safe_load((REPO_ROOT / "recipe/phimm/config/base/long_eval_asr.yaml").read_text())

    assert config["defaults"][0] == "remax_asr"
    assert config["actor_rollout_ref"]["model"]["path"] == "${model.path}"
    assert config["trainer"]["val_only"] is True
    assert config["trainer"]["v1"]["trainer_mode"] == "sync"
    assert config["val_reward"]["group_segment"] is True


def test_long_rollout_dummy_reward_target_exists():
    module = ast.parse((REPO_ROOT / "recipe/phimm/long_asr_rollout.py").read_text())
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    config = yaml.safe_load((REPO_ROOT / "recipe/phimm/config/base/long_rollout_asr.yaml").read_text())

    reward = config["reward"]["custom_reward_function"]
    assert reward == {
        "path": "pkg://recipe.phimm.long_asr_rollout",
        "name": "dummy_score",
    }
    assert reward["name"] in functions


def test_de_fleurs_eval_matches_reference_decode_settings():
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "recipe/phimm/config/eval/eval_openasr_ml_verb_2607v1_de_fleurs.yaml"
        ).read_text()
    )

    assert config["data"]["model_version"] == 2607
    assert config["data"]["max_response_length"] == 512
    assert config["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] == 8
    assert config["rollout"]["n_gpus_per_node"] == 8
    assert config["val_reward"]["reward_function_by_data_source"] == {"de_fleurs": "openasr"}


def test_async_rollout_enables_qwen35_audio_vllm_plugin():
    module = ast.parse((REPO_ROOT / "recipe/phimm/asr_rollout.py").read_text())
    init_ray = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "init_ray"
    )
    string_literals = {
        node.value for node in ast.walk(init_ray) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "VLLM_PLUGINS" in string_literals
    assert "qwen35_audio" in string_literals
    assert "QWEN35_AUDIO_DISABLE_CUDNN" in string_literals


def test_long_rollout_uses_reference_generation_guards():
    config = yaml.safe_load((REPO_ROOT / "recipe/phimm/config/base/long_rollout_asr.yaml").read_text())
    rollout = config["actor_rollout_ref"]["rollout"]

    assert rollout["no_repeat_ngram_size"] == 15
    assert rollout["stop_token_ids"] == [248044, 248046]


def test_async_audio_rollout_preserves_rendered_prompt_prefix():
    config = yaml.safe_load((REPO_ROOT / "recipe/phimm/config/base/long_rollout_asr.yaml").read_text())
    dataset_module = ast.parse((REPO_ROOT / "recipe/phimm/data/rl_dataset.py").read_text())
    agent_loop_module = ast.parse(
        (REPO_ROOT / "verl/experimental/agent_loop/single_turn_agent_loop.py").read_text()
    )

    dataset_strings = {
        node.value
        for node in ast.walk(dataset_module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    agent_loop_strings = {
        node.value
        for node in ast.walk(agent_loop_module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "full_prompt_text" in dataset_strings
    assert "full_prompt_text" in agent_loop_strings

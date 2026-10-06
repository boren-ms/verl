from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).parents[3]
CONFIG_ROOT = ROOT / "recipe/phimm/config"
LS_CONFIG_DIR = CONFIG_ROOT / "v2609_ls"
TRAINER_CONFIG_DIR = ROOT / "verl/trainer/config"
ACTIVE_TRAINING_RECIPE = "remax_2609v0_ls_s400_bs128_n4_r256_g32"
MODEL_PATH = (
    "az://orngwus2cresco/data/speech/projects/phi-fastllm-2609/amlt-results/"
    "fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.3-r2/54000/qwen_hf"
)
TRAINING_SETTINGS = [
    ("rare_s500_bs512_n2_r256_g16", 512, 2, 256, 2, 500, 1.2, 5e-6, False),
    ("rare_s1k_bs256_n1_r256_g16", 256, 1, 256, 2, 1000, 1.2, 5e-6, False),
    ("rare_s1k_bs128_n32_r256_g32_lr2_filter", 128, 32, 256, 4, 1000, 1.2, 5e-8, True),
    ("rare_s1k_bs256_n16_r256_g32_filter_temp08", 256, 16, 256, 4, 1000, 0.8, 5e-6, True),
    ("rare_s1k_bs256_n16_r256_g32_lr2_filter", 256, 16, 256, 4, 1000, 1.2, 5e-6, True),
    ("rare_s400_bs256_n8_r128_g32_filter", 256, 8, 128, 4, 400, 1.0, 7e-6, True),
    ("rare_bad_s1k_bs256_n4_r256_g32", 256, 1, 256, 4, 1000, 1.2, 5e-6, False),
    ("rare_s1k_bs256_n4_r256_g32_filter_temp08", 256, 4, 256, 4, 1000, 0.8, 1e-6, True),
    ("rare_bad_s1k_bs256_n2_r256_g16", 256, 1, 256, 2, 1000, 1.2, 5e-6, False),
]


def compose_ls(config_name):
    config_path = LS_CONFIG_DIR / config_name
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        return compose(
            config_name=config_path.name,
            overrides=[f"hydra.searchpath=[file://{CONFIG_ROOT},file://{TRAINER_CONFIG_DIR}]"],
        )


@pytest.mark.parametrize(
    "config_name",
    sorted(path.relative_to(LS_CONFIG_DIR).with_suffix("").as_posix() for path in LS_CONFIG_DIR.rglob("*.yaml")),
)
def test_ls_configs_compose_without_sibling_recipes(config_name):
    local = OmegaConf.load(LS_CONFIG_DIR / f"{config_name}.yaml")
    recipe_name = Path(config_name).name
    for default in local.defaults:
        assert default == "_self_" or (
            isinstance(default, str)
            and default.split("@")[0]
            in {
                "/ppo_trainer",
                "/generation",
                "base",
                "../base",
                "eval_base",
                "gen_base",
                "/data/train_data/ls_rare_nonempty_verb_langhint",
                "/data/val_data/ls_kw_verb_langhint",
            }
        )
    if recipe_name.startswith(("gen_2609", "remax")):
        assert "/data/train_data/ls_rare_nonempty_verb_langhint" in local.defaults
    if config_name == "base":
        assert "/data/val_data/ls_kw_verb_langhint" in local.defaults
    config = compose_ls(config_name)
    assert config.data.version == 2609
    generation = recipe_name.startswith("gen")
    model = config.model if generation else config.actor_rollout_ref.model
    assert model.path == MODEL_PATH
    if generation:
        assert config.custom_reward_function.reward_kwargs.version == 2609
    else:
        assert config.trainer.project_name == (
            "v2609_ls_eval" if recipe_name in {"eval_base", "eval_2609v0"} else "v2609_ls"
        )
        assert config.val_reward.reward_kwargs.version == 2609
        assert all(reward.reward_kwargs.version == 2609 for reward in config.val_reward.reward_functions.values())
        assert config.actor_rollout_ref.actor.filter_nonfinite_log_probs
        assert config.actor_rollout_ref.ref.filter_nonfinite_log_probs
        assert config.trainer.filter_nonfinite_log_probs
        assert config.data.val_batch_size == 256
        assert {row.post_process.add_field.fields.data_source for row in config.data.val_data} == {
            "ls_clean",
            "ls_other",
        }
        assert all(row.add_task_info.lang_hint for row in config.data.val_data)
        assert OmegaConf.to_container(config.val_reward.reward_function_by_data_source) == {
            "ls_clean": "openasr_en",
            "ls_other": "openasr_en",
        }
    if recipe_name.startswith(("gen_2609", "remax")):
        assert config.data.train_data.add_task_info.lang_hint
        assert config.data.train_data.post_process.add_field.fields.data_source == "ls_rare_verb"


@pytest.mark.parametrize("suffix,batch_size,n,rank,nnodes,steps,temperature,lr,filtered", TRAINING_SETTINGS)
def test_ls_training_preserves_experiment_settings(
    suffix, batch_size, n, rank, nnodes, steps, temperature, lr, filtered
):
    config = compose_ls(f"bakup/remax_2609v0_ls_{suffix}")
    assert config.data.train_batch_size == batch_size
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == batch_size
    assert config.actor_rollout_ref.rollout.n == n
    assert not config.actor_rollout_ref.rollout.gt_rollout
    assert config.actor_rollout_ref.rollout.temperature == temperature
    assert config.actor_rollout_ref.model.lora_rank == rank
    assert config.actor_rollout_ref.actor.optim.lr == lr
    assert config.trainer.nnodes == nnodes
    assert config.trainer.total_training_steps == steps
    assert config.trainer.total_epochs == 30
    assert config.trainer.test_freq == 50
    assert config.trainer.save_freq == 50
    assert config.algorithm.filter_groups.enable == filtered
    assert config.reward_functions.asr_measure.reward_kwargs.version == 2609
    assert config.reward_function_by_data_source.ls_rare_verb == "asr_measure"
    if filtered:
        assert config.algorithm.filter_groups.metric == "remax_advantage"
        assert config.algorithm.filter_groups.mode == "nonzero"
        assert config.algorithm.filter_groups.max_num_gen_batches == 10
    if "rare_bad" in suffix:
        assert config.data.train_data.pre_process is None
        assert config.data.train_data.jsonl_paths == (
            "az://orngwus2cresco/data/boren/data/verl/gen_qwen/2607v1a_earning_step650/"
            "ls_rare_nonempty_verb_g32_nonzero_n_err.jsonl"
        )
        assert config.actor_rollout_ref.rollout.enforce_eager
    else:
        assert config.data.train_data.pre_process.rename_fields.mappings.text == "transcription"


@pytest.mark.parametrize(
    "suffix,nnodes,output_name",
    [
        ("", 1, "ls_rare_nonempty_verb"),
        ("_10k_g16", 2, "ls_rare_nonempty_verb_10k_seed42"),
        ("_g32", 4, "ls_rare_nonempty_verb_g32"),
    ],
)
def test_ls_generation_uses_2609_outputs(suffix, nnodes, output_name):
    config = compose_ls(f"gen/gen_2609v0_ls_rare_nonempty{suffix}")
    assert config.trainer.nnodes == nnodes
    assert config.data.output_path == f"az://orngwus2cresco/data/boren/data/verl/gen_qwen/2609v0/{output_name}"
    assert config.data.resume_from_output
    assert config.data.batch_size == 512
    assert config.data.output_split_size == 20000
    assert config.rollout.n == 1
    assert config.rollout.temperature == 0.0
    assert config.rollout.engine_kwargs.vllm.quantization == "fp8"
    if suffix == "_10k_g16":
        assert config.data.train_data.jsonl_paths.endswith("_sample10k_seed42.jsonl")
        assert config.rollout.engine_kwargs.vllm.compilation_config == 0
    if suffix == "_g32":
        assert config.rollout.enforce_eager


def test_ls_eval_reuses_validation_data_and_resumes():
    config = compose_ls("eval/eval_2609v0")
    assert config.trainer.val_only
    assert config.trainer.val_before_train
    assert config.trainer.validation_resume
    assert config.trainer.validation_resume_save_freq == 2
    assert OmegaConf.to_container(config.data.train_data) == OmegaConf.to_container(config.data.val_data)
    assert not config.data.shuffle
    assert not config.data.use_interleave


def test_ls_local_defaults_match_2609_reference():
    reference = OmegaConf.load(CONFIG_ROOT / "v2609_asr/base.yaml")
    local = OmegaConf.load(LS_CONFIG_DIR / "base.yaml")
    assert local.actor_rollout_ref.model == reference.actor_rollout_ref.model
    assert local.actor_rollout_ref.rollout.engine_kwargs == reference.actor_rollout_ref.rollout.engine_kwargs


def test_ls_datasets_use_original_shared_data_folders():
    config = compose_ls(ACTIVE_TRAINING_RECIPE)
    assert not (LS_CONFIG_DIR / "train_data.yaml").exists()
    assert not (LS_CONFIG_DIR / "val_data.yaml").exists()
    assert config.data.train_data == OmegaConf.load(
        CONFIG_ROOT / "data/train_data/ls_rare_nonempty_verb_langhint.yaml"
    )
    assert config.data.val_data == OmegaConf.load(
        CONFIG_ROOT / "data/val_data/ls_kw_verb_langhint.yaml"
    )


def test_ls_training_recipes_are_archived():
    assert {path.stem for path in LS_CONFIG_DIR.glob("remax_*.yaml")} == {ACTIVE_TRAINING_RECIPE}
    assert {path.stem for path in (LS_CONFIG_DIR / "bakup").glob("remax_*.yaml")} == {
        f"remax_2609v0_ls_{settings[0]}" for settings in TRAINING_SETTINGS
    }


def test_ls_active_training_settings_and_rewards():
    config = compose_ls(ACTIVE_TRAINING_RECIPE)
    assert config.data.train_batch_size == 128
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert config.actor_rollout_ref.rollout.n == 4
    assert not config.actor_rollout_ref.rollout.gt_rollout
    assert config.actor_rollout_ref.rollout.temperature == 1.2
    assert config.actor_rollout_ref.model.lora_rank == 256
    assert config.actor_rollout_ref.actor.optim.lr == 5e-6
    assert config.trainer.nnodes == 4
    assert config.trainer.n_gpus_per_node * config.trainer.nnodes == 32
    assert config.trainer.total_training_steps == 400
    assert config.trainer.total_epochs == 30
    assert config.trainer.test_freq == 50
    assert config.trainer.save_freq == 50
    assert not config.algorithm.filter_groups.enable
    assert config.reward_function_by_data_source.ls_rare_verb == "asr_measure"
    rewards = config.reward_functions.asr_measure.reward_kwargs
    assert rewards.version == 2609
    assert OmegaConf.to_container(rewards.reduce) == {"mode": "average", "total": 4}
    assert OmegaConf.to_container(rewards.measures) == {
        "char": {"beta": 1.0},
        "word": {"beta": 1.0},
        "keyword": {"beta": 1.0},
        "lang": {"beta": 0.5},
        "fmt": {"beta": 0.5},
    }


def test_ls_generation_and_evaluation_configs_use_dedicated_folders():
    assert not list(LS_CONFIG_DIR.glob("gen_*.yaml"))
    assert not list(LS_CONFIG_DIR.glob("eval_*.yaml"))
    assert not (LS_CONFIG_DIR / "base_eval.yaml").exists()
    assert {path.name for path in (LS_CONFIG_DIR / "gen").glob("*.yaml")} == {
        "gen_base.yaml",
        "gen_2609v0_ls_rare_nonempty.yaml",
        "gen_2609v0_ls_rare_nonempty_10k_g16.yaml",
        "gen_2609v0_ls_rare_nonempty_g32.yaml",
    }
    assert {path.name for path in (LS_CONFIG_DIR / "eval").glob("*.yaml")} == {
        "eval_base.yaml",
        "eval_2609v0.yaml",
    }

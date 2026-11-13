"""Translate only the selected legacy recipe's settings to main's V1 schema."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict


def copy_existing(source, target, prefix="", omitted=None):
    """Keep main's dataclass targets while preserving supported source settings."""
    omitted = omitted if omitted is not None else []
    for key, value in source.items():
        if key == "_target_":
            continue
        path = f"{prefix}.{key}" if prefix else key
        if key not in target.keys():
            omitted.append(path)
        elif isinstance(value, DictConfig) and isinstance(target[key], DictConfig):
            copy_existing(value, target[key], path, omitted)
        else:
            target[key] = value
    return omitted


def adapt_config(source):
    """Use current trainer defaults; import ASR settings, never the legacy trainer."""
    # A nested compose otherwise inherits the recipe's primary config directory.
    # Keep the outer Hydra context intact for the caller.
    state = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        root = Path(__file__).resolve().parents[6]
        with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
            current = compose(config_name="ppo_trainer")
    finally:
        GlobalHydra.instance().hydra = state
    # Resolve first: source interpolation paths are meaningful in the source config.
    source = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    omitted = []
    for key in ("actor_rollout_ref", "critic", "data", "trainer", "algorithm", "ray_kwargs", "global_profiler"):
        omitted.extend(copy_existing(source[key], current[key], key))
    with open_dict(current):
        # Preserve source-only data and reward declarations as recipe-owned settings.
        for key in ("reward_functions", "reward_function_by_data_source", "val_reward"):
            current[key] = source.get(key, {})
        for key, value in source.data.items():
            if key not in current.data:
                current.data[key] = value
        current.data.train_files = "__asr_train__"
        current.data.val_files = "__asr_val__"
        current.data.validation_shuffle = False
        current.data.custom_cls.path = str(Path(__file__).with_name("dataset.py"))
        current.data.custom_cls.name = "ASRDataset"
        current.actor_rollout_ref.model.external_lib = "hf_qwen35_audio"
        rollout = current.actor_rollout_ref.rollout
        rollout.engine_kwargs = source.actor_rollout_ref.rollout.engine_kwargs
        rollout.mode = "async"
        rollout.agent.default_agent_loop = "asr"
        rollout.agent.agent_loop_config_path = str(Path(__file__).parents[1] / "config/agentic/agents.yaml")
        rollout.agent.agent_loop_manager_class = "recipe.phimm.agentic.agent_loop.ASRManager"
        old_rollout = source.actor_rollout_ref.rollout
        validation_ngram_size = old_rollout.val_kwargs.get("no_repeat_ngram_size", 0)
        rollout.custom = {
            "stop": old_rollout.get("stop", []),
            "stop_token_ids": old_rollout.get("stop_token_ids", []),
            "no_repeat_ngram_size": old_rollout.get("no_repeat_ngram_size", 0),
            "no_repeat_ngram_window_size": old_rollout.get("no_repeat_ngram_window_size", 512),
            # The source mutates shared SamplingParams for validation. A zero
            # validation size omits extra_args and retains the training block.
            # DAPO also uses this validation path for its greedy ReMax baseline.
            "validation_no_repeat_ngram_size": validation_ngram_size
            if validation_ngram_size > 0
            else old_rollout.get("no_repeat_ngram_size", 0),
            "validation_no_repeat_ngram_window_size": old_rollout.val_kwargs.get("no_repeat_ngram_window_size", 512)
            if validation_ngram_size > 0
            else old_rollout.get("no_repeat_ngram_window_size", 512),
            "validation_response_length": old_rollout.val_kwargs.get("response_length", None)
            or old_rollout.response_length,
        }
        rollout.engine_kwargs.vllm.logits_processors = [
            "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"
        ]
        # Source vllm_rollout_spmd defaults every replica to seed 0, disables
        # the multimodal cache, and bypasses compile for its H100 LoRA kernels.
        # The imported recipe composes against main's rollout defaults, which
        # introduce seed=42 and processed_logprobs. Neither setting existed in
        # the source YAML: its LLM constructor used seed=0 and vLLM's raw
        # logprobs (temperature_scaled_log_probs=False in Base1).
        rollout.seed = old_rollout.engine_kwargs.vllm.get("seed", 0)
        rollout.engine_kwargs.vllm.seed = rollout.seed
        rollout.logprobs_mode = "raw_logprobs"
        rollout.engine_kwargs.vllm.mm_processor_cache_gb = 0
        if source.actor_rollout_ref.model.lora_rank > 0:
            rollout.engine_kwargs.vllm.compilation_config = {"mode": 0}
        current.trainer.v1.trainer_class = "recipe.phimm.agentic.trainer.ASRTrainerSync"
        current.trainer.v1.trainer_mode = "sync"
        current.trainer.checkpoint_callback_class = "recipe.phimm.agentic.trainer.ASRCheckpointCallback"
        # The source actor uses the initial actor logprobs on its single on-policy
        # minibatch, with min(exp(actor-rollout), 5) token importance weights.
        current.algorithm.rollout_correction.bypass_mode = False
        current.algorithm.rollout_correction.rollout_is = "token"
        current.algorithm.rollout_correction.rollout_is_threshold = source.actor_rollout_ref.actor.tis_imp_ratio_cap
        current.algorithm.rollout_correction.rollout_is_batch_normalize = False
        current.actor_rollout_ref.actor.fsdp_config.use_torch_compile = source.actor_rollout_ref.actor.use_torch_compile
        current.actor_rollout_ref.ref.fsdp_config.use_torch_compile = source.actor_rollout_ref.actor.use_torch_compile
        current.asr = {"source_only_keys": omitted, "validation_remote_dir": source.trainer.validation_data_dir}
        if source.data.get("mix") and not source.trainer.val_only:
            rows = sum(item.sample for group in source.data.mix for item in group.datasource)
            current.asr.expected_training_rows = rows
            current.asr.expected_training_steps = rows // source.data.train_batch_size * source.trainer.total_epochs
        current.trainer.validation_data_dir = current.trainer.default_local_dir + "/val_data_gen"
    return current

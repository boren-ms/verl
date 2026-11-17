"""Run Qwen audio SFT on the existing Ray cluster with native verl workers."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf


def cache_model_paths(paths):
    """Populate each node's local model cache before serializing HFModelConfig."""
    from verl.utils.fs import copy_to_local

    return {path: copy_to_local(path) for path in dict.fromkeys(paths) if path}


def prepare_model_caches(config):
    """The native SFT driver passes local paths to workers on every GPU node."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if len(nodes) != config.trainer.nnodes:
        raise ValueError(f"SFT requires {config.trainer.nnodes} GPU nodes; found {len(nodes)}")
    paths = [config.model.path, config.model.hf_config_path, config.model.tokenizer_path]
    cache = ray.remote(num_cpus=1)(cache_model_paths)
    results = ray.get(
        [
            cache.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(paths)
            for node in nodes
        ]
    )
    if any(result != results[0] for result in results[1:]):
        raise ValueError("SFT workers require consistent cache paths across GPU nodes")
    print(f"ASR SFT model cache ready on {len(nodes)} GPU nodes", flush=True)


@hydra.main(config_path="../config/v2611_dev/sft", config_name="sft_base3a_2611r2", version_base=None)
def main(config):
    """Keep the driver on CPU and place the configured SFT workers on GPUs."""
    import ray
    from recipe.phimm.sft.trainer_ray import ASRSFTTrainerRay

    from verl.utils.device import auto_set_device

    auto_set_device(config)
    destination = Path(config.trainer.default_local_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, destination / "effective_config.yaml", resolve=True)
    ray.init(address="auto")
    try:
        prepare_model_caches(config)
        trainer = ASRSFTTrainerRay(config)
        # These tensors are driver-side metric aggregates, not model inputs.
        trainer.device_name = "cpu"
        print(f"ASR SFT target: {trainer.total_training_steps} optimizer updates", flush=True)
        trainer.fit()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

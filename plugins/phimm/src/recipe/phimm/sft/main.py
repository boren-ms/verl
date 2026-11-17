"""Launch Qwen audio SFT with torchrun and native verl SFT configuration."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../config/v2611_dev/sft", config_name="sft_base3a_2611r2", version_base=None)
def main(config):
    """Initialize distributed SFT without rollout, reward, critic, or Ray workers."""
    from recipe.phimm.sft.trainer import ASRSFTTrainer
    from torch.distributed import destroy_process_group, get_rank

    from verl.utils.device import auto_set_device
    from verl.utils.distributed import initialize_global_process_group

    auto_set_device(config)
    initialize_global_process_group()
    try:
        if get_rank() == 0:
            destination = Path(config.trainer.default_local_dir).expanduser()
            destination.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, destination / "effective_config.yaml", resolve=True)
        ASRSFTTrainer(config).fit()
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()

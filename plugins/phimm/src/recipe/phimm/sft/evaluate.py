"""Decode an ordered sequence of native SFT checkpoints on a separate Ray pool."""

import hydra
from recipe.phimm.main_asr_remax import main as run_recipe
from recipe.phimm.sft.validation_state import checkpoint_steps


@hydra.main(config_path="../config/v2611_dev/sft", config_name="eval_sft_checkpoints_base3a_2611r2", version_base=None)
def main(config):
    checkpoint_steps(config.asr.sft_validation.steps)
    if not config.trainer.val_only or config.trainer.resume_mode != "disable":
        raise ValueError("SFT checkpoint decoding requires val_only and disabled training resume")
    if list(config.actor_rollout_ref.actor.checkpoint.load_contents) != ["model"]:
        raise ValueError("SFT checkpoint decoding must load model weights only")
    if not config.actor_rollout_ref.model.lora.get("merge", False):
        raise ValueError("SFT checkpoint decoding requires merged rollout weights")
    run_recipe.__wrapped__(config)


if __name__ == "__main__":
    main()

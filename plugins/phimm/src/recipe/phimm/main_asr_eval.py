"""Full OpenASR evaluation using the same agent loop and scoring adapter."""

import hydra
from recipe.phimm.main_asr_remax import main as run_recipe


@hydra.main(config_path="config/v2611_dev/eval", config_name="eval_openfull_zip", version_base=None)
def main(source):
    trainer = source.trainer
    if not trainer.val_only or not trainer.val_before_train or source.actor_rollout_ref.model.lora_rank != 0:
        raise ValueError("Evaluation requires val_only, val_before_train and a merged model (lora_rank=0)")
    run_recipe.__wrapped__(source)


if __name__ == "__main__":
    main()

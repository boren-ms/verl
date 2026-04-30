"""Eval entry point — identical to main_asr_dapo but with config_path pointing to config/eval."""

import hydra

from recipe.phimm.main_asr_dapo import run_ppo


@hydra.main(config_path="config/eval", config_name="eval_asr", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()

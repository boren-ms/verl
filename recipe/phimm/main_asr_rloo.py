"""RLOO entry point — identical to main_asr_dapo but with config_path pointing to config/rloo."""

import hydra

from recipe.phimm.main_asr_dapo import run_ppo


@hydra.main(config_path="config/rloo", config_name="rloo", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()

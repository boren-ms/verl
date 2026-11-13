"""Selected Qwen ASR recipe on main's agentic V1 PPO/ReMax training stack."""

import os
import sysconfig
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict

from recipe.phimm.utils.env import EnvMgr

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True)


@hydra.main(config_path="config/v2611_dev/main", config_name="remax_main_base3a_2611r2", version_base=None)
def main(config):
    if not config.actor_rollout_ref.model.path:
        raise ValueError("Set actor_rollout_ref.model.path to a verified HF checkpoint")
    if not config.trainer.experiment_name or config.trainer.experiment_name == "remax_main_base3a_2611r2":
        raise ValueError("Set trainer.experiment_name to a unique reproduction name before launching")
    import torch

    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

    runtime_env = config.ray_kwargs.ray_init.runtime_env
    with open_dict(runtime_env):
        runtime_env.env_vars = dict(runtime_env.get("env_vars", {}))
    env = runtime_env.env_vars
    with open_dict(env):
        env.update(EnvMgr().envs())
        # vLLM 0.24 may otherwise discover TileLang's incomplete cudart stub.
        env.VLLM_CUDART_SO_PATH = os.environ.get("VLLM_CUDART_SO_PATH") or str(
            next(
                (Path(sysconfig.get_path("purelib")) / "nvidia").glob(
                    f"*/lib/libcudart.so.{torch.version.cuda.split('.')[0]}"
                )
            )
        )
        for key in ("DATA_PATH", "USER_HOME_PATH", "USER_DATA_PATH", "USER_OUTPUT_PATH"):
            if key in os.environ:
                env[key] = os.environ[key]
    destination = Path(config.trainer.default_local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, destination / "effective_config.yaml", resolve=True)
    run_ppo(config, task_runner_class=TaskRunnerV1)


if __name__ == "__main__":
    main()

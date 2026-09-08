from omegaconf import OmegaConf


def reward_manager_kwargs(config, reward_section):
    kwargs = config.get(reward_section, {}).get("reward_kwargs", {})
    if not kwargs:
        return {}
    if OmegaConf.is_config(kwargs):
        return OmegaConf.to_container(kwargs, resolve=True)
    return dict(kwargs)

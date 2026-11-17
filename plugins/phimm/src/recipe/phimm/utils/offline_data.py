"""Check whether an experiment still requires offline audio preparation."""


def active_audio_transforms(config):
    plan = config.get("offline_data") or {}
    return {
        name: dict(plan[name])
        for name in ("random_segment", "random_merge")
        if plan.get(name) is not None and plan[name].get("enabled", True)
    }


def require_prepared_audio(config):
    if not config.trainer.get("val_only", False) and active_audio_transforms(config):
        raise ValueError(
            "This preset requires offline audio preparation. Run python -m recipe.phimm.tools.offline_audio "
            "with the same config and overrides, then launch its generated training.yaml."
        )

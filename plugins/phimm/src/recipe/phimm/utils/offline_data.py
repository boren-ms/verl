"""Read transform plans for the offline audio preparation tool."""


def active_audio_transforms(config):
    plan = config.get("offline_data") or {}
    return {
        name: dict(plan[name])
        for name in ("random_segment", "random_merge", "svad_segment")
        if plan.get(name) is not None and plan[name].get("enabled", True)
    }

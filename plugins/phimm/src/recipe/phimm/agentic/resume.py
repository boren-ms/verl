"""Checkpoint resume helpers that do not require the GPU training stack."""


def restore_dataloader(dataloader, state, global_steps):
    """Restore a saved loader; at an epoch boundary, consume its exhausted pass.

    The state saved after an epoch's last batch yields no batches on its next
    pass. Draining that empty pass keeps the sampler's generator state, so the
    next epoch matches an uninterrupted run instead of repeating epoch one.
    """
    dataloader.load_state_dict(state)
    steps = len(dataloader)
    if not steps or global_steps % steps:
        return False
    remaining = sum(1 for _ in dataloader)
    if remaining:
        raise ValueError(f"Epoch-boundary dataloader state at step {global_steps} still yields {remaining} batches")
    return True

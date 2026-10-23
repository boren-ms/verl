"""Remove greedy-only rollout fields before merging sampled ReMax responses."""


def clear_baseline_outputs(batch, baseline_output):
    # The termination reason belongs to the response just scored. Keeping the
    # greedy reason can conflict with sampled length/stop reasons at union, or
    # silently label sampled responses with the greedy reason when they agree.
    batch.pop(
        batch_keys=list(baseline_output.batch.keys()),
        non_tensor_batch_keys=[
            key for key in ("rollout_finish_reason",) if key in batch.non_tensor_batch
        ],
    )

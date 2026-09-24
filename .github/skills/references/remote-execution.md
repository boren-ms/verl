# Shared Remote Execution Contract

Read this reference before any workspace skill allocates a Brix pool, submits a
remote job, or installs a monitor. Workflow-specific constraints may narrow this
contract (for example a node allowlist or same-training-pool evaluation), not
silently broaden the authorized work.

## Scope and Ownership

- Operate only on the user's requested jobs/checkpoints or the recovery set
  identified by the workload owner. Never adopt, stop, pause, overwrite, or
  clean up unrelated work.
- A status-only request is read-only: do not submit, repair, export, evaluate,
  or install automation merely to answer it.
- Use `remote-development` for connectivity, environment setup, and authorized
  pool creation/resume. Do not resume/create capacity when a permitted free
  Ready pool is already available. Never pause other pools to tidy up after
  allocation.
- Respect explicit node constraints. If a specified pool is busy, queue the work
  or report the conflict; do not silently select a pool outside the allowlist.
- One coordinator owns a work matrix, node assignments, aggregate cache, and
  any recurring monitor. Child jobs return state to that coordinator and do not
  create a second queue, cache writer, or schedule.

## Occupancy and Topology

Discover pools using `brix pools`; inspect only Ready pools eligible for the
workflow. Immediately before allocation collect all three signals:

```bash
brix ssh <POOL> -- 'bash -l -c "ray job list"'
brix ssh <POOL> -- 'bash -l -c "ray status"'
brix ssh <POOL> -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
```

For multi-node pools inspect GPU activity on every participating node, not only
the head. A pool is eligible only when:

1. No Ray submission is `RUNNING` or `PENDING`.
2. All GPUs have utilization at most 5% and memory use at most 5000 MiB, with
   no unexplained processes, reservations, or draining/unhealthy state.
3. Ray reports sufficient healthy, available GPU resources for the job.

A failed/empty/ambiguous query is unknown occupancy, **not a free pool**. Retain
the error and resolve it before submission; never hide failures with
`2>/dev/null || true` or an "empty jobs" fallback. Idle GPU utilization alone
does not prove Ray resources are available.

For a `verl-n<N>-*` pool, verify `N` against healthy Ray nodes. Set
`trainer.nnodes` to the verified participating count, not the number of free
pools and not an inherited training default. For other explicitly allowed pool
names, derive topology from live Ray state. A name/live-count mismatch blocks
allocation. A one-node-only workflow must select a verified one-node pool.
Assign at most one active work row per pool and recheck occupancy before refilling.

## Synchronization and Submission

Verify the remote checkout contains the intended local code before a pool's first
submission. Sync through the established remote-development mechanism (for example
`rcall-brix sync <POOL>`), and sync again after relevant local changes. Reuse a
verified unchanged checkout for subsequent rows; do not skip initial verification
just because this session made no edits.

Preserve dirty user files. Check a sync helper's behavior before use: do not
implicitly commit/push unrelated edits. Environment preparation and direct Ray
submission must retain the runtime environment required by
[quick_run.sh](../../../quick_run.sh).

Use a unique experiment/output identity per work row and attempt. Inspect matching
live jobs and durable manifests before submitting so retries do not duplicate
active or complete work. Do not use config-wide cleanup as a routine submission
step; stop only an exact tracked job when replacement is authorized.

## Completion and Monitoring

Use Ray terminal status plus validated expected outputs to establish completion.
Missing job history, a final-looking metric line, or idle GPUs alone is not
success. During recovery, complete durable artifacts with matching provenance
can establish prior completion even when Ray history has expired.

During an active monitoring task, check about every five minutes, use
non-detached processes, and keep current phase/metrics durable. A one-shot status
check returns after its snapshot.

Create or modify persistent recurring automation **only when the user explicitly
requests a schedule/automation**. A request to monitor or recover is not that
request. Use the available scheduler's actual API and supported intervals; an
emitted `/every` line is not evidence that a schedule exists. If five-minute
recurrence is unsupported, report that limitation rather than claim it was
installed or silently choose another interval.

Before changing an existing automation, list it and identify its stable ID and
workload scope. Maintain one owner for the whole workload (or one standalone
pipeline), update that schedule rather than duplicating it on every poll, and
never delete unrelated schedules. Persist/re-discover replacement job IDs instead
of pinning the schedule forever to one Ray ID. Disable only the owned monitor when
the requested work completes or the user stops it. Report blocked/failed work
explicitly; do not relabel partial completion as success.

---
name: verl-experiment-distributor
description: "Distribute a set of verl experiment YAMLs across eligible Brix nodes, assigning exactly one dedicated subagent to each experiment and delegating every run lifecycle to verl-asr-run. Use when: run multiple experiment YAMLs, spread configs across nodes, resume an experiment batch, check distributed experiment status, or recover a multi-run workload from local logs."
argument-hint: "<yaml-paths-or-glob> [--nodes <node,...>] [launch|resume|status]"
---

# Distribute verl Experiment YAMLs

Coordinate a batch of independent verl experiment YAMLs across Brix nodes.
Create one dedicated subagent for every experiment run and require that
subagent to invoke `verl-asr-run` for submission, monitoring, repair,
resubmission, and completion. The coordinator owns only workload discovery,
placement, durable local state, and aggregate reporting.

Do not submit or monitor experiment jobs directly from the coordinator. Do not
give one subagent multiple experiments. At most one active subagent may own a
run at a time; if an agent is no longer reachable, record the handoff before
creating one replacement agent for that run.

## Operations

- `launch` (default): create or extend a workload, assign all currently
  placeable experiments, and leave the remainder queued.
- `resume`: reconstruct a workload from local state plus live Brix/Ray state,
  reconnect to reachable run agents, and create replacement agents only where
  necessary.
- `status`: reconcile and report state without launching, resubmitting, or
  changing node assignments.

## Inputs

Accept one or more explicit `.yaml` paths, directories, or glob patterns. A
directory expands recursively to YAML files. Resolve every input relative to
the repository root, then sort by normalized repository-relative path for
deterministic queue order. Ignore non-YAML files, reject missing paths, and
deduplicate paths before doing any remote work.

By default, YAMLs must resolve under `recipe/phimm/config/`. Record both:

- `yaml_path`: normalized repository-relative path including `.yaml`.
- `config`: the Hydra config reference expected by `verl-asr-run`, without the
  `.yaml` suffix.

Never infer two runs from the same YAML unless the user explicitly supplies
distinct override sets. Treat each YAML-plus-overrides combination as one
experiment and include a normalized override digest in its run ID.

If `--nodes` is provided, use only those nodes. Otherwise discover eligible
Ready `verl-*` GPU nodes. Never create, resume, pause, or stop a Brix pool
unless the user explicitly requests it.

## Durable Local Layout

Store all coordinator state and run logs under this ignored local directory:

```text
tmp/verl_experiment_distributor/<workload-id>/
├── manifest.json
├── state.json
├── events.jsonl
└── runs/
    └── <run-id>.log
```

Choose `<workload-id>` from a sanitized user label when supplied. Otherwise
use `<UTC-YYYYMMDDTHHMMSSZ>-<8-char-input-digest>`. On `resume` or `status`,
use an explicitly named workload. If none is named, select the most recently
updated workload only when exactly one unambiguous candidate exists.

### `manifest.json`

Write the immutable launch intent before creating subagents:

```json
{
  "schema_version": 1,
  "workload_id": "20260925T030120Z-a1b2c3d4",
  "created_at_utc": "2026-09-25T03:01:20Z",
  "repo_root": "/home/boren/code/verl",
  "requested_nodes": [],
  "runs": [
    {
      "run_id": "remax_example-a1b2c3d4",
      "yaml_path": "recipe/phimm/config/remax/remax_example.yaml",
      "yaml_sha256": "<sha256>",
      "config": "remax/remax_example",
      "overrides": []
    }
  ]
}
```

Generate each run ID from a sanitized YAML stem plus the first eight
characters of a SHA-256 digest over normalized `yaml_path`, YAML content hash,
and normalized overrides. This makes run identity stable across resumes and
prevents same-name configs in different folders from colliding.

Never silently rewrite launch intent. If inputs differ from an existing
manifest, either create a new workload or explicitly add new unique runs while
preserving every old run entry and event.

### `state.json`

Keep one current snapshot with:

```json
{
  "schema_version": 1,
  "workload_id": "20260925T030120Z-a1b2c3d4",
  "updated_at_utc": "2026-09-25T03:06:20Z",
  "runs": {
    "remax_example-a1b2c3d4": {
      "status": "running",
      "attempt": 1,
      "node": "verl-n1-i0",
      "agent_id": "<agent-or-chat-id>",
      "ray_job_id": "<ray-job-id>",
      "progress": "training 10/200 (5%)",
      "last_heartbeat_utc": "2026-09-25T03:06:20Z",
      "last_error": null,
      "next_action": "monitor",
      "log_path": "runs/remax_example-a1b2c3d4.log"
    }
  }
}
```

Allowed run states are `queued`, `assigned`, `launching`, `running`,
`repairing`, `succeeded`, `failed`, `blocked`, and `stale`. The coordinator is
the only writer of `state.json`. Write a complete temporary snapshot in the
same directory and atomically rename it over `state.json`; never leave a
partially written snapshot.

### `events.jsonl`

Append one JSON object for every transition, dispatch, heartbeat, job-ID
replacement, error, and completion. Each event must contain
`timestamp_utc`, `run_id`, `event`, `attempt`, `node`, and `agent_id`, plus
relevant details. This append-only journal is the recovery audit trail; never
truncate it on resume.

### Per-run logs

Create `runs/<run-id>.log` before dispatch. The dedicated run subagent owns
that file and appends timestamped plain-text updates for:

- assignment and the exact YAML/config/overrides;
- invocation of `verl-asr-run`;
- submission and replacement Ray job IDs;
- status, phase, progress, W&B URL, and Ray URL when available;
- failure diagnosis, repairs, validation, and resubmissions;
- terminal outcome and durable output/report paths.

Keep logs concise and useful for recovery. Do not write credentials, tokens,
SAS query strings, environment dumps, or unredacted secret-bearing commands.
The coordinator may append a dispatch or orphaned-agent marker, but must not
rewrite prior log content. A replacement agent continues the same run log and
records the previous agent ID and handoff reason.

## Procedure

### 1. Discover and validate experiments

1. Expand and normalize the requested YAML inputs.
2. Verify every file exists, is readable, and is within the allowed config
   root unless the user explicitly requested another Hydra search path.
3. Compute its content SHA-256 and deterministic run ID.
4. Resolve the config reference that `verl-asr-run` must receive.
5. Reject duplicate run IDs or ambiguous config references.
6. Create the workload directory, immutable manifest, initial `queued` state,
   event journal, and one log file per run before inspecting nodes.

If a YAML changes after the manifest is created, mark an unlaunched run
`stale`. Do not launch it until a new run identity is recorded. Do not alter
the identity of an already submitted run; record the mismatch and preserve its
history.

### 2. Discover capacity safely

Discover every allowed Ready node, then inspect both Ray jobs and GPU usage:

```bash
brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-'
brix ssh <NODE> -- 'bash -l -c "
  python /root/code/verl/ray_job.py list 2>/dev/null || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
"'
```

A node is free only when it has no active Ray submission and every GPU is at
most 5% utilization and at most 5000 MiB allocated. A listed Ray job, unknown
GPU activity, unhealthy node, or incomplete inspection makes the node
ineligible. Never stop, replace, adopt, or overwrite an unrelated job.

Sort eligible nodes by natural node name and assign queued runs in manifest
order. One run owns one node while its delegated pipeline is active. Do not
oversubscribe a node or move a live run merely to improve packing. Leave
unplaced runs `queued` with `next_action: "wait_for_capacity"`.

### 3. Create exactly one subagent per run

For every newly assigned run, create a dedicated general-purpose subagent or
session with a title containing its run ID. Record the returned agent/chat ID
before it performs remote work. Never batch multiple runs into one prompt and
never ask multiple agents to inspect or manage the same run concurrently.

The run-agent prompt must include:

```text
You exclusively own experiment <run-id>.
First invoke the verl-asr-run skill, then use it to launch or resume only:
- YAML: <absolute-yaml-path>
- config: <config>
- overrides: <overrides>
- assigned node: <node>

Manage this one experiment through submission, continuous monitoring,
diagnosis, repair, resubmission, and terminal completion. Do not select another
node, run another experiment, or manage aggregate workload state.

Append every meaningful update to:
<absolute-workload-dir>/runs/<run-id>.log

Report structured updates to the coordinator with:
run_id, timestamp_utc, status, attempt, node, ray_job_id, progress,
last_error, next_action, wandb_url, ray_url, and output_paths.
Continue until succeeded, explicitly blocked, or irrecoverably failed.
```

The coordinator translates each structured response into an event and an
atomic state snapshot. The run agent must let `verl-asr-run` own code sync,
submission, monitoring, autofix, resubmission, and its job-specific persistent
monitor. The coordinator must not duplicate those actions.

### 4. Refill capacity

Whenever a run becomes terminal:

1. Reconcile its final agent response with live Ray state and durable outputs.
2. Mark it `succeeded`, `failed`, or `blocked` and record a terminal event.
3. Recheck the released node using both occupancy signals.
4. If the node is still eligible, assign the next queued run and create its
   dedicated subagent.

Do not reuse a node while cleanup, checkpoint upload, report generation, or
another stage owned by `verl-asr-run` is still active.

### 5. Resume safely

On `resume`:

1. Read `manifest.json`, `state.json`, `events.jsonl`, and all referenced
   per-run logs. Validate schema versions and ensure every manifest run has
   exactly one state row and log path.
2. Recompute YAML hashes. Mark changed unlaunched inputs `stale`; retain the
   original identity of submitted runs.
3. Rediscover allowed nodes, active Ray jobs, GPU occupancy, and durable
   completion. Live state is authoritative for execution; the manifest is
   authoritative for workload membership.
4. Match live jobs by persisted Ray ID plus config/experiment identity. Never
   adopt a live job that is not in the manifest.
5. Reconnect to the recorded run agent/chat when it is reachable. Send it the
   latest reconciled state and ask it to continue the same run.
6. If the prior agent is unavailable, append an `agent_orphaned` event and log
   marker, increment `attempt`, then create exactly one replacement agent for
   that run. Include the prior log, Ray job ID, node, and last known progress
   in its prompt and instruct it to invoke `verl-asr-run` in resume mode.
7. If no matching live job exists, ask the dedicated agent and
   `verl-asr-run` to determine from logs and durable outputs whether the run
   succeeded, failed, or requires a safe resubmission. Never resubmit merely
   because a saved job ID disappeared from Ray history.
8. Refill verified-free nodes from the queued runs after all existing runs
   have been reconciled.

An interrupted coordinator must therefore be recoverable from local files
without relying on conversation memory. Never delete completed run logs or
replace historical agent/job IDs in `events.jsonl`.

### 6. Check status

`status` is read-only except for appending reconciliation observations and
refreshing the snapshot. Do not create agents or submit/resubmit jobs.
Reconcile local state against live nodes and report:

| Run | YAML | Node | Agent | Ray job | Status | Progress | Last heartbeat | Next action |
|---|---|---|---|---|---|---|---|---|

Sort rows in manifest order. Clearly label:

- stale local state versus confirmed live state;
- queued runs waiting for capacity;
- agents that are unreachable or no longer reporting;
- missing Ray jobs whose durable outcome is not yet known;
- terminal output/report paths.

End with aggregate counts for total, queued, active, succeeded, failed,
blocked, and stale runs, plus the absolute workload directory containing the
logs.

## Completion and Failure Rules

- The workload is complete only when every manifest run is terminal and no
  delegated `verl-asr-run` stage remains active.
- `succeeded` requires the job-specific completion checks and durable outputs
  required by `verl-asr-run`, not only a `SUCCEEDED` Ray status.
- `failed` requires an irrecoverable failure or exhausted user-specified retry
  limit. Without such a limit, let `verl-asr-run` diagnose and repair
  recoverable failures.
- `blocked` means a concrete external prerequisite prevents progress; record
  the exact prerequisite and next action.
- A missing agent heartbeat alone is not job failure.
- A missing Ray job alone is not proof that the experiment never ran.
- Never report aggregate success while any run is queued, assigned,
  launching, running, repairing, blocked, stale, or has an unknown durable
  outcome.

## Quality Checks

Before launch or resume, verify:

1. Manifest run IDs, YAML paths, config references, and log paths are unique.
2. Every manifest run has one state entry and one per-run log.
3. Every active run has exactly one assigned node and one active subagent.
4. No node is assigned to more than one active run.
5. Every assigned node passed both Ray and GPU occupancy checks.
6. Every subagent prompt names exactly one run and requires `verl-asr-run`.
7. `state.json` parses after its atomic rewrite and every `events.jsonl` line
   is valid standalone JSON.
8. Terminal success is backed by `verl-asr-run` completion evidence and
   durable output paths.
9. The final status table accounts for every manifest run and links the user
   to the workload directory and per-run logs.

## Examples

Launch every YAML in a directory on any eligible Ready `verl-*` nodes:

```text
/verl-experiment-distributor recipe/phimm/config/remax/experiments/ launch
```

Launch three selected experiments on a restricted node pool:

```text
/verl-experiment-distributor \
  recipe/phimm/config/remax/a.yaml \
  recipe/phimm/config/remax/b.yaml \
  recipe/phimm/config/remax/c.yaml \
  --nodes verl-n1-i0,verl-n1-i2 launch
```

Resume or check a known workload:

```text
/verl-experiment-distributor 20260925T030120Z-a1b2c3d4 resume
/verl-experiment-distributor 20260925T030120Z-a1b2c3d4 status
```

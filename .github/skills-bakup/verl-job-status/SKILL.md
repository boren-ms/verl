---
name: verl-job-status
description: 'Check training/generation job status across all verl-* Brix nodes. Only use when explicitly requested with phrases like: verl job status, check verl nodes, verl update, check all verl jobs, verl training progress. Do NOT trigger on short generic commands like "update" or "status" alone.'
argument-hint: 'Optional node filter like verl-n2-i0, or omit for all verl-* nodes'
---

# verl-* Job Status Check

Report the status of Ray jobs running on all `verl-*` Brix nodes in a single summary table.

## When to Use
- User explicitly mentions "verl" nodes/jobs (e.g. "check verl nodes", "verl job status", "verl update")
- User asks to check job status across verl-* nodes specifically
- User says "check all verl jobs" or "verl training progress"

## When NOT to Use
- Short generic commands like "update", "status", "check progress" without mentioning verl
- When the user is already monitoring a specific job in context (use direct kubectl commands instead)
- When the user says "update" referring to updating code, configs, or other artifacts

## Prerequisites
- All verl-* nodes are on cluster `prod-westus2-cw-6` in namespace `boren`
- Use `kubectl` directly (faster, avoids `rcall-brix` Informer sync timeouts):
  ```
  kubectl --context prod-westus2-cw-6 -n boren exec [-it] <pod>-0 -- bash -l -c '<command>'
  ```

## Procedure

### 1. Discover active verl-* nodes

List pods to find all verl-* nodes:
```bash
kubectl --context prod-westus2-cw-6 -n boren get pods -o name | grep verl | sed 's|pod/||' | sort -u
```

Or use the known set if already established in conversation context.

### 2. For each node, list jobs and check status

Use `ray_job.py list` to get all jobs with their Ray status (RUNNING/SUCCEEDED/FAILED/STOPPED) in one call:
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'python /root/code/verl/ray_job.py list 2>/dev/null'
```

This returns a compact listing with job IDs, entrypoints, and statuses. If the output shows no RUNNING submission jobs, report the node as **IDLE** and note the last job's status.

**Additionally**, check the Ray job status directly for known job IDs to confirm RUNNING vs SUCCEEDED vs FAILED:
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job status <job_id> 2>&1 | tail -3'
```

This is critical because log-based checks (`grep "Training Progress"`) only show the last logged progress — a job may have SUCCEEDED or FAILED since the last progress line was written.

**Always check both** — `ray job status` for the authoritative state, and logs for progress details.

If no RUNNING submission jobs, report the node as **IDLE** and include the last job's final status (SUCCEEDED/FAILED).

### 3. Get job details per type

#### Training jobs (GRPO/ReMax/PPO)
Check both status and progress:
```bash
# Ray job status (authoritative — always check this)
ray job status <job_id> 2>&1 | tail -3

# Progress bar (from logs)
ray job logs <job_id> 2>&1 | grep "Training Progress" | tail -n 1

# Current step
ray job logs <job_id> 2>&1 | grep -oP "step:\d+" | tail -n 1

# Val metrics (p_err per dataset group)
ray job logs <job_id> 2>&1 | grep "val-core" | grep "step:" | tail -n 1

# W&B run link
ray job logs <job_id> 2>&1 | grep "wandb.*View run" | tail -n 1

# Config name
ray job list 2>&1 | grep <job_id> | grep -oP "entrypoint='[^']+'"
```

#### Eval jobs
Check status and scoring progress:
```bash
# Ray job status (authoritative)
ray job status <job_id> 2>&1 | tail -3

# Scoring progress
ray job logs <job_id> 2>&1 | grep "len reward_extra" | tail -n 1

# Final metrics (if completed)
ray job logs <job_id> 2>&1 | grep "val-aux.*p_err" | tail -n 1
```

#### Data generation jobs
Check status and batch progress:
```bash
# Ray job status (authoritative)
ray job status <job_id> 2>&1 | tail -3

# Batch progress
ray job logs <job_id> 2>&1 | grep -E "Batch |Overall|All Done" | tail -n 3
```

#### Detect job type
- Training: entrypoint contains `main_asr_dapo` or `main_asr_remax`
- Eval: entrypoint contains `main_asr_eval`
- Data generation: entrypoint contains `main_asr_gen` or `quick_run`

### 4. Report format

Present results as a markdown table:

```
| Node | Job ID | Config/Type | Ray Status | Progress | Key Metrics |
|------|--------|-------------|------------|----------|-------------|
```

The **Ray Status** column must reflect the actual `ray job status` output (RUNNING/SUCCEEDED/FAILED/STOPPED), not inferred from logs.

Include:
- For training (RUNNING): step X/Y (Z%), latest val p_err for key datasets, W&B link
- For training (SUCCEEDED): ✅ final step, checkpoint path, total time
- For eval (RUNNING): scoring progress
- For eval (SUCCEEDED): ✅ final metrics
- For data gen (RUNNING): batch X/Y (Z%)
- For FAILED: 💥 error type and brief reason
- For idle (no jobs): last job status and reason if FAILED
- For loading: current stage (model loading, FSDP wrap, CUDA graph capture, etc.)

Flag SUCCEEDED and FAILED jobs prominently with ✅ and 💥 emoji.

### 5. Failed job diagnostics

If a job FAILED (detected via `ray job status`), get error details:
```bash
ray job logs <job_id> 2>&1 | grep -E "Error|Traceback|OOM|killed|NCCL" | tail -n 5
```

Briefly note the error type:
- `NCCL timeout` / `WorkNCCL.*timeout` → NCCL collective timeout (often OOM-related)
- `ActorDiedError` + `SIGKILL` → OOM killed by system
- `ActorUnavailableError` → pod restart/connection reset (infrastructure)
- `OutOfMemoryError` / `CUDA out of memory` → GPU OOM
- `ValueError: Total available GPUs 0` → GPU contention (another job using GPUs)
- `RayTaskError` → code error (show last traceback line)
- `ModuleNotFoundError` → missing dependency (run `ray_tool.py prepare_env`)

## Tips
- Use `kubectl exec` instead of `rcall-brix ssh` — it's faster and avoids Informer sync timeouts
- Run node checks in parallel when possible (independent reads)
- For `timeout` on long log reads: `timeout 30 ray job logs <id> 2>&1 | ...`
- Val scoring progress: `grep "len reward_extra" | tail -n 1` shows samples scored so far

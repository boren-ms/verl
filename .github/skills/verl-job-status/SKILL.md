---
name: verl-job-status
description: 'Check training/generation job status across all verl-* Brix nodes. Use when: update, job status, check training, monitor jobs, verl nodes, what is running, update all nodes, check progress, training progress, data gen progress.'
argument-hint: 'Optional node filter like verl-n2-i0, or omit for all verl-* nodes'
---

# verl-* Job Status Check

Report the status of Ray jobs running on all `verl-*` Brix nodes in a single summary table.

## When to Use
- User says "update" with no other context
- User asks to check job status across verl nodes
- User wants training progress or data generation progress
- User asks "what is running" or "check all nodes"

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

### 2. For each node, find running jobs

```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job list 2>&1 | grep "SUBMISSION.*RUNNING" | grep -oP "submission_id='\''(raysubmit_[^'\'']+)"'
```

If no RUNNING submission jobs, report the node as **IDLE**.

### 3. Get job details per type

#### Training jobs (GRPO/ReMax/PPO)
Check progress and latest val metrics:
```bash
# Progress bar
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

#### Data generation jobs
Check batch progress:
```bash
ray job logs <job_id> 2>&1 | grep -E "Batch |Overall|All Done" | tail -n 3
```

#### Detect job type
- Training: entrypoint contains `main_asr_dapo` or `main_asr`
- Data generation: entrypoint contains `main_asr_gen` or `quick_run`

### 4. Report format

Present results as a markdown table:

```
| Node | Job ID | Config/Type | Status | Progress | Key Metrics |
|------|--------|-------------|--------|----------|-------------|
```

Include:
- For training: step X/Y (Z%), latest val p_err for key datasets, W&B link
- For data gen: batch X/Y (Z%)
- For idle: last job status (FAILED/SUCCEEDED) and reason if FAILED
- For loading: current stage (model loading, FSDP wrap, CUDA graph capture, etc.)

### 5. Failed job diagnostics

If a job FAILED, briefly note the error type:
- `ActorUnavailableError` → pod restart/connection reset (infrastructure)
- `OutOfMemoryError` → OOM
- `RayTaskError` → code error (show last traceback line)

## Tips
- Use `kubectl exec` instead of `rcall-brix ssh` — it's faster and avoids Informer sync timeouts
- Run node checks in parallel when possible (independent reads)
- For `timeout` on long log reads: `timeout 30 ray job logs <id> 2>&1 | ...`
- Val scoring progress: `grep "len reward_extra" | tail -n 1` shows samples scored so far

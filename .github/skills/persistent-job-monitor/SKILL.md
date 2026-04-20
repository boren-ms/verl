---
name: persistent-job-monitor
description: 'Persistently monitor a running job until it reaches a target, polling every 5 minutes. Use when: watching a job until done, keep checking job, monitor until target step, wait for job to finish, poll job status, do not stop until job completes, track job to completion, continuous monitoring. Triggers: "keep monitoring", "don''t stop until done", "watch until finished", "poll every 5 minutes", "monitor until step X", "watch job".'
argument-hint: '<node> <job_id> [target_step|DONE] — e.g. verl-n1-i0 raysubmit_abc123 35'
---

# Persistent Job Monitor

Continuously monitor a running Ray job on a Brix node, polling every 5 minutes, and **never stop until the job reaches the specified target**. The target can be a step number, SUCCEEDED status, or FAILED status.

## When to Use
- User wants to watch a job until it finishes or reaches a specific step
- User says "keep monitoring", "don't stop", "watch until done", "poll every 5 min"
- After submitting a job and wanting hands-off monitoring to completion
- When resuming monitoring of an already-running job

## When NOT to Use
- One-time status checks (use `verl-job-status` skill instead)
- Submitting a new job (use `submit-remote-job` skill instead)

## Prerequisites
- A running Ray job ID (e.g. `raysubmit_XXXXX`)
- The Brix node name where the job is running (e.g. `verl-n1-i0`)
- Access via `kubectl` (cluster `prod-westus2-cw-6`, namespace `boren`) or `rcall-brix`

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| Node | Yes | Brix node name, e.g. `verl-n1-i0` |
| Job ID | Yes | Ray job ID, e.g. `raysubmit_abc123` |
| Target | No | Step number (e.g. `35`) or `DONE` (default: `DONE` = wait for SUCCEEDED/FAILED) |

If the user doesn't provide a job ID, discover it:
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job list 2>&1 | grep "RUNNING" | grep -oP "submission_id='\''(raysubmit_[^'\'']+)"'
```

## Procedure

### Step 1 — Resolve inputs and determine target

- Parse node name, job ID, and target from user input
- If target is a number, monitor until `step:N` where N >= target
- If target is `DONE` or omitted, monitor until job status is SUCCEEDED or FAILED
- Determine total_steps from config if possible (look for `total_training_steps` or `n_gpus * total_epochs` in logs)

### Step 2 — Initial status check

Run all diagnostic commands to establish baseline:

```bash
# Job status
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job status <JOB_ID>'

# Latest step
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job logs <JOB_ID> 2>&1 | grep "step:" | tail -n 5'

# W&B link (capture once)
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job logs <JOB_ID> 2>&1 | grep "wandb.*View run" | tail -n 1'

# GPU utilization
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

Present the initial status report (see Step 4 for format).

### Step 3 — Poll loop (every 5 minutes)

**CRITICAL: Do NOT stop the loop. Keep polling until the target is reached.**

Each polling cycle:

#### 3a. Check job status
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job status <JOB_ID>'
```

**Exit conditions** (only these stop the loop):
- Status is `SUCCEEDED` → report final metrics and stop
- Status is `FAILED` → report error details and stop
- Target step reached → report metrics at target and stop

If none of the exit conditions are met, continue to 3b.

#### 3b. Get latest training progress
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job logs <JOB_ID> 2>&1 | grep "step:" | tail -n 10'
```

#### 3c. Get validation metrics (if any new ones appeared)
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job logs <JOB_ID> 2>&1 | grep -E "val-core|val-aux" | tail -n 20'
```

#### 3d. Check GPU utilization
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

#### 3e. Check for errors
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job logs <JOB_ID> 2>&1 | grep -E "Traceback|Error|FAILED|OOM" | tail -n 5'
```

#### 3f. Wait 5 minutes
After reporting status, wait 5 minutes before the next poll. Use the agent's built-in turn notification mechanism — do NOT run `sleep` in the terminal. Instead, tell the user:

> Waiting 5 minutes before next check. Next poll at HH:MM.

Then on the next turn, continue the loop.

**IMPORTANT**: Between polls, do NOT ask the user if they want to continue. Just keep going. The user explicitly requested persistent monitoring.

### Step 4 — Status report format (show after every poll)

#### Status header
```
**Job**: `<JOB_ID>` | **Status**: RUNNING | **Progress**: step X/N (XX%)
**Node**: <NODE> | **Target**: step N / DONE
**W&B**: [<RUN_NAME>](https://msaip.wandb.io/genai/<PROJECT>/runs/<RUN_ID>)
**GPU**: 0: 95% (65G/80G) | 1: 92% (64G/80G) | ...
**Next poll**: HH:MM (in 5 min)
```

#### Training metrics table (accumulated across all polls)

| Step | Progress | score/mean | entropy | pg_loss | grad_norm | lr | throughput | step_time |
|------|----------|------------|---------|---------|-----------|-----|-----------|-----------|
| 1    | 2.9%     | 0.486      | 1.07    | -0.0008 | 0.0014    | 5e-6| 724 tok/s | 24.6s     |

Extract from `step:N` log lines:
- `critic/score/mean` → score/mean
- `actor/entropy` → entropy
- `actor/pg_loss` → pg_loss
- `actor/grad_norm` → grad_norm
- `actor/lr` → lr
- `perf/throughput` → throughput
- `timing_s/step` → step_time

#### Validation metrics table (accumulated)

| Step | p_err (WER) | p_ins_edge | n_err | n_ref | reward/mean |
|------|-------------|------------|-------|-------|-------------|
| 0    | 5.62        | 5.57       | 165.8 | 29.5  | 0.486       |

Extract from `val-core/` and `val-aux/` log lines.

**Always show ALL accumulated steps — never truncate. The full trajectory is essential for tracking trends.**

### Step 5 — Completion report

When the target is reached, show:

```
## Job Complete ✓

**Job**: `<JOB_ID>` | **Final Status**: SUCCEEDED
**Node**: <NODE> | **Total Steps**: N
**Duration**: approximately X hours
**W&B**: [<RUN_NAME>](URL)

### Final Training Metrics
<last training step metrics>

### Final Validation Metrics
<full validation trajectory table>

### Summary
- Best val p_err: X.XX at step N
- Final val p_err: X.XX
- Training trend: improving/plateaued/degrading
```

### Step 6 — Handle failures during monitoring

If the job FAILS during monitoring:
1. Get the last 40 lines of logs for error diagnosis
2. Report the error clearly
3. Ask the user if they want to:
   - Fix and resubmit (switch to `submit-remote-job` skill)
   - Abandon the job

If the job appears stuck (same step for 3+ consecutive polls = 15+ min):
1. Check GPU utilization — if GPUs are active, the job may be in a long step
2. Check for deadlock indicators in logs
3. Report the stall but do NOT automatically cancel

## Tips
- Use `kubectl exec` over `rcall-brix ssh` for speed
- Combine multiple `grep` commands into a single `kubectl exec` call to reduce round trips
- Track W&B URL once on first poll, reuse in subsequent reports
- If connection to node times out, retry once before reporting connectivity issues
- Scale WER/p_err values ×100 when displaying (e.g., 0.0891 → 8.91%)

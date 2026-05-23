---
name: verl-auto-monitor
description: 'Full-pipeline monitor for all verl-* nodes: list jobs with status/progress, auto-fix failures, auto-run post-training evals (eval_openasr + eval_openasr_ml), and generate openasr-report xlsx. Use when: "monitor all nodes", "auto monitor", "full pipeline status", "check and fix all jobs".'
argument-hint: 'Optional: interval (e.g. "every 10m"), or node filter (e.g. "i10 only")'
---

# verl Auto-Monitor — Full Pipeline

Comprehensive monitoring skill that combines node discovery, job status checking, automatic failure recovery, post-training evaluation, and report generation into a single automated pipeline.

## When to Use

- User wants a full status overview of all verl nodes with automatic follow-up actions
- User wants training jobs to automatically trigger eval_openasr + eval_openasr_ml on completion
- User wants failures auto-diagnosed and retried
- User says "monitor all", "auto monitor", "full pipeline", "check and fix"

## Prerequisites

- Cluster: `prod-westus2-cw-6`, namespace: `boren`
- All commands via kubectl for speed:
  ```
  kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c '<command>'
  ```
- Python env: `/home/boren/.virtualenvs/openai/bin/python` (for openpyxl/reports)
- Code repo: `/home/boren/code/verl`

## Full Pipeline Procedure

### Phase 1 — Discover Nodes

```bash
kubectl --context prod-westus2-cw-6 -n boren get pods -o name | grep verl | sed 's|pod/||; s|-0$//' | sort -u
```

### Phase 2 — Check All Jobs (Status + Progress)

For **each node in parallel**, run two checks:

#### 2a. Job listing with Ray status
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'python /root/code/verl/ray_job.py list 2>/dev/null'
```

This returns job IDs, entrypoints, and statuses (RUNNING/SUCCEEDED/FAILED).

#### 2b. Job-type-specific progress

Detect job type from entrypoint:
- `main_asr_remax` or `main_asr_dapo` → **Training**
- `main_asr_eval` → **Eval**
- `main_asr_gen` → **DataGen**

**Training** (RUNNING):
```bash
ray job logs <job_id> 2>&1 | grep 'Training Progress' | tail -1
```

**Eval** (RUNNING):
```bash
ray job logs <job_id> 2>&1 | grep 'len reward_extra' | tail -1
```

**DataGen** (RUNNING):
```bash
ray job logs <job_id> 2>&1 | grep -E 'Batch ' | tail -1
```

#### 2c. Report table

```
| Node | Job ID | Type | Config | Ray Status | Progress | Notes |
|------|--------|------|--------|------------|----------|-------|
```

- RUNNING → 🔄 with progress
- SUCCEEDED → ✅ with final step/checkpoint
- FAILED → 💥 with error summary
- IDLE → ⬜ no active jobs

#### 2d. GPU Health Check (1-min average utilization)

For each node with a RUNNING job, check the **1-minute average** GPU utilization to detect silently crashed/hung jobs (Ray status still shows RUNNING but GPUs are idle).

```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits && echo "---DMON---" && timeout 60 nvidia-smi dmon -s u -d 5 -c 12 2>/dev/null'
```

This runs `nvidia-smi dmon` for 60 seconds (12 samples × 5s interval) to get a 1-minute rolling GPU utilization average.

**Parsing the dmon output:**
- Each line has columns: `gpu_idx  sm_util  mem_util  enc_util  dec_util`
- Average `sm_util` across all GPUs and all 12 samples = **1-min avg GPU util**
- Skip header lines (starting with `#`)

**Interpretation thresholds:**

| 1-min Avg GPU Util | Memory Loaded (>30G/GPU) | Diagnosis | Action |
|--------------------|--------------------------|-----------|--------|
| >10% | Yes | ✅ **Healthy** — actively computing | None |
| 0-10% | Yes | ⚠️ **Possibly stalled** — loaded but idle | Check Ray job logs for errors; may be between steps (save/val), wait one more cycle before acting |
| 0% | No (<5G/GPU) | 🔄 **Loading** — model not yet loaded | Normal during startup (first 5-10 min). If >15 min, may be stuck |
| 0% | Yes (>30G/GPU) | 💥 **Likely crashed** — model loaded but no compute | Check `ray job status` — if still RUNNING, check logs for hang/deadlock. If FAILED, trigger Phase 3 auto-fix |

**Quick single-shot check (when dmon is too slow):**
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'for i in $(seq 1 6); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 10; done'
```
This takes 6 samples over 60 seconds. Average across all samples and GPUs for 1-min avg.

**Add GPU columns to the status table:**

```
| Node | Job | Type | Ray Status | Progress | GPU Util (1m avg) | GPU Mem | Health |
|------|-----|------|------------|----------|-------------------|---------|--------|
```

**Auto-action on low GPU util:**
1. If a RUNNING job shows 0% GPU util for 1 min AND memory is loaded (>30G/GPU):
   - First, check `ray job logs <job_id> 2>&1 | tail -20` for recent errors
   - If logs show NCCL timeout, deadlock, or no new output for >10 min → mark as **crashed**
   - Trigger Phase 3 (Auto-Fix) for the crashed job
2. If GPU util is 0% but memory is <5G/GPU → job is still loading, skip
3. If GPU util is low (1-10%) with high memory → likely doing checkpoint save or validation pass, wait

### Phase 3 — Auto-Fix Failures

When a job has FAILED status, automatically diagnose and attempt recovery.

#### 3a. Get error details
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'ray job logs <job_id> 2>&1 | tail -30'
```

#### 3b. Diagnosis → Fix mapping

| Error Pattern | Diagnosis | Auto-Fix |
|---------------|-----------|----------|
| `NCCL timeout` / `WorkNCCL.*timeout` | NCCL collective timeout (often OOM) | Resubmit same job (transient). If fails twice, reduce `rollout_n` or `val_batch_size` in config |
| `ActorDiedError` + `SIGKILL` / `OOM` | GPU OOM | Resubmit with reduced `gpu_memory_utilization` (0.85→0.80) or `val_batch_size` |
| `ActorUnavailableError` | Pod restart / infra | Resubmit same job unchanged |
| `ValueError: Total available GPUs 0` | GPU contention | Wait for other job to finish, then resubmit |
| `ModuleNotFoundError` | Missing dependency | Run `ray_tool.py prepare_env`, then resubmit |
| `CUDA out of memory` | GPU OOM during forward/backward | Reduce batch size or `rollout_n` |
| `FileNotFoundError` / `blobfile` error | Missing checkpoint or data | Verify blob path exists with `bbb ls`, fix path if needed |
| `hydra.*error` / `ConfigError` | Bad config | Report to user — cannot auto-fix config errors |

#### 3c. Auto-fix procedure

1. **Push latest code** (in case fix was applied locally):
   ```bash
   bpush <node>
   ```

2. **Prepare env** (fixes missing deps):
   ```bash
   kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
     'cd /root/code/verl && python3 ray_tool.py prepare_env'
   ```

3. **Resubmit the failed job** using the same entrypoint from `ray job list`:
   ```bash
   kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
     'cd /root/code/verl && ray job submit --working-dir=/root/code/verl --no-wait -- <original_entrypoint>'
   ```

4. **Track the retry**: Note the new job ID, report the fix applied.

5. **If auto-fix fails twice**: Stop retrying and report to user with full error details.

### Phase 4 — Post-Training Auto-Eval

When a **training job** reaches SUCCEEDED status:

#### 4a. Find the latest checkpoint
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'bbb ls az://orngwus2cresco/data/boren/outputs/verl_repeat/<train_config>/ 2>/dev/null | grep global_step_ | sort -t_ -k3 -n | tail -1'
```

Set `CHECKPOINT_PATH` to the full blob path of the latest `global_step_*`.

#### 4b. Submit eval_openasr
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'cd /root/code/verl && ray job submit --working-dir=/root/code/verl --no-wait -- \
   python3 -m recipe.phimm.main_asr_eval \
   --config-name eval_openasr \
   trainer.experiment_name=<train_config>_eval_openasr \
   trainer.resume_mode=resume_path \
   trainer.resume_from_path=<CHECKPOINT_PATH>'
```

#### 4c. Monitor eval_openasr until completion

Poll every 5 minutes:
```bash
ray job status <eval_job_id> 2>&1 | tail -3
```

When SUCCEEDED, capture metrics:
```bash
ray job logs <eval_job_id> 2>&1 | grep 'val-aux.*p_err/mean@1'
```

#### 4d. Submit eval_openasr_ml (after eval_openasr completes)
```bash
kubectl --context prod-westus2-cw-6 -n boren exec <node>-0 -- bash -l -c \
  'cd /root/code/verl && ray job submit --working-dir=/root/code/verl --no-wait -- \
   python3 -m recipe.phimm.main_asr_eval \
   --config-name eval_openasr_ml \
   trainer.experiment_name=<train_config>_eval_openasr_ml \
   trainer.resume_mode=resume_path \
   trainer.resume_from_path=<CHECKPOINT_PATH>'
```

Monitor until completion, then capture ML metrics.

**Note**: If the node is busy (another eval running), use a different idle node. Check all nodes for availability first.

#### 4e. Eval sequencing

Run eval_openasr first, then eval_openasr_ml sequentially on the same node (they need all 8 GPUs). If multiple training jobs complete simultaneously on different nodes, run evals in parallel across nodes.

### Phase 5 — Generate OpenASR Report

After **both** eval_openasr and eval_openasr_ml complete for a model:

#### 5a. Extract metrics to JSON

Parse the Ray job logs for both eval jobs and create a metrics JSON:
```python
# From eval_openasr logs — extract p_err for 8 datasets
# ami, earnings22, gigaspeech, ls_clean, ls_other, spgispeech, tedlium, voxpopuli

# From eval_openasr_ml logs — extract p_err for 13 datasets
# de_fleurs, de_mcv, es_fleurs, es_mcv, es_mls, fr_fleurs, fr_mcv, fr_mls,
# it_fleurs, it_mcv, it_mls, pt_fleurs, pt_mls
```

Save combined metrics to `tmp/openasr_report/<model_label>.json`.

#### 5b. Build xlsx report
```bash
cd /home/boren/code/verl && \
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/openasr-report/scripts/build_openasr_xlsx.py \
  "<train_config>@step<N>" \
  --metrics tmp/openasr_report/<model_label>.json \
  --out tmp/openasr_report/<model_label>.xlsx
```

#### 5c. Extend existing report (if prior report exists)

If there's an existing report to extend:
```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/openasr-report/scripts/build_openasr_xlsx.py \
  "<train_config>@step<N>" \
  --metrics tmp/openasr_report/<model_label>.json \
  --extend-xlsx tmp/openasr_report/<existing>.xlsx \
  --out tmp/openasr_report/<combined>.xlsx
```

#### 5d. Present summary

Show a summary table with:
- Model label and checkpoint path
- OpenASR avg p_err and WERR vs baseline
- OpenASR-ML avg p_err and WERR vs baseline (with per-language breakdown)
- Link to xlsx file

### Phase 6 — Scheduled Monitoring (Optional)

If user requests continuous monitoring, set up a schedule:

```
Every N minutes:
1. Run Phase 2 (status check)
2. If any job FAILED → Phase 3 (auto-fix)
3. If any training SUCCEEDED → Phase 4 (auto-eval)
4. If any eval pair completed → Phase 5 (report)
5. Present compact status update
```

Use the `manage_schedule` tool with an appropriate interval (default: 10 minutes).

## Report Format

### Per-check status table
```
## verl Status — HH:MM UTC

| Node | Type | Config | Status | Progress |
|------|------|--------|--------|----------|
| i1   | Train | no_repeat_n8_e1 | ✅ SUCCEEDED | 205/205 → auto-eval queued |
| i3   | DataGen | entity_en_hc | 🔄 RUNNING | 695/2265 (31%) |
| i10  | Eval | eval_openasr | 🔄 RUNNING | scoring... |
| i11  | Train | no_repeat_s100 | 💥 FAILED | NCCL timeout → auto-retrying |
| i12  | — | — | ⬜ IDLE | last: SUCCEEDED |
```

### Post-eval report summary
```
## OpenASR Report — <model>@step<N>

| Metric | OpenASR | OpenASR-ML |
|--------|---------|------------|
| Avg p_err | X.XX% | X.XX% |
| WERR vs baseline | +X.X% | +X.X% |

📄 Report: tmp/openasr_report/<model>.xlsx
```

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **verl-job-status** | 2 | Node discovery and status checking patterns |
| **verl-asr-run** | 3, 4 | Job submission, monitoring, and failure handling |
| **openasr-report** | 5 | Excel report generation with baseline comparison |

## Error Handling

- **Max 2 auto-retries** per failed job. After 2 failures, stop and report to user.
- **GPU contention**: If a node has no free GPUs, queue the eval and try another node or wait.
- **Checkpoint not found**: Verify blob path with `bbb ls`. If missing, report and skip eval.
- **Config errors**: Cannot auto-fix — report to user with the specific config issue.
- **Node not responding**: Skip node, report as unreachable, continue with others.

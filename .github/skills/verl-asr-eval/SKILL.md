---
name: verl-asr-eval
description: 'Run ASR evaluation on a remote verl Brix node and perform word error analysis on the results. Use when evaluating a verl ASR model checkpoint on LibriSpeech, OpenASR, or entity datasets, submitting eval jobs via quick_run.sh, monitoring Ray job progress, and analyzing per-dataset WER with word-level error breakdowns. Combines remote-development patterns with verl eval pipeline and asr-word-error-analysis.'
argument-hint: 'Eval config name and optional node, e.g. eval_libri_h100 on verl-n1-i0'
---

# verl ASR Evaluation

Run an ASR evaluation job on a remote verl Brix node, monitor it to completion, then perform word error analysis on the validation output.

## When to Use

- User wants to evaluate a verl ASR model checkpoint (e.g. Phi4-7b-STT) on eval datasets
- User says "run eval", "evaluate on librispeech", "eval_openasr", "check WER"
- User wants to submit a verl eval job to a remote node and wait for results
- User wants word-level error analysis on verl validation JSONL output

## Inputs

| Parameter | Required | Default | Example |
|-----------|----------|---------|---------|
| **eval_config** | No | `eval_libri_h100` | `eval_openasr`, `eval_libri_h100` |
| **node** | No (auto) | First Ready `verl-*` node | `verl-n1-i0`, `verl-n2-i1` |
| **model_path** | No | From config | `/data/boren/data/ckp/hf_models/Phi4-7b-STT-2603-SR2` |
| **word_error_sets** | No | `1` (first data source only) | `all` (all data sources) |

## Available Eval Configs

| Config | Datasets | Notes |
|--------|----------|-------|
| `eval_libri_h100` | LibriSpeech (h100 subset) | Fast eval, uses TP=2 |
| `eval_openasr` | OpenASR (ami, common_voice, earnings22, etc.) | Full OpenASR suite |

Configs live at `recipe/phimm/config/eval_*.yaml` and compose from `recipe/phimm/config/base/eval_asr.yaml` which sets `val_only: True` and `val_before_train: True`.

## Procedure

### Step 0 — Find a Ready verl node and check occupancy

1. **List all verl nodes** and their status:
   ```bash
   rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-'
   ```
   Separate into Ready nodes and Paused/Suspended nodes.

2. **Check occupancy** on each Ready node using two signals:

   **a) GPU utilization** — check if GPUs are actively in use:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
   ```
   A node is **GPU-busy** if any GPU shows utilization > 5% or memory used > 5000 MiB.

   **b) Ray jobs** — check for running Ray jobs:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list 2>/dev/null || echo No Ray jobs"'
   ```

   A node is considered **occupied** if either check is positive (GPU-busy OR running Ray jobs). A node is **free** only if GPUs are idle AND no Ray jobs are running.

3. **Assign node**: pick the first free (unoccupied) Ready `verl-*` node. If the user specified a node, use that even if it is busy (see below).

4. **If the chosen node is busy** (GPU-busy or has running Ray jobs):
   - **Do NOT kill or stop the existing job.** Report what is currently running (job ID, config name, runtime, GPU utilization).
   - **Wait for the node to become free.** Poll every **5 minutes** using the occupancy checks from step 2.
   - After each poll, report status: `"Node {NODE} still busy — GPU util: {UTIL}%, Ray jobs: {COUNT}. Waiting..."`.
   - Once the node is idle (GPUs idle AND no running Ray jobs), proceed to Step 1.

5. **If no unoccupied Ready node exists** and no specific node was requested:
   - Check if any `verl-*` nodes are in **Paused** or **Suspended** state:
     ```bash
     rcall-brix ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Paused|Suspended' | awk '{print $1}' | grep '^verl-'
     ```
   - If a Paused/Suspended node is found, **automatically resume** the first one:
     ```bash
     rcall-brix resume {NODE}
     ```
   - **Poll until Ready**: check status every 15 seconds until the node reaches `Ready`:
     ```bash
     rcall-brix ls '{NODE}' 2>&1
     ```
     Report each poll: `"Resuming {NODE}... status: {STATUS}"`.
   - Once `Ready`, use this node and proceed to Step 1.
   - If **no Paused/Suspended nodes** exist either, report the status of all `verl-*` nodes and ask the user which busy node to wait on.

### Step 1 — Resolve inputs

- **eval_config**: Extract from the user's request. Default: `eval_libri_h100`. The config name is the YAML filename without `.yaml` under `recipe/phimm/config/`.
- **model_path**: Usually baked into the eval config. If the user specifies a custom model path, it will be passed as a hydra override.
- **word_error_sets**: Default `1` — only analyze the first data source. If user says "all", analyze all data sources.

### Step 2 — Push code and submit the eval job

**IMPORTANT**: Always push the latest code to the remote node before submitting any job. This ensures the remote node runs the same code as your local workspace.

1. **Push code** to the remote node using `bpush`:
   ```bash
   bpush {NODE}
   ```
   This commits and pushes the current workspace to the node's git checkout. Alternatively, use `submit_job.sh` which handles both push and submit (see step 3b).

2. **Clean up** any previous job with the same config name:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py cleanup {EVAL_CONFIG}"'
   ```

3. **Submit the eval job** using `quick_run.sh`:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l /root/code/verl/quick_run.sh recipe/phimm/config/{EVAL_CONFIG}.yaml'
   ```

   **3b. Alternative (preferred)**: Use `submit_job.sh` which handles push + submit in one command:
   ```bash
   bash submit_job.sh {NODE} recipe/phimm/config/{EVAL_CONFIG}.yaml false true false
   ```

   If a custom model path is specified, append a hydra override:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m recipe.phimm.main_asr_dapo --config-name {EVAL_CONFIG} trainer.experiment_name={EVAL_CONFIG} actor_rollout_ref.model.path={MODEL_PATH}"'
   ```

   Save the output — it contains the Ray job ID (e.g. `raysubmit_XXXX`).

4. **Capture local log**:
   ```bash
   mkdir -p logs/{NODE}
   ```
   The job output is also streamed to `{EVAL_CONFIG}.log` on the node.

### Step 3 — Monitor until completion

Poll every **5 minutes** until the job finishes. The eval typically takes 10–30 minutes depending on dataset size and GPU count.

1. **Check Ray job status**:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'
   ```

2. **Tail the log** for progress:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "tail -30 /root/code/verl/{EVAL_CONFIG}.log"'
   ```

3. **Detect completion**: The job is done when:
   - `ray_job.py list` shows no running jobs matching `{EVAL_CONFIG}`, OR
   - The log contains `"Initial validation metrics:"` followed by the metrics dict, OR
   - The log contains a final WER summary line

4. **If the job crashes**, check the log tail for errors. Common fixes:
   - OOM: reduce `gpu_memory_utilization` or `val_batch_size`
   - Missing data: check `DATA_PATH` env var on the node

5. **Extract metrics from the log** once done:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "grep -A 50 \"Initial validation metrics\" /root/code/verl/{EVAL_CONFIG}.log | head -60"'
   ```

   The metrics dict looks like:
   ```
   val-core/{data_source}/reward/mean@1/0 = <value>
   val-aux/{data_source}/n_err/sum@1/0 = <total_errors>
   val-aux/{data_source}/n_ref/sum@1/0 = <total_ref_words>
   ```

   **Compute WER** per data source: `WER = n_err / n_ref`.

6. Present results as a table:
   ```
   | Dataset | WER | Errors | Ref Words |
   |---------|-----|--------|-----------|
   | librispeech | 5.2% | 1234 | 23456 |
   | ami | 18.3% | 5678 | 31000 |
   ```

### Step 4 — Word Error Analysis

After the eval job completes, perform word-level error analysis on the validation JSONL output.

1. **Locate the validation data** on the remote node. The output is saved to `{default_local_dir}/val_data_gen/{data_source}/{step}.jsonl`:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "find /root/code/verl/outputs/ -path \"*/val_data_gen/*/*.jsonl\" -newer /root/code/verl/{EVAL_CONFIG}.log 2>/dev/null | head -20"'
   ```

   If `validation_data_dir` is not set in config, the JSONL files may not be dumped. In that case, skip to Step 4b.

   **Alternative**: Check the hydra output directory:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "ls -la /root/code/verl/outputs/{EVAL_CONFIG}*/val_data_gen/ 2>/dev/null || echo No val_data_gen found"'
   ```

2. **Download the JSONL files** locally:
   ```bash
   rcall-brix ssh {NODE} -- 'bash -l -c "find /root/code/verl/outputs/ -path \"*/val_data_gen/*/*.jsonl\" | head -20"'
   ```
   For each file:
   ```bash
   rcall-brix scp {NODE}:{REMOTE_JSONL_PATH} /tmp/verl_eval/{DATA_SOURCE}/
   ```
   Or use `bbb cp` if the data was uploaded to Azure.

3. **Run word error analysis** using the `asr-word-error-analysis` script. The verl validation JSONL uses `gts` for reference and `output` for hypothesis:

   ```bash
   python ~/.github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
     --input-path /tmp/verl_eval/{DATA_SOURCE}/{STEP}.jsonl \
     --ref-column gts \
     --hyp-column output \
     --dataset {DATA_SOURCE} \
     --output-dir ~/data/results/verl_word_error/{EVAL_CONFIG}/{DATA_SOURCE}/ \
     --write-html \
     --top-n 30
   ```

   By default, run analysis on **only the first data source** (e.g. just `librispeech`). If the user requested `all`, run for every data source found.

4. **Present word error analysis results**: Read `summary.json` and show:
   - Overall WER, substitution/deletion/insertion rates
   - Top 10 substitution confusion pairs from `substitutions.csv`
   - Top 5 deletions and insertions
   - Link to the HTML report for visual inspection

#### Step 4b — Fallback if no JSONL dump

If `validation_data_dir` is not configured and no JSONL files are found:

1. The WER metrics are still available from the log (Step 3.5–3.6). Present those.
2. Suggest re-running with `trainer.validation_data_dir` set:
   ```
   Add to the eval config or as a hydra override:
     trainer.validation_data_dir=/root/code/verl/outputs/{EVAL_CONFIG}/val_data_gen
   ```
3. If the user agrees, re-submit the job with the override and repeat from Step 2.

## Quality Checks

- **Step 2**: `ray job submit` output shows a job ID. If it errors, check `ray_tool.py prepare_env` ran.
- **Step 3**: WER values are reasonable (0–100%). If WER > 50%, flag as suspicious.
- **Step 4**: `summary.json` exists and WER matches the log metrics (within rounding).
- **Word error HTML**: `report.html` is generated and not empty.

## Important Notes

- The verl eval configs use hydra. The config search path is `recipe/phimm/config/` with base configs in `recipe/phimm/config/base/`.
- `quick_run.sh` determines the module automatically: configs starting with `gen_*` use `main_asr_gen`, otherwise `main_asr_dapo`.
- Eval configs inherit from `base/eval_asr.yaml` which sets `val_only: True` — only validation runs, no training.
- The remote workspace is at `/root/code/verl` on Brix nodes.
- **Always push code first** before submitting jobs. Use `bpush {NODE}` or `submit_job.sh` (which pushes automatically). Never submit a job without syncing code first.
- For remote operations, use `rcall-brix ssh`, `rcall-brix scp`, or the convenience wrappers `bpush` and `submit_job.sh`.
- The word error analysis script (`analyze_word_errors.py`) supports custom column names via `--ref-column` and `--hyp-column`. verl JSONL uses `gts` and `output`.

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **remote-development** | 0, 2 | Node discovery, sync, remote commands |
| **asr-word-error-analysis** | 4 | Word-level error analysis on validation JSONL |

---
name: verl-asr-run
description: 'Run the full ASR stack on remote verl Brix nodes: training -> checkpoint evaluation (`long_eval_inhouse_2605_all_seg`) -> report generation, while continuously monitoring until completion with structured metrics. Use when: running RL training (ReMax, GRPO), automatically running the post-training in-house long-audio all-locale eval, evaluating checkpoints on LibriSpeech/in-house/entity datasets, submitting jobs via quick_run.sh, monitoring Ray job progress, tracking training metrics, pushing code and resubmitting after fixes, and analyzing per-dataset WER with word-level error breakdowns. Triggers: "submit job", "train on remote", "launch training", "run eval", "evaluate on librispeech", "long_eval_inhouse", "check WER", "monitor job", "check training status", "push and submit", "run config on node", "training evaluation report".'
argument-hint: 'Config name and optional node, e.g. remax_ls_lr05 on verl-n1-i0, or eval_libri_h100'
---

# verl ASR Run

Run a full ASR pipeline on a remote verl Brix node: **training -> evaluation -> report -> persistent auto-monitor**. Submit jobs via `submit_job.sh`, continuously monitor until completion with structured metrics, automatically evaluate the trained checkpoint on `long_eval_inhouse_2605_all_seg` (all locales: en-US, da-DK, hu-HU, nb-NO, nl-NL, cs-CZ), then report the aggregate TER/EER measures. Optionally perform word error analysis on validation output. After submission, always install a `/every 5m update job status and autofix job` schedule (Step 6) so status updates and auto-fixes continue without manual re-prompting.

Refer to the **remote-development** skill for node connectivity, `brix`, `bpush`, `bbb`, and environment setup.

## When to Use

- User wants to **train** an ASR model with RL (ReMax, GRPO) — "run training", "submit job", "train on node"
- User wants to **evaluate** a checkpoint — "run eval", "evaluate on librispeech", "long_eval_inhouse", "check WER"
- User wants to submit any verl ASR job to a remote node and monitor until completion
- User asks to monitor an existing job — "check status", "update", "how's the job"
- User needs to fix code, push, and resubmit after a failure
- User wants full stack execution: training followed by the `long_eval_inhouse_2605_all_seg` eval and report generation
- User wants word-level error analysis on verl validation JSONL output
- User wants to run multiple jobs (see batch submission below)

## Inputs

| Parameter | Required | Default | Example |
|-----------|----------|---------|---------|
| **config** | Yes | — | `remax_ls_lr05`, `eval_libri_h100`, `gen_libri` |
| **node** | No (auto) | First Ready `verl-*` node | `verl-n1-i0`, `verl-n2-i1` |
| **model_path** | No | From config | `/data/boren/data/ckp/hf_models/Phi4-7b-STT-2603-SR2` |
| **post_train_eval** | No | Always run `long_eval_inhouse_2605_all_seg` for training jobs | `long_eval_inhouse_2605_all_seg` |
| **report** | No | In-house DTER `.xlsx` report (inhouse-dter-report, `--schema all_seg`) + HTML chart report (inhouse-dter-html) + TER/EER summary after the eval succeeds | `inhouse_dter_report/*.xlsx` + `*.html` |
| **word_error_sets** | No | `1` (first data source only) | `all` (all data sources) |

## Job Types

Determined by config name prefix:

| Prefix | Module | Type | Notes |
|--------|--------|------|-------|
| `remax_*` | `recipe.phimm.main_asr_remax` | Training | RL training with validation at intervals |
| `grpo_*` and other training configs | `recipe.phimm.main_asr_dapo` | Training | RL training with validation at intervals |
| `long_eval_*` | `recipe.phimm.main_long_eval_asr` | Long-audio eval | Gen-style eval; SVAD-explode -> generate -> regroup -> TER/EER; writes `result_details.jsonl` + `measures.json` |
| `eval_*` | `recipe.phimm.main_asr_eval` | Eval-only | `val_only: True`, runs validation then exits |
| `gen_*` | `recipe.phimm.main_asr_gen` | Generation | Inference/generation only |

## Available Configs (examples)

### Training Configs
| Config | Algorithm | Notes |
|--------|-----------|-------|
| `remax_ls_*` | ReMax | LibriSpeech RL training |
| `grpo_*` | GRPO | GRPO RL training |

### Eval Configs
| Config | Datasets | Notes |
|--------|----------|-------|
| `eval_libri_h100` | LibriSpeech (h100 subset) | Fast eval |
| `long_eval_inhouse_2605_all_seg` | In-house 2605 all locales (en-US, da-DK, hu-HU, nb-NO, nl-NL, pre-segmented) | Long-audio gen-style eval; TER/EER measures (**default post-training eval**) |
| `long_eval_inhouse_2605_enus_seg` | In-house 2605 en-US only (pre-segmented) | Long-audio gen-style eval; en-US TER/EER only |
| `eval_openasr` | OpenASR (ami, common_voice, earnings22, etc.) | Full OpenASR suite |
| `eval_openasr_ml` | OpenASR-ML (FLEURS, MCV, MLS by language) | Multilingual OpenASR suite; report per-language and overall averages |

Configs live under `recipe/phimm/config/` by job family, including `recipe/phimm/config/remax/`, `recipe/phimm/config/eval/`, and `recipe/phimm/config/gen/`. Training configs compose from `recipe/phimm/config/base/dapo_asr.yaml`; eval configs compose from `recipe/phimm/config/base/eval_asr.yaml` (which sets `val_only: True` and `val_before_train: True`).

## Job Submission Pipeline

```
submit_jobs_repeat.sh  (batch wrapper — calls submit_job.sh N times)
  └─ submit_job.sh <node> <config> [dry_run] [cleanup] [sync_code]
       ├─ bpush <node>                    # push code to remote (if sync_code=true)
       ├─ ray_job.py cleanup <config>     # cancel previous run of same config
       └─ brix ssh <node> -- "bash -l /root/code/verl/quick_run.sh <config>"
            └─ quick_run.sh <config>
                 ├─ ray_tool.py prepare_env   # install deps on all Ray nodes
                 └─ ray job submit ... python3 -m <module> --config-name <config>
```

## Procedure

### Step 0 — Find a Ready verl node and check occupancy

1. **List all verl nodes** and their status:
   ```bash
   brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-'
   ```
   Separate into Ready nodes and Paused/Suspended nodes.

2. **Check occupancy** on each Ready node using two signals:

   **a) GPU utilization** — check if GPUs are actively in use:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
   ```
   A node is **GPU-busy** if any GPU shows utilization > 5% or memory used > 5000 MiB.

   **b) Ray jobs** — check for running Ray jobs:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list 2>/dev/null || echo No Ray jobs"'
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
     brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Paused|Suspended' | awk '{print $1}' | grep '^verl-'
     ```
   - If a Paused/Suspended node is found, **automatically resume** the first one:
     ```bash
     brix resume {NODE}
     ```
   - **Poll until Ready**: check status every 15 seconds until the node reaches `Ready`:
     ```bash
     brix pools '{NODE}' 2>&1
     ```
     Report each poll: `"Resuming {NODE}... status: {STATUS}"`.
   - Once `Ready`, use this node and proceed to Step 1.
   - If **no Paused/Suspended nodes** exist either, report the status of all `verl-*` nodes and ask the user which busy node to wait on.

### Step 1 — Resolve inputs

- **config**: Extract from the user's request (required). The config name is the YAML filename without `.yaml` under `recipe/phimm/config/`.
- **Job type**: Determined automatically from config prefix:
   - `gen_*` → generation (uses `main_asr_gen` module)
   - `eval_*` → eval-only (uses `main_asr_eval` with `val_only: True`)
   - `remax_*` → ReMax training (uses `main_asr_remax`)
   - Everything else → training (uses `main_asr_dapo`)
- **model_path**: Usually baked into the config. If the user specifies a custom model path, it will be passed as a hydra override.
- **post_train_eval**: For training jobs, always run `long_eval_inhouse_2605_all_seg` on the completed checkpoint as part of the full-stack pipeline (covers en-US, da-DK, hu-HU, nb-NO, nl-NL).
- **report**: After the post-training eval succeeds, build the canonical in-house DTER `.xlsx` report with the **inhouse-dter-report** skill (`--schema all_seg`) and surface the headline TER/EER from `measures.json` (Step 4c).
- **word_error_sets**: Default `1` — only analyze the first data source. If user says "all", analyze all data sources.

### Step 2 — Push code and submit the job

**IMPORTANT**: Always push the latest code to the remote node before submitting any job. This ensures the remote node runs the same code as your local workspace.

1. **Push code** to the remote node using `bpush`:
   ```bash
   bpush {NODE}
   ```
   This commits and pushes the current workspace to the node's git checkout. Alternatively, use `submit_job.sh` which handles both push and submit (see step 3b).

2. **Clean up** any previous job with the same config name:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py cleanup {CONFIG}"'
   ```

3. **Submit the job** using `quick_run.sh`:
   ```bash
   brix ssh {NODE} -- 'bash -l /root/code/verl/quick_run.sh recipe/phimm/config/{CONFIG}.yaml'
   ```

   **3b. Alternative (preferred)**: Use `submit_job.sh` which handles push + submit in one command:
   ```bash
   bash submit_job.sh {NODE} recipe/phimm/config/{CONFIG}.yaml false true true
   ```

   If a custom model path is specified, append a hydra override:
   ```bash
   brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m {MODULE} --config-name {CONFIG} trainer.experiment_name={CONFIG} actor_rollout_ref.model.path={MODEL_PATH}"'
   ```

   Save the output — it contains the Ray job ID (e.g. `raysubmit_XXXX`).

4. **Capture local log**:
   ```bash
   mkdir -p logs/{NODE}
   ```
   The job output is also streamed to `{CONFIG}.log` on the node.

### Step 3 — Monitor until completion

Poll periodically until the job finishes. **Do NOT stop after a single check — keep monitoring until the job reaches SUCCEEDED or FAILED.**

- **Eval jobs**: Poll every **5 minutes**. Typically takes 10–30 minutes.
- **Training jobs**: Poll every **5 minutes** during startup, then adapt cadence based on step time. Training can take hours to days.

#### 3a. Check Ray job status
```bash
brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'
```

Or by job ID:
```bash
brix ssh {NODE} -- 'bash -l -c "ray job status {JOB_ID}"'
```

#### 3b. Tail the log for progress
```bash
brix ssh {NODE} -- 'bash -l -c "tail -30 /root/code/verl/{CONFIG}.log"'
```

Or via Ray logs:
```bash
brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} | tail -n 30"'
```

#### 3c. Check GPU utilization
```bash
brix ssh {NODE} -- 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```
Run this on every monitoring check. Parse the CSV output into the status header.

#### 3d. Monitoring phases (check in order)
1. **Startup**: reward functions loaded, dataset loaded, worker count
2. **Model sync**: `bbb sync` fetching model shards from blob
3. **Checkpoint loading**: `Loading checkpoint shards: X%`
4. **LoRA application**: `Applying LoRA to actor module`
5. **vLLM init**: LoRA warnings about vision encoder layers (expected)
6. **CUDA graph capture**: `Capturing CUDA graph shapes: X% | N/M`
7. **W&B init**: `wandb: Syncing run <NAME>` — capture the **run URL**
8. **Training**: `step:N` lines with metrics
9. **Validation**: `val-core/` and `val-aux/` metrics at validation steps
10. **Checkpointing**: `save_checkpoint` timing at save steps

#### 3e. Detect completion

**For eval jobs**, the job is done when:
- `ray_job.py list` shows no running jobs matching `{CONFIG}`, OR
- The log contains `"Initial validation metrics:"` followed by the metrics dict, OR
- The log contains a final WER summary line

**For training jobs**, the job is done when:
- Ray job status shows `SUCCEEDED` or `FAILED`, OR
- The log shows the final step completed and checkpoint saved

#### 3f. Handle failures

If the job status is FAILED or errors appear:
1. Get the error from logs: `ray job logs {JOB_ID} | tail -n 40`
2. Diagnose the root cause (read the traceback)
3. Fix the code locally
4. Push (`bpush {NODE}`) and resubmit (go back to Step 2)
5. Track all job IDs across resubmissions

Common failure patterns:
- **OOM**: reduce `gpu_memory_utilization`, `val_batch_size`, or `rollout_n` in config
- **Missing data**: check `DATA_PATH` env var on the node
- **Missing function**: reward module doesn't have the expected function → add it
- **Import error**: wrong module path or API mismatch → fix the call
- **Model incompatibility**: PEFT/vLLM doesn't support the model → revert model or patch
- **Checkpoint error**: model shard missing on blob → check blob path in config
- **"brix: command not found"**: Ensure `~/.openai/bin` is on PATH
- **Sync fails**: Check node is Ready with `brix pools`

#### 3g. Extract and present metrics

**Status header** (show on every poll):
```
**Job**: `{JOB_ID}` | **Status**: RUNNING | **Progress**: step X/N (XX%)
**Node**: {NODE} | **Config**: {CONFIG}
**W&B**: [{RUN_NAME}](https://msaip.wandb.io/genai/{PROJECT}/runs/{RUN_ID})
**Ray**: http://{HEAD_IP}:9209
**GPU**: 0: 95% (65G/80G) | 1: 92% (64G/80G) | ...
```

- Extract the W&B URL from logs: look for `wandb: 🚀 View run at https://...`
- Extract the Ray dashboard from logs: look for the Ray dashboard URL in startup output
- GPU line: format each GPU as `{idx}: {util}% ({mem_used}G/{mem_total}G)` from `nvidia-smi` output

**Training metrics table** (for training jobs — accumulate all observed steps):

| Step | Progress | score/mean | entropy | pg_loss | grad_norm | lr | throughput | step_time |
|------|----------|------------|---------|---------|-----------|-----|-----------|-----------|
| 1    | 2.9%     | 0.486      | 1.07    | -0.0008 | 0.0014    | 5e-6| 724 tok/s | 24.6s     |
| 2    | 5.7%     | 0.501      | 1.05    | -0.0012 | 0.0018    | 5e-6| 731 tok/s | 24.2s     |

Extract from `step:N` log lines:
- `critic/score/mean` → score/mean
- `actor/entropy` → entropy
- `actor/pg_loss` → pg_loss
- `actor/grad_norm` → grad_norm
- `actor/lr` → lr (scientific notation)
- `perf/throughput` → throughput (append tok/s)
- `timing_s/step` → step_time (append s)
- `progress` → Progress (format as %)

**Validation metrics** (for both training and eval — accumulate across val steps):

Only report metrics with the `p_` prefix. Do not include raw count metrics (`n_err`, `n_ref`, `wer`, etc.) or `reward/mean`. All `p_` values are ratios (0–1 in logs); display as percentages (×100) with a `%` suffix on every number.

Extract from validation log lines:
- `val-aux/{data_source}/p_err/mean@1` → p_err
- `val-aux/{data_source}/p_edge/mean@1` → p_edge
- `val-aux/{data_source}/p_fmt/mean@1` → p_fmt
- `val-aux/{data_source}/p_lang/mean@1` → p_lang
- `val-aux/{data_source}/p_bracket/mean@1` → p_bracket

Report each `p_` metric in its own **separate table**:

**p_err table** (primary — always show):

| Step | Dataset | p_err |
|------|---------|-------|
| 0    | librispeech | 5.62% |
| 10   | librispeech | 4.98% |

**p_edge table** (separate — always show):

| Step | Dataset | p_edge |
|------|---------|--------|
| 0    | librispeech | 1.20% |
| 10   | librispeech | 0.98% |

**Quality metrics table** (separate — show only if any value deviates from ideal, i.e. p_fmt < 100%, p_lang < 100%, or p_bracket > 0%):

| Step | Dataset | p_fmt | p_lang | p_bracket |
|------|---------|-------|--------|-----------|
| 0    | librispeech | 99.50% | 99.80% | 0.10% |

If all quality metrics are at their ideal values (p_fmt = 100%, p_lang = 100%, p_bracket = 0%) across all datasets, omit the quality metrics table and note: "Quality metrics nominal (p_fmt=100%, p_lang=100%, p_bracket=0% for all datasets)."

**OpenASR-ML average rows**:
- For `eval_openasr_ml`, always include the original per-dataset rows and add derived average rows grouped by language plus one overall row.
- Infer language from dataset names: `de_*` → German, `fr_*` → French, `it_*` → Italian, `es_*` → Spanish, `pt_*` → Portuguese.
- Compute average p_err as a simple arithmetic mean of `p_err` across dataset rows in that language. Display as percent with `%` suffix.
- Compute average `p_edge` as a simple arithmetic mean across dataset rows in that language.
- The overall average is a simple arithmetic mean across all active OpenASR-ML dataset rows observed for that eval step. Include only datasets present in the metrics; do not invent rows for disabled datasets.

**Show ALL steps observed — accumulate across monitoring checks to compare progression.**

### Step 4 — Training summary (when SUCCEEDED)

When the job completes, provide:
1. **Full validation metrics table** across ALL val steps (complete trajectory)
2. **Final training metrics** from the last step (for training jobs)
3. **Checkpoint save location** from the `save_checkpoint` log
4. **W&B run link** (clickable)
5. **Total training time** (from first step to last step)
6. **Trend summary**: did WER improve? By how much? Best val step?

For training jobs, Step 4 is an interim training summary. Continue to Step 4a (HF export) then Step 4b (long_eval) and do not give the final response until the automatic `long_eval_inhouse_2605_all_seg` eval completes successfully and Step 4c report generation is finished.

### Step 4a — Export checkpoint to HF safetensors format

Before submitting the post-training eval, the verl FSDP checkpoint must be converted to HF-compatible safetensors format. The `long_eval_*` job loads the model via `model.path` (an HF-format directory with `model.safetensors`, `config.json`, tokenizer files, etc.), NOT via `trainer.resume_from_path`.

#### 4a.1 Find the latest checkpoint

Derive `{TRAIN_OUTPUT_DIR}` from the training config's `trainer.default_hdfs_dir` or from the observed checkpoint path, usually:

```text
az://orngwus2cresco/data/boren/outputs/{PROJECT}/{TRAIN_CONFIG}
```

Then list and select the largest step number:

```bash
bbb ls {TRAIN_OUTPUT_DIR}/ | grep 'global_step_' | sed -E 's#.*/global_step_([0-9]+)/?#\1 #' | sort -n | tail -1
```

Prefer the best checkpoint by validation DTER if the training log clearly identifies it. Otherwise pick the latest (highest `global_step_*`).

The checkpoint is also available locally on the node at:
```text
/root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/
```

#### 4a.2 Convert FSDP shards to HF format

Run `convert_verl_to_pt.py` with `--match-lora-merged` to merge LoRA adapters into base weights:

```bash
brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 plugins/qwen35_audio/hf_model/convert_verl_to_pt.py --input /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP} --output /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf/model.pt --match-lora-merged --lora-alpha 640 --lora-rank 320"'
```

**CRITICAL**: Always use `--match-lora-merged --lora-alpha 640 --lora-rank 320`. Without `--match-lora-merged`, LoRA adapters are NOT merged into base weights and the exported model produces ~100% DTER (garbage output). The `--lora-alpha` and `--lora-rank` arguments set the correct LoRA scaling factor (alpha/rank = 2.0) used during training.

#### 4a.3 Convert PyTorch to safetensors and strip `.base_layer.` keys

The `--match-lora-merged` flag wraps LoRA-target linear layer keys with `.base_layer.` (e.g. `model.layers.0.self_attn.q_proj.base_layer.weight`). These must be stripped for HF model loading to work correctly. Convert the PyTorch file to safetensors format and strip the keys in one step:

```bash
brix ssh {NODE} -- 'bash -l -c "cd /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf && python3 -c '\''
import torch; from safetensors.torch import save_file; import os
sd = torch.load(\"model.pt\", map_location=\"cpu\", weights_only=False)
new_sd = {k.replace(\".base_layer.\", \".\"): v for k, v in sd.items()}
save_file(new_sd, \"model.safetensors\")
os.remove(\"model.pt\")
print(\"Saved\", len(new_sd), \"tensors as safetensors\")
'\'' "' 
```

**CRITICAL**: If `.base_layer.` keys are NOT stripped, HF loads the model but all LoRA-target weights are randomly initialized, producing ~100% DTER.

#### 4a.4 Copy base model config and tokenizer files

The HF directory needs config, tokenizer, and custom model code files from the base model:

```bash
brix ssh {NODE} -- 'bash -l -c "
  CKPT_DIR=/root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf
  BASE_MODEL={BASE_MODEL_PATH}
  bbb cpr \$BASE_MODEL /tmp/base_model_files
  for f in config.json tokenizer_config.json tokenizer.json merges.txt vocab.json preprocessor_config.json *.py; do
    cp /tmp/base_model_files/\$f \$CKPT_DIR/ 2>/dev/null || true
  done
  ls \$CKPT_DIR/
"'
```

Where `{BASE_MODEL_PATH}` is the `actor_rollout_ref.model.path` (or `model.path`) from the training config (e.g. `az://orngwus2cresco/.../qwen_hf/`).

#### 4a.5 Upload HF export to blob

```bash
brix ssh {NODE} -- 'bash -l -c "bbb sync /root/checkpoints/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf/ az://orngwus2cresco/data/boren/outputs/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf/"'
```

Set `{CHECKPOINT_PATH}` to `az://orngwus2cresco/data/boren/outputs/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf/` and report both the selected step and path before proceeding.

#### 4a.6 Clear blobfile cache on the node

If any prior failed attempt cached a bad model file, clear it so the eval downloads the fresh export:

```bash
brix ssh {NODE} -- 'bash -l -c "rm -rf /root/.blobfile/*/boren/outputs/{PROJECT}/{TRAIN_CONFIG}/global_step_{LATEST_STEP}/qwen_hf/"'
```

### Step 4b — Mandatory post-training in-house long-audio eval

After Step 4a produces the HF export, evaluate it with:

1. `recipe/phimm/config/eval/long_eval_inhouse_2605_all_seg.yaml` (covers all locales: en-US, da-DK, hu-HU, nb-NO, nl-NL, cs-CZ)

Run the eval as a normal remote job and monitor it through Step 3 until `SUCCEEDED` or `FAILED`. Use the same node if it is free; otherwise repeat Step 0 to find or resume a free node. Always sync code before submitting the post-training eval.

Because `submit_job.sh` does not support arbitrary hydra overrides, use a direct Ray submission for the checkpoint eval:

```bash
brix ssh {NODE} -- 'bash -l -c "cd /root/code/verl && python3 ray_tool.py prepare_env && ray job submit --working-dir=/root/code/verl --no-wait -- python3 -m recipe.phimm.main_long_eval_asr --config-name long_eval_inhouse_2605_all_seg trainer.experiment_name={TRAIN_CONFIG}_long_eval_inhouse_2605_all_seg model.path={CHECKPOINT_PATH} data.output_path=az://orngwus2cresco/data/boren/data/verl/eval/{TRAIN_CONFIG}_step{LATEST_STEP}/inhouse_2605_all_seg rollout.max_num_seqs=16 rollout.gpu_memory_utilization=0.75 nccl_timeout=3600"'
```

**Note**: The eval uses lowered settings (`max_num_seqs=16`, `gpu_memory_utilization=0.75`, `nccl_timeout=3600`) to avoid NCCL timeouts on long decode batches. The `long_eval_*` config uses top-level `rollout.*` keys (not `actor_rollout_ref.rollout.*`).

The output path uses the `inhouse_2605_all_seg` directory convention so each per-corpus slug (e.g. `enus_conv_fy21q1`, `dadk_conv_om_fy23q1`) lands directly under it as `{OUTPUT_PATH}/<slug>/measures.json` — the layout expected by `--schema all_seg`.

**Sanity check**: After the eval produces DTER numbers, verify they are in a reasonable range (10–30%). If DTER is ~100% for all corpora, the HF export is broken — go back to Step 4a and check that LoRA was merged and `.base_layer.` keys were stripped.

Track the Ray job ID. If the eval fails, diagnose and fix using Step 3f, then resubmit before continuing.

### Step 4c — Report the in-house DTER measures via the **inhouse-dter-report** skill

After the eval job succeeds, generate the canonical in-house DTER comparison report using the **inhouse-dter-report** skill. This is the required reporting mechanism for the in-house evaluation — it builds the standardized `.xlsx` report that inserts the trained model as a new column next to the fixed `Qwen3.5-audio` baseline (column A) with per-locale, overall, and WERR columns.

The long-audio eval writes two artifacts per data source under `{OUTPUT_PATH}/{DATA_SOURCE}/`:
- `result_details.jsonl` — per-recording `ref`/`hyp`, `dter`, `eer`, and `dter_detail`.
- `measures.json` — micro-averaged TER + EER for that source.

The Ray job logs also emit per-corpus `val-aux/<corpus>/dter_n_err/mean@1` and `val-aux/<corpus>/dter_n_ref/mean@1` aggregates, from which the **inhouse-dter-report** script recovers the canonical **micro-DTER** (`dter_n_err / dter_n_ref`).

Required behavior:
- Invoke the **inhouse-dter-report** skill with `--schema all_seg` (the schema matching `long_eval_inhouse_2605_all_seg`'s locales × 3 corpora — en-US, nl-NL, da-DK, hu-HU, nb-NO, cs-CZ).
- Use model label `{TRAIN_CONFIG}@step{LATEST_STEP}` unless the user provided a custom label.
- Prefer sourcing directly from the per-corpus `measures.json` blob layout with `--model <label> {OUTPUT_PATH}/` — the script auto-discovers each `<slug>/measures.json` under the eval output directory and recovers micro-DTER.
- Alternatively, source from the eval Ray job logs with `--from-ray {NODE} {EVAL_JOB_ID}` so the micro-DTER is parsed from `val-aux/<corpus>/dter_n_err|dter_n_ref/mean@1`.
- If neither is available, fall back to `--metrics <json>` built from each data source's `measures.json` (download via `bbb cp`), passing per-corpus `dter` fractions.

Build the report (preferred: directly from the eval output blob):

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
  --schema all_seg \
  --model "{TRAIN_CONFIG}@step{LATEST_STEP}" \
    az://orngwus2cresco/data/boren/data/verl/eval/{TRAIN_CONFIG}_step{LATEST_STEP}/inhouse_2605_all_seg/ \
  --out tmp/inhouse_dter_report/{TRAIN_CONFIG}_step{LATEST_STEP}_all_seg.xlsx
```

Or, sourcing from the eval Ray job logs:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/inhouse-dter-report/scripts/build_inhouse_dter_xlsx.py \
  "{TRAIN_CONFIG}@step{LATEST_STEP}" \
  --schema all_seg \
  --from-ray {NODE} {EVAL_JOB_ID} \
  --out tmp/inhouse_dter_report/{TRAIN_CONFIG}_step{LATEST_STEP}_all_seg.xlsx
```

To extend an existing report with this model as an additional column, add `--extend-xlsx <prior.xlsx>`.

Refer to the **inhouse-dter-report** skill (`.github/skills/inhouse-dter-report/SKILL.md`) for schema details, baseline values, the micro-DTER recovery formula, and additional options.

Also surface the headline measures inline from `measures.json` so the user sees the result without opening the xlsx:

```bash
bbb cp {OUTPUT_PATH}/{DATA_SOURCE}/measures.json /tmp/verl_eval/measures.json
cat /tmp/verl_eval/measures.json
```

Report the per-corpus DTER table (from the generated xlsx — all locales × 3 corpora plus per-locale and overall average rows) and the per-locale headline measures:

| Model | Locale | Dataset | TER (dter) | EER | n_recordings |
|-------|--------|---------|------------|-----|--------------|
| {TRAIN_CONFIG}@step{LATEST_STEP} | en-US | average | 14.09% | 4.20% | N |
| {TRAIN_CONFIG}@step{LATEST_STEP} | nl-NL | average | 21.46% | ... | N |
| {TRAIN_CONFIG}@step{LATEST_STEP} | da-DK | average | 23.47% | ... | N |
| {TRAIN_CONFIG}@step{LATEST_STEP} | hu-HU | average | 23.01% | ... | N |
| {TRAIN_CONFIG}@step{LATEST_STEP} | nb-NO | average | 21.19% | ... | N |
| {TRAIN_CONFIG}@step{LATEST_STEP} | overall | average | 20.64% | ... | N |

Present TER and EER as percentages (×100) with a `%` suffix. State the path to the generated `.xlsx` report.

#### 4c.2 Generate HTML chart report via the **inhouse-dter-html** skill

After the `.xlsx` report is written, always generate a self-contained HTML visualization using the **inhouse-dter-html** skill. This produces a Chart.js-powered single-file HTML with per-locale WERR bar charts, per-dataset WERR deltas, ranking tables, and takeaway bullets — making results reviewable without opening Excel.

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/inhouse-dter-html/scripts/build_inhouse_dter_html.py \
  tmp/inhouse_dter_report/{TRAIN_CONFIG}_step{LATEST_STEP}_all_seg.xlsx \
  --out tmp/inhouse_dter_report/{TRAIN_CONFIG}_step{LATEST_STEP}_all_seg.html
```

The HTML is written next to the xlsx with the same basename and `.html` extension. State the path to both the `.xlsx` and `.html` reports in the final response.

For deeper utterance-level inspection of `result_details.jsonl`, use the **inhouse-asr-compare** or **asr-word-error-analysis** skill.

Post-training final response requirements:
1. Show the training final summary from Step 4.
2. Show the in-house DTER comparison report (xlsx path + HTML path + per-locale and overall DTER/WERR across all locales) built with the **inhouse-dter-report** skill (`--schema all_seg`) and visualized with the **inhouse-dter-html** skill, plus the headline TER/EER measures for `long_eval_inhouse_2605_all_seg`.
3. Include W&B and Ray links for the training job and the eval job when available.
4. State which checkpoint path was evaluated.
5. Include the eval `data.output_path` (location of per-corpus `result_details.jsonl` + `measures.json`).
6. Include the path to the generated `.html` chart report.

#### W&B result check (optional post-job):
```bash
python ./wandb_result.py --metric val-aux search '{CONFIG}'
```
This queries W&B for the run's validation metrics and saves an Excel summary to `~/wandb_results/`.

### Step 5 — Word Error Analysis

After the job completes, perform word-level error analysis on the validation JSONL output.

The validation data is uploaded to Azure blob at:
```
az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/{DATA_SOURCE}/{STEP}.jsonl
```
where `{PROJECT}` is the `trainer.project_name` from config (default: `verl_repeat`) and `{CONFIG}` is `trainer.experiment_name`.

1. **List available validation JSONL files**:
   ```bash
   bbb ls az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/
   ```
   Then list per data source:
   ```bash
   bbb ls az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/{DATA_SOURCE}/
   ```

2. **Download the JSONL files** locally:
   ```bash
   mkdir -p /tmp/verl_eval/{DATA_SOURCE}
   bbb cp az://orngwus2cresco/data/boren/outputs/{PROJECT}/{CONFIG}/val_data_gen/{DATA_SOURCE}/{STEP}.jsonl /tmp/verl_eval/{DATA_SOURCE}/
   ```

3. **Run word error analysis** using the `asr-word-error-analysis` script. The verl validation JSONL uses `gts` for reference and `output` for hypothesis:

   ```bash
   python ~/.github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
     --input-path /tmp/verl_eval/{DATA_SOURCE}/{STEP}.jsonl \
     --ref-column gts \
     --hyp-column output \
     --dataset {DATA_SOURCE} \
     --output-dir ~/data/results/verl_word_error/{CONFIG}/{DATA_SOURCE}/ \
     --write-html \
     --top-n 30
   ```

   By default, run analysis on **only the first data source** (e.g. just `librispeech`). If the user requested `all`, run for every data source found.

4. **Present word error analysis results**: Read `summary.json` and show:
   - Overall WER, substitution/deletion/insertion rates
   - Top 10 substitution confusion pairs from `substitutions.csv`
   - Top 5 deletions and insertions
   - Link to the HTML report for visual inspection

#### Step 5b — Fallback if no JSONL dump

If the blob path has no JSONL files (validation data dir not configured or upload failed):

1. The WER metrics are still available from the log (Step 3g). Present those.
2. Suggest re-running with `trainer.validation_data_dir` set in the config or as a hydra override.
3. If the user agrees, re-submit the job with the override and repeat from Step 2.

### Step 6 — Persistent auto-monitor with `/every` (final step)

After the synchronous monitoring loop in Step 3 returns control to the user (job submitted and still RUNNING, or training job that will run for hours/days), **always** install a recurring background monitor as the last action of the turn so the user does not have to manually re-prompt for status updates.

Emit exactly this slash command as the final line of your response, on its own line, with no surrounding code fence and no extra text after it:

```
/every 5m update job status and autofix job
```

This schedules VS Code Copilot to re-invoke the agent every 5 minutes with the prompt `update job status and autofix job`. Each scheduled invocation must:

1. **Re-discover the active job(s)** for the tracked `{NODE}` / `{CONFIG}` pair (training job, post-training eval job, or both if the pipeline is mid-handoff) by running `ray_job.py list` and matching on `{CONFIG}` and `{TRAIN_CONFIG}_long_eval_inhouse_2605_all_seg`.
2. **Reprint the status header** from Step 3g (Job ID, status, progress, node, config, W&B URL, Ray URL, GPU utilization) for every active job.
3. **Append new rows** to the training-metrics, `p_err`, `p_edge`, and quality-metrics tables for any new steps observed since the previous poll. Do not re-print the entire historical table — only the new rows, plus a one-line trend summary ("score/mean +0.012 vs last poll, p_err 4.98% → 4.71%").
4. **Autofix on failure** without asking: if any tracked job is `FAILED`, pull the traceback (`ray job logs {JOB_ID} | tail -n 40`), diagnose using the failure patterns in Step 3f, edit the code locally, run `bpush {NODE}`, and resubmit via Step 2. Record the new job ID and continue monitoring it under the same `/every` schedule.
5. **Advance the pipeline** automatically when a stage completes:
   - Training `SUCCEEDED` → run Step 4a (HF export), then Step 4b (`long_eval_inhouse_2605_all_seg`).
   - Eval `SUCCEEDED` → run Step 4c (inhouse-dter-report `--schema all_seg` + inhouse-dter-html) and present the final DTER/EER summary.
6. **Stop the schedule** with `/every stop` (emitted as the final line) only after the full stack is complete: training `SUCCEEDED`, post-training eval `SUCCEEDED`, in-house DTER `.xlsx` report generated, HTML chart report generated, and the headline TER/EER table presented. Until then, every scheduled response must end with `/every 5m update job status and autofix job` to keep the loop alive.

**Rules**:
- `/every` lines must appear verbatim and must be the **very last line** of the assistant message — no trailing prose, no markdown blockquotes, no code fences around them.
- Never install `/every` for a job that has already reached a terminal state at the time of response — emit `/every stop` instead.
- The persistent loop replaces manual user pings; do not ask the user "should I keep checking?" — just schedule it.

## Quality Checks

- **Step 2**: `ray job submit` output shows a job ID. If it errors, check `ray_tool.py prepare_env` ran.
- **Step 3 (eval)**: WER values are reasonable (0–100%). If WER > 50%, flag as suspicious.
- **Step 3 (training)**: Training metrics should show learning (score/mean trending up, pg_loss decreasing). If metrics are flat or diverging, flag.
- **Step 5**: `summary.json` exists and WER matches the log metrics (within rounding).
- **Word error HTML**: `report.html` is generated and not empty.

## Important Notes

- The verl configs use hydra. The config search path is `recipe/phimm/config/` with base configs in `recipe/phimm/config/base/`.
- `quick_run.sh` determines the module automatically: configs starting with `long_eval_*` use `main_long_eval_asr`, configs starting with `gen_*` use `main_asr_gen`, configs starting with `eval_*` use `main_asr_eval`, configs starting with `remax_*` use `main_asr_remax`, and other training configs use `main_asr_dapo`.
- Eval configs inherit from `base/eval_asr.yaml` which sets `val_only: True` — only validation runs, no training.
- `long_eval_*` configs inherit from `base/long_eval_asr.yaml` (gen-style): SVAD-explode -> generate -> regroup per recording -> DisfluencyTolerant TER + entity EER, writing `result_details.jsonl` + `measures.json` under `data.output_path`. They load the model via `model.path` (HF export), not `trainer.resume_from_path`.
- Training configs inherit from `base/dapo_asr.yaml` (or `grpo_asr.yaml`, `grpo_asr_full.yaml`) — full RL training with periodic validation.
- The remote workspace is at `/root/code/verl` on Brix nodes.
- **Always push code first** before submitting jobs. Use `bpush {NODE}` or `submit_job.sh` (which pushes automatically). Never submit a job without syncing code first.
- For remote operations, use `brix ssh`, `brix scp`, or the convenience wrappers `bpush` and `submit_job.sh`.
- The word error analysis script (`analyze_word_errors.py`) supports custom column names via `--ref-column` and `--hyp-column`. verl JSONL uses `gts` and `output`.
- For long-running training jobs, use the **persistent-job-monitor** skill to poll every 5 minutes until a target step is reached.

## Command Reference

| Action | Command |
|--------|---------|
| Push code | `bpush {NODE}` |
| Submit job | `bash submit_job.sh {NODE} recipe/phimm/config/{CONFIG}.yaml false true true` |
| Job status | `brix ssh {NODE} -- 'bash -l -c "ray job status {JOB_ID}"'` |
| Job logs (tail) | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| tail -n N"'` |
| Step progress | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep \"step:\" \| tail -n 10"'` |
| Val metrics | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep -E \"val-core\|val-aux\" \| tail -n 30"'` |
| Stop job | `brix ssh {NODE} -- 'bash -l -c "ray job stop {JOB_ID}"'` |
| Check errors | `brix ssh {NODE} -- 'bash -l -c "ray job logs {JOB_ID} \| grep -E \"Traceback\|Error\" \| tail -n 20"'` |
| Fetch TER/EER measures | `bbb cp {OUTPUT_PATH}/{DATA_SOURCE}/measures.json /tmp/verl_eval/measures.json && cat /tmp/verl_eval/measures.json` |
| W&B results | `python ./wandb_result.py --metric val-aux search '{CONFIG}'` |
| List jobs | `brix ssh {NODE} -- 'bash -l -c "python /root/code/verl/ray_job.py list"'` |

## Batch Submission

For submitting multiple jobs to different nodes, edit `submit_jobs_repeat.sh` to list the jobs, or call `submit_job.sh` multiple times:
```bash
bash submit_job.sh <node1> <config1> false true true
bash submit_job.sh <node2> <config2> false true true
```

Or use `submit_jobs_repeat.sh` which wraps multiple `submit_job.sh` calls:
```bash
bash submit_jobs_repeat.sh
```

## Response Style

- **Always show** the status header with job ID, W&B URL, and Ray dashboard URL
- **Use tables** for metrics — never dump raw log lines to the user
- **Accumulate** metrics across monitoring polls — show the full step-by-step progression
- Parse `step:N - key:val - key:val` format into structured table rows
- When monitoring, report the current phase and what to expect next
- On failure, show the error, diagnose, and proceed to fix without asking
- **Do not stop monitoring** until the full stack is complete: training SUCCEEDED, `long_eval_inhouse_2605_all_seg` SUCCEEDED, and the in-house DTER `.xlsx` report (inhouse-dter-report, `--schema all_seg`) + HTML chart report (inhouse-dter-html) plus TER/EER measures summary generated
- **End every non-terminal response with `/every 5m update job status and autofix job`** on its own final line (see Step 6). Replace with `/every stop` only once the full stack is complete.

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **remote-development** | 0, 2 | Node discovery, sync, remote commands |
| **persistent-job-monitor** | 3 | Long-running training job monitoring |
| **inhouse-dter-report** | 4c | Build the canonical in-house DTER `.xlsx` comparison report (`--schema all_seg`) |
| **inhouse-dter-html** | 4c | Generate self-contained Chart.js HTML visualization from the DTER `.xlsx` report |
| **inhouse-asr-compare** | 4c | Inspect `result_details.jsonl` / TER/EER per-utterance diffs |
| **asr-word-error-analysis** | 5 | Word-level error analysis on validation JSONL |

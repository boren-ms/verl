---
name: eval-2607-benchmark-report
description: "Evaluate one or more checkpoints of the same 2607 ASR model on the standard in-house DTER, OpenASR-ML, and MixLang benchmarks remotely, create baseline-aware Excel reports, and merge reports from different checkpoint steps into one multi-sheet workbook. Use when: evaluate 2607 benchmarks, run 2607 model evaluation, build or merge 2607 benchmark Excel reports, compare checkpoints across steps, or compare a 2607 checkpoint against its reference model."
argument-hint: '<model-label> <model-path> [--node <verl-node>] [--include-digits-enus] [--include-digits-tier1] [--artifact-root <az-path>] [--out <xlsx>] [--merge-report <step>=<xlsx> ...]'
---

# 2607 Benchmark Evaluation And Multi-Checkpoint Excel Report

Evaluate one or more checkpoint steps of the same candidate ASR model on the standard 2607 benchmark suite on remote `verl-*` Brix nodes. Produce a baseline-aware `.xlsx` report for each checkpoint, then merge reports for different steps into one model-level workbook when more than one checkpoint is supplied. The merged workbook has a cross-step `summary` sheet and separately named benchmark sheets for every checkpoint. Preserve each evaluation's W&B metadata, key metric logs, and detailed decoding results under a durable remote blob path.

Delegate remote submission, monitoring, failure recovery, and result retrieval to the `verl-asr-run` skill. This skill owns benchmark selection, baseline discipline, and workbook assembly.

## Inputs

| Input | Required | Meaning |
|---|---:|---|
| `model-label` | Yes | For one checkpoint, the workbook column label such as `remax_2607@step560`; for a merge, the shared model/run label such as `remax_2607`. |
| `model-path` | Yes | Candidate HF checkpoint path, normally an `az://.../qwen_hf/` directory. |
| `--node` | No | Ready remote `verl-n<N>-*` node. Let `verl-asr-run` select one when omitted; evaluations use that running pool's verified `N` as `trainer.nnodes`. |
| `--include-digits-enus` | No | Add the optional en-US digits benchmark. |
| `--include-digits-tier1` | No | Add the optional Tier 1 digits benchmark. |
| `--out` | No | Final workbook path. Default: `tmp/eval_2607_reports/<model-label>.xlsx`. |
| `--artifact-root` | No | Durable `az://.../` root for raw evaluation artifacts. Default: the common remote evaluation output root for `<model-label>`; it must not be node-local storage. |
| `--merge-report <step>=<xlsx>` | No | Existing single-checkpoint report to merge. Repeat once per checkpoint step, in desired sheet order. All reports must belong to the same model family and use the same benchmark/config schema. |

## Benchmark Contract

Run every required config with the candidate model path as a Hydra override. Each config's committed reference model is the baseline for its sheet; do not substitute a global or fixed baseline from another report.

| Workbook sheet | Config | Metric | Reference source |
|---|---|---|---|
| `inhouse_dter` | `recipe/phimm/config/eval/long_eval_inhouse_2607_all_seg30.yaml` | micro-DTER | `model.path` in the config and its reference result directory. |
| `digits_enus` | `recipe/phimm/config/eval/eval_digits_enus_2607.yaml` | Digit CER and WER | Only include when `--include-digits-enus`; baseline is `actor_rollout_ref.model.path`. |
| `openasr_ml` | `recipe/phimm/config/eval/eval_openasr_ml_verb_2607.yaml` | WER / `p_err` | `actor_rollout_ref.model.path` in the config. |
| `mixlang` | `recipe/phimm/config/eval/long_eval_mixlang_fy26q2_zh_seg_2607.yaml` | DTER / TER | `model.path` in the config and its reference result directory. |
| `digits_tier1` | `recipe/phimm/config/eval/eval_digits_tier1_2607.yaml` | Digit CER and WER | Only include when `--include-digits-tier1`; baseline is `actor_rollout_ref.model.path`. |

The reference checkpoint and candidate must use the same config, data, locale, scoring backend, and segment settings. In particular, do not compare MixLang's zh-CN TER to an unrelated DTER result.

## Procedure

1. Resolve the exact committed config files and record the reference model path from each file before submitting jobs. Confirm the candidate path is an HF-loadable model directory.
2. Select a free Ready `verl-n<N>-*` node using `verl-asr-run`'s occupancy checks. Derive `EVAL_NNODES=N` from the selected running pool name (for example, `verl-n2-i3` means `EVAL_NNODES=2`) and confirm that count against the healthy nodes reported by `ray status` on the pool. Do not copy `trainer.nnodes` from the config or assume it is `1`; if the name and live Ray count disagree, resolve the pool health/count before submission. Push the current workspace with `rcall-brix sync <node>` before submitting work.
3. For each required config, run a candidate evaluation remotely through `verl-asr-run`, using a unique experiment name and output path below `--artifact-root`. Always pass `trainer.nnodes=${EVAL_NNODES}` together with the candidate model path and output/experiment overrides. Configure the evaluation so detailed decoding outputs are written or uploaded to that remote blob path, never left only on the Brix node.
4. Use the embedded 2607v1 baseline for in-house DTER, OpenASR-ML, and MixLang unless an explicit matching baseline source is supplied. Run the same config for its reference only when a benchmark has no embedded baseline (such as optional digits) or the caller explicitly requests a refreshed reference. Any reference evaluation must run on its selected pool with that pool's independently derived and verified `trainer.nnodes`.
5. Monitor every Ray job to `SUCCEEDED`. On failure, use `verl-asr-run` to diagnose and repair the root cause, then rerun only the failed benchmark. Do not start workbook creation from incomplete or failed output.
6. Before releasing a node or creating the workbook, retain the raw information for every candidate run and every explicitly executed reference run under `<artifact-root>/<benchmark>/<candidate|reference>/`:
   - `wandb.json`: W&B entity, project, run ID, run name, and run URL. If W&B is disabled or unavailable, record the reason explicitly instead of inventing a run identity.
   - `key_metrics.log` or `key_metrics.json`: the unmodified evaluation metric records used to build the report, including the final `val-aux/...` values or equivalent scorer output.
   - Detailed decoding results: upload all per-utterance hypothesis/reference outputs, such as `result_details*.jsonl`, together with scorer outputs such as `measures.json`. Preserve the dataset subdirectory structure when multiple datasets are evaluated.
   - `artifact_manifest.json`: benchmark, role, config path and revision, model path, Ray node and job ID, verified `trainer.nnodes`, W&B metadata path, key-metrics path, detailed-decoding remote paths, and completion timestamp.

   Confirm every recorded `az://` object or prefix exists and is readable. A W&B URL alone is not a substitute for the raw key-metrics log, and a Ray job log or node-local path is not a durable detailed-decoding result.
7. Extract metrics from the final candidate and reference outputs:
   - `inhouse_dter`: calculate micro-DTER as $\sum edits / \sum reference\ tokens$ per corpus; do not use segment-macro DTER. Compute `overall avg` as the arithmetic mean of the displayed corpus micro-DTER values so it matches the embedded baseline convention; do not pool counts across corpora.
   - Optional `digits_enus` and `digits_tier1`: preserve both Digit CER and WER, with every evaluated dataset as a row.
   - `openasr_ml`: use final `p_err`/WER per dataset, language averages, and an overall average.
   - `mixlang`: use the final reference-compatible DTER/TER from `measures.json`, retaining the configured zh-CN scoring context.
8. Build one single-checkpoint workbook per evaluated step with [scripts/build_2607_report.py](./scripts/build_2607_report.py). It reuses existing eval outputs and produces sheets in this order: `summary`, `inhouse_dter`, optional `digits_enus`, `openasr_ml`, `mixlang` (its own separate sheet), then optional `digits_tier1`. Keep these intermediate workbooks; they are the inputs to the multi-checkpoint merge. Each benchmark sheet holds a reference column (`A`), a candidate column (`B`), and a delta column `A->B` computed exactly like `inhouse-dter-report`:

   $$\mathrm{delta}=1-\frac{\mathrm{candidate\ error}}{\mathrm{reference\ error}}$$

   Every group and overall `avg` value cell uses an Excel `AVERAGE(...)` formula over the displayed dataset cells for that metric; no fixed, weighted, pooled-count, or stored-numeric override is allowed. Overall formulas reference only dataset ranges and exclude intermediate group-average rows. The delta column is labelled per metric — `TERR` for DTER/TER, `WERR` for WER, `CERR` for CER. Positive delta means the candidate improved. Data and average-row delta cells use the Excel formula `=1-C{row}/B{row}`. All error and delta cells use percentage formatting, and the delta column carries a red→white→green color scale centered at 0.
9. The script also emits the `summary` sheet with one row per included benchmark metric: the aggregate baseline and candidate values, the overall delta, and the source config. Digits rows appear only for the explicitly requested digits evaluations.
10. When two or more checkpoints of the same model are available, merge their completed single-checkpoint workbooks with [scripts/merge_2607_reports.py](./scripts/merge_2607_reports.py). Pass each report as `--report <checkpoint-label>=<path>` in ascending step order unless the caller requests another order. The merged workbook contains:
   - `summary`: one row per checkpoint, benchmark, and metric, with the checkpoint label, baseline, candidate, delta, config, model paths, result sources, artifact sidecars, and source workbook.
   - `<checkpoint>_<benchmark>`: a faithful copy of each benchmark sheet, retaining formulas, percentage formatting, conditional formatting, widths, and provenance values. Sheet names are sanitized for Excel, limited to 31 characters, and given a stable hash suffix only when truncation causes a collision.

   Do not copy the per-checkpoint `summary` sheets verbatim; consolidate them into the single cross-step `summary`. Reject the merge if checkpoint labels repeat or if the reports differ in included benchmark/metric/config schema. Do not silently combine reports from different model families, baseline revisions, optional benchmark sets, or config revisions.
11. Preserve baseline and raw-artifact provenance: keep the config path, reference model path, candidate path, metric definition, W&B run URL/ID, key-metrics remote path, detailed-decoding remote paths, artifact-manifest remote path, and source workbook path with the merged workbook. When the workbook format cannot hold all paths cleanly, write a sidecar `<workbook-stem>.artifacts.json` per checkpoint and retain its local and remote paths in the merged summary.

### Consolidated report script

Each benchmark takes a candidate source and an optional baseline source. A source is auto-detected as one of: an `az://.../` root or local directory holding `<slug>/measures.json` (used for the DTER benchmarks `inhouse_dter` and `mixlang`); a `*.json` metrics file (`{ds: value}` or `{ds: {metric: value}}`); a text log with `val-aux/<ds>/<metric>/mean@1` lines; or `ray:<node>:<job_id>` to pull `ray job logs`.

**Collecting the baseline result.** The baseline (column `A` in the report schema) uses the current embedded 2607v1 values when `--<bench>-baseline` is omitted:

| Benchmark | Embedded label | Source column | Embedded overall |
|---|---|---:|---:|
| `inhouse_dter` | `2607v1,LID` | C | 17.77% |
| `openasr_ml` | `2607v1` | B | 2.92% |
| `mixlang` | `2607v1` | B | 21.24% |
| `digits_enus` / `digits_tier1` | none | — | Supply `ray:<node>:<job_id>`, log, or JSON via `--<bench>-baseline`. |

An explicit `--<bench>-baseline` source overrides the embedded values. For optional digits, collect the baseline by running the same config with the reference model through `verl-asr-run` and passing its Ray job as the baseline source (or a captured log/JSON). Collect candidate results with the candidate model path.

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/eval-2607-benchmark-report/scripts/build_2607_report.py \
   --label "cand@step560" \
  --inhouse-dter  az://orngwus2cresco/.../cand/inhouse_2605_all_seg30/ \
  --mixlang           az://orngwus2cresco/.../cand/long_audio_mixlang_fy26q2_zh_seg/ \
  --openasr-ml           ray:verl-n1-i0:raysubmit_CAND2 \
  --out tmp/eval_2607_reports/cand_step560.xlsx
```

The `inhouse_dter`, `openasr_ml`, and `mixlang` baselines come from the embedded table above, so their `--*-baseline` flags may be omitted. Add `--digits-enus <cand>` with `--digits-enus-baseline <base>` only when `--include-digits-enus` was requested. Add `--digits-tier1 <cand>` with `--digits-tier1-baseline <base>` only when `--include-digits-tier1` was requested. Only benchmarks whose candidate source is supplied get a sheet.

### Merge reports across checkpoint steps

Use the merge helper only after every source workbook passes the single-checkpoint quality gates. The repeated `--report` arguments define checkpoint and sheet order.

```bash
/home/boren/.virtualenvs/openai/bin/python \
   .github/skills/eval-2607-benchmark-report/scripts/merge_2607_reports.py \
   --model-label "remax_2607" \
   --report "step280=tmp/eval_2607_reports/remax_2607_step280.xlsx" \
   --report "step560=tmp/eval_2607_reports/remax_2607_step560.xlsx" \
   --report "step840=tmp/eval_2607_reports/remax_2607_step840.xlsx" \
   --out tmp/eval_2607_reports/remax_2607_all_steps.xlsx
```

Do not merge by placing multiple candidate columns into the same benchmark sheet. Keep each checkpoint's benchmark sheet self-contained so its baseline, formulas, and provenance remain auditable. Use the merged `summary` sheet for cross-step sorting and comparison.

## Quality Gates

Before delivering the workbook, verify all of the following:

- Every required benchmark has a successful candidate job and either the matching embedded 2607v1 baseline or an explicitly supplied matching reference result.
- Every submitted candidate and reference evaluation explicitly used `trainer.nnodes` equal to the verified node count of its running `verl-n<N>-*` pool.
- `digits_enus` and `digits_tier1` are absent unless explicitly requested.
- Baselines are the embedded 2607v1 values or verified matching reference outputs supplied explicitly.
- Candidate and baseline use identical data/locale/scoring parameters for each sheet.
- Every executed evaluation has a readable remote `artifact_manifest.json`, raw key-metrics log, and detailed decoding result path; none points only to node-local storage.
- Every executed evaluation records its W&B run ID and URL, or an explicit reason W&B was unavailable, and the W&B identity agrees with the experiment represented in the workbook.
- In-house corpus values are micro-DTER and its `overall avg` is the arithmetic corpus average; OpenASR-ML includes language and overall averages; any included digits sheet contains CER and WER; MixLang records its zh-CN scoring backend.
- Every sheet has non-empty baseline and candidate values, percentage formatting, and a correctly signed delta (`1 - B/A`).
- Every single-checkpoint workbook opens successfully and contains exactly the expected sheet set, with `mixlang` on its own separate sheet, and its provenance references the remote raw-artifact paths.
- A merged workbook contains exactly one `summary` sheet plus one prefixed benchmark sheet per checkpoint and included benchmark; it contains no copied per-checkpoint summary sheets.
- All merged reports have unique checkpoint labels and identical benchmark, metric, config, baseline revision, and optional-digits coverage. Their candidate model paths identify different steps of the same model family.
- The merged `summary` has a non-empty row for every checkpoint/benchmark/metric combination and retains both artifact sidecar paths and the source workbook path.
- Reopen the merged workbook with `openpyxl`, confirm its expected sheet names and row counts, and verify at least one copied delta cell still contains an Excel formula rather than a cached value.

## Existing Helpers

Use existing report skills as parsers/formatting references only after checking their embedded baseline matches the current config reference:

- `inhouse-dter-report` for the all-segment micro-DTER schema.
- `digits-report` for digits CER/WER presentation.
- `openasr-report` for OpenASR-ML dataset and language-average presentation.

Those helpers may contain fixed historical baselines. Their baseline must be overridden or the metrics assembled directly when it differs from the reference model declared in the 2607 config.
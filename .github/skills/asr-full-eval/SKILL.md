---
name: asr-full-eval
description: "Run a complete ASR model evaluation pipeline: launch remote eval on Brix nodes, fetch experiment reports with WER/EER metrics, compare target model against baseline on every dataset, and perform deep entity error analysis on regression datasets. Use when evaluating a new ASR model end-to-end, comparing fine-tuned checkpoints against baseline, investigating entity recognition regressions, or producing a full evaluation report with detail comparisons and entity diagnostics. Orchestrates asr-remote-eval, asr-experiment-report, asr-detail-compare, and asr-entity-error-analysis. Inputs: run name(s), optional eval config and baseline model (e.g. 'full eval en_ht_v1_egs_cer02_err10 against baseline on entity_raw.yaml')."
---

# ASR Full Eval

End-to-end ASR evaluation pipeline: launch remote evaluation, fetch results, compare against baseline, and deep-dive into entity errors on regression datasets.

## Inputs

| Parameter | Required | Default | Example |
|-----------|----------|---------|---------|
| **run_name(s)** | Yes | — | `en_ht_v1_egs_cer02_err10` |
| **eval_config** | No | `entity_raw.yaml` | `openasr_entity.yaml` |
| **seed** | No | `4o-mini-asr-v1` | `4o-mini-asr` |
| **baseline_model** | No | `baseline` | `en_ht_v1_egs_step_5394` |
| **results_root** | No | `az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1/` | custom root |
| **regression_threshold** | No | `0.0` (any WER increase) | `0.5` (only ≥0.5% abs WER increase) |

## Variable Conventions

| Variable | Derivation | Example |
|----------|-----------|---------|
| `{RUN}` | User-provided run name | `en_ht_v1_egs_cer02_err10` |
| `{STEP}` | Latest checkpoint step from Phase 1 | `5394` |
| `{RUN}_step_{STEP}` | Full model name in results root | `en_ht_v1_egs_cer02_err10_step_5394` |
| `{BASELINE}` | Baseline model name (default: `baseline`) | `baseline` |
| `{DATASET}` | Full dataset path with group prefix | `en-US-entity-v3/Insurance` |
| `{DATASET_FLAT}` | `{DATASET}` with `/` replaced by `_` | `en-US-entity-v3_Insurance` |
| `{RESULTS_ROOT}` | Azure results root path | `az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1/` |

## Procedure

### Phase 1 — Launch Remote Evaluation (invoke `asr-remote-eval`)

Use the **asr-remote-eval** skill to run the model on a remote Brix node.
Pass `eval_config` as the eval config YAML and `seed` as the seed model.

1. Discover Ready `bus-eval-prod*` nodes, check occupancy, assign nodes.
2. Discover checkpoint steps; select the latest (or user-specified) step per run.
3. Generate eval bash scripts and launch on remote nodes in tmux.
4. Monitor every 10 minutes until all evaluations finish (auto-retry on crash).

**Wait until all evaluations complete** before proceeding.

### Phase 2 — Fetch Results & Generate Reports (invoke `asr-experiment-report`)

Use the **asr-experiment-report** skill to collect and organize results.

1. Run `report_summary.py` for each model → summary Excel + Markdown.
2. Run per-dataset word-error analysis (`asr-word-error-analysis` script, `--write-html`).
3. Optionally reshape Excel with `excel-metric-analysis`.
4. Extract the **dataset list** and **per-dataset WER/EER** from the summary Markdown
   (or from `bbb ls {RESULTS_ROOT}/{RUN}_step_{STEP}/` discovery).
   Store as a list of `(dataset, wer, eer)` tuples — this feeds Phase 3.

### Phase 3 — Compare Against Baseline (invoke `asr-detail-compare`)

Compare the target model against the baseline on **every evaluated dataset**.
Datasets are independent — run comparisons in parallel (batch all datasets in one pass).

1. For each dataset, run `asr-detail-compare`:
   ```bash
   python .github/skills/asr-detail-compare/scripts/compare_result_details.py \
     --baseline-model {BASELINE} \
     --target-model {RUN}_step_{STEP} \
     --dataset {DATASET} \
     --results-root {RESULTS_ROOT} \
     --output-dir ~/data/results/full_eval/{RUN}/{DATASET_FLAT}/ \
     --write-html --join-columns audio_file --top-n 30
   ```
2. Collect `*.summary.json` from each comparison. Build a **baseline vs target table**:

   | Dataset | Baseline WER | Target WER | ΔWER | Status |
   |---------|-------------|------------|------|--------|
   | openasr/ami | 16.2% | 15.8% | -0.4% | ✅ improved |
   | entity-v3/Insurance | 8.1% | 9.3% | +1.2% | ❌ regressed |

3. **Identify regression datasets**: target WER > baseline WER by more than `regression_threshold`.
4. Present the full comparison table; highlight regressions prominently.

### Phase 4 — Deep-Dive on Regressions

**If no datasets regressed**, skip Phase 4 entirely and proceed to Phase 5.

**For each regression dataset from Phase 3**, choose the appropriate analysis:

- **`en-US-entity-v3/*` datasets** (have `<NE>` tags) → invoke `asr-entity-error-analysis` (steps 1–4 below).
- **`openasr/*` or other non-entity datasets** → invoke `asr-word-error-analysis` with `--write-html` on the target model to surface the top substitution/deletion/insertion patterns driving the WER increase. Compare against the Phase 3 detail-compare HTML for utterance-level context.

For each regression `en-US-entity-v3/*` dataset:

1. Run entity error analysis on **both** the target and the baseline:
   ```bash
   # Run for BOTH models: {RUN}_step_{STEP} and {BASELINE}
   for MODEL in "{RUN}_step_{STEP}" "{BASELINE}"; do
     SUFFIX=$([ "$MODEL" = "{BASELINE}" ] && echo "_{BASELINE}" || echo "")
     python .github/skills/asr-entity-error-analysis/scripts/analyze_entity_errors.py \
       --model "$MODEL" --dataset {DATASET} \
       --results-root {RESULTS_ROOT} \
       --output-dir ~/data/results/full_eval/{RUN}/entity_analysis/{DATASET_FLAT}${SUFFIX}/ \
       --write-html --top-entities 30
   done
   ```
2. Compare entity results side-by-side:
   - Both `summary.json` — compare overall EER, substitution/deletion/insertion rates.
   - Both `entity_substitutions.csv` — find new substitution patterns in target.
   - Both `entity_type_breakdown.csv` — compare per-entity-type error rates.
3. Produce a **regression entity diagnosis** per dataset:

   | Metric | Baseline | Target | Δ |
   |--------|----------|--------|---|
   | EER | 12.3% | 15.1% | +2.8% |
   | Entity Substitutions | 45 | 62 | +17 |
   | Entity Deletions | 12 | 18 | +6 |
   | Top new error | — | "Goldman Sachs" → "Goldman sacks" | — |

### Phase 5 — Final Summary Report

Compile all phases into a unified report:

1. **Overall summary table**: all datasets with baseline WER, target WER, ΔWER, baseline EER, target EER, ΔEER.
2. **Regression highlights**: regression datasets with root-cause entity errors identified in Phase 4.
3. **Improvement highlights**: datasets where the target improved over baseline.
4. **Artifact index**: paths to all generated HTML reports (detail compare, word error, entity error).
5. Present with clear recommendations on model quality.

## Output Artifacts

| Phase | Artifact | Location |
|-------|----------|----------|
| 1 | Remote eval logs | `/tmp/eval_{RUN_SHORT}.log` on Brix node |
| 2 | Summary Excel + Markdown | `~/data/results/report/{RUN}_{ts}.xlsx` |
| 2 | Per-dataset word error HTML | `~/data/results/word_error_analysis/...` |
| 3 | Detail compare HTML + CSV | `~/data/results/full_eval/{RUN}/{DATASET_FLAT}/` |
| 4 | Entity error HTML + CSV | `~/data/results/full_eval/{RUN}/entity_analysis/{DATASET_FLAT}/` |
| 5 | Final summary table | Presented inline |

## Quality Checks

- **Phase 1**: All eval sessions finish without crash (retry up to 3×).
- **Phase 2**: `report.html` count matches total dataset count per experiment.
- **Phase 3**: `summary.json` exists for every dataset comparison. Spot-check `ref` matches between baseline and target.
- **Phase 4**: Entity analysis ran for all `en-US-entity-v3/*` regressions; word-error analysis ran for non-entity regressions. If no regressions, Phase 4 was skipped.
- **Phase 5**: ΔWER values in final table match individual `summary.json` files.

## Dependent Skills

| Skill | Phase | Purpose |
|-------|-------|---------|
| **asr-remote-eval** | 1 | Remote evaluation on Brix nodes |
| **asr-experiment-report** | 2 | Result fetching and report generation |
| **asr-detail-compare** | 3 | Utterance-level baseline vs target comparison |
| **asr-entity-error-analysis** | 4 | Entity-level error diagnosis on regressions |
| **asr-word-error-analysis** | 2, 4 | Per-dataset word error analysis (via asr-experiment-report in Phase 2; directly on non-entity regressions in Phase 4) |

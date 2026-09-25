---
name: suggest-next-asr-recipes
description: 'Suggest and create the next ASR training candidate recipes from running and completed W&B experiments in a project, always using validation-dataset p_err as the quality objective. Use when: suggest next recipe or receipe, propose next experiments, decide what to train next, analyze a project experiment sweep, or generate evidence-backed candidate YAMLs with a clear recommendation report.'
argument-hint: '<project-or-wandb-url> [recipe directory] [candidate count] [target validation datasets]'
---

# Suggest Next ASR Recipes

Turn a project's running and completed experiments into a small, prioritized set
of testable next training recipes. **Always base quality judgments and candidate
selection on validation-dataset `p_err`, lower is better.** Produce a clear report
and actual, well-named YAML files; do not stop at suggested override strings.

This skill analyzes and creates configs only. Do not submit jobs, change running
experiments, update the workload cache, sync code, commit, or create monitors.
Invoke `verl-asr-run` separately only if the user explicitly asks to launch.

## Inputs and Defaults

| Input | Default / resolution |
|---|---|
| Project | Required, or extract it from the supplied W&B project URL. |
| Recipe directory | `recipe/phimm/config/<project>`; verify it exists rather than creating a guessed project. |
| W&B location | Use the supplied URL's host/entity/project, discarding workspace query parameters. Otherwise use `https://msaip.wandb.io`, entity `genai`, and the project. |
| Candidate count | Up to 3; create fewer when evidence is weak or ongoing runs already cover the useful hypotheses. |
| Target validation datasets | User-specified set, otherwise all datasets in the relevant composed project validation config. Freeze this set before ranking. |
| Dataset weights | Equal-weight arithmetic mean; use different weights only if explicitly supplied by the user, documenting them before ranking. |
| Output directory | `tmp/recipe_suggestions/<project>/<UTC timestamp>/` for the report and evidence; candidate YAMLs go in the recipe directory. |

Example:

```text
/suggest-next-asr-recipes v2607_earning
  recipes: recipe/phimm/config/v2607_earning
  wandb: https://msaip.wandb.io/genai/v2607_earning?nw=nwuserboren
  candidates: 3
```

## 1. Inventory Recipes and Experiments

1. Read the worktree status and preserve existing modifications. Inspect all
   project recipes and their inherited Hydra defaults, train/validation manifests,
   reward/scoring configuration, and relevant trainer code. Distinguish training
   recipes from `base`, `eval_*`, `gen_*`, and `val_only` configurations.
2. Compose configs; filenames alone do not establish their effective settings.
   Record model initialization, train-data mixture, algorithm, reward parameters,
   learning rate/schedule, batch sizes, rollout count/temperature, LoRA settings,
   seed, training budget, validation cadence, and resources.
3. Use W&B's read-only API to enumerate the project, including running, finished,
   failed, crashed, killed, and pending runs. Include finished and running
   training experiments in the comparison. Keep failed/interrupted and unstarted
   experiments in the coverage table, but not in the main winner ranking.
   Eval-only runs may establish a reference, not evidence that a training recipe
   improved. Mark stale running runs with their last observed activity.
4. Map runs to recipes using logged configuration, `trainer.experiment_name`,
   run metadata, and code revision, not display-name similarity alone. Use run ID
   as the identity: duplicate names, retries, resumes, and seeds are distinct.
   Compare logged effective configs with local composition and list runtime
   overrides or code/config drift. The current file is not proof of what an old
   run used. Unmapped or unreconstructable runs cannot justify a precise ablation.
5. Fetch full, unsampled validation history as described in
   [evidence and config validation](references/evidence-and-validation.md).
   Do not rank using W&B summary values, chart screenshots, sampled `history()`,
   or `wandb_result.py` alone; these cannot establish best-step and trend evidence.
6. Save `evidence.json` with snapshot start/end UTC, Git revision and dirty state,
   source URL, run IDs/URLs/states, relevant sanitized config fields, recipe
   mappings, validation history, exclusions, and the comparison contract.
   Record collection gaps explicitly. Never persist credentials, signed URL
   tokens, or unfiltered W&B configs.

W&B unavailability is not evidence of no experiments. Try an existing authenticated
session or a user-provided export/log with the same provenance and validation
metrics. Identify offline/stale evidence as such. If usable evidence is unavailable,
write a blocked report with the exact access/data problem; do not invent metrics
or create supposedly evidence-backed candidates.

## 2. Establish the Validation `p_err` Contract

### Exact metric and dataset identity

- Accept exact `val-core/<data_source>/p_err/mean@1` or
  `val-aux/<data_source>/p_err/mean@1` keys. Discover which namespace each run
  actually logged. Record original keys; collapse aliases only when their values
  agree at the same step. Conflicting aliases invalidate that observation.
- Do not substitute training reward, training error, loss, `score`, `p_edge`,
  `pb_err`, `dter_p_err`, CER, or `best@N`/`mean@N` for `p_err/mean@1`.
  Training diagnostics may explain instability or cost, never choose the winner.
  If exact validation `p_err` is absent, mark the run unscored.
- Verify matching validation manifest/version, examples or subset, reference
  style, preprocessing, chunk/parent aggregation, scorer and code version,
  decoding settings, and metric definition. Group incompatible runs into separate
  cohorts; a shared `data_source` name does not prove comparability. If a historical
  scoring change cannot be ruled out, mark comparability unverified.
- Keep lexical and verbatim datasets separate. For the current `v2607_earning`
  base, inspect `data/val_data/earnings_aa_chunked.yaml`: it supplies
  `earning_chunk_verb` and `earning_chunk_lex`, scored with `long_audio_grouped`.
  Recheck these defaults on every invocation. Do not mix their parent-level
  scoring with another chunk-level validation protocol.
- Default to the complete configured target set, not the intersection that makes
  a run look good. Runs with different complete sets may be reported separately.
  An explicitly requested subset must be fixed and disclosed before ranking.

### Score and history rules

For a run at training step `s`, with the fixed target set `D`:

```text
A(s) = sum(p_err[d, s] for d in D) / len(D)
```

For user-specified positive weights, use `sum(w[d] * p_err[d, s]) / sum(w[d])`.
Every required dataset must have a finite, nonnegative value at the **same actual
training step**. Zero is valid; values above 1 are possible. Never fill missing
values with zero, forward-fill across steps, interpolate, or average each
dataset's independently best checkpoint.

Resolve training steps from logged `step` or the repository's explicit W&B
logging step (`_step`); check consistency when both exist. Do not confuse
`timing_s/step` or sample-count axes with the optimizer step. Merge sparse rows
only for the same run/step with nonconflicting values; expose conflicts and
nonmonotonic/restarted histories rather than silently stitching attempts.

For every scored run report:

- Initial validation, if present, with its real step. A resumed run's first
  validation is not necessarily an untrained step-0 baseline.
- Latest **complete** validation vector and aggregate, plus any later incomplete
  observation (do not present an older complete row as up-to-date).
- Best complete trained step by minimum `A(s)`, breaking exact ties with the
  later step. Show step-0 separately; if training never beats it, say so.
- The last three complete validation points, or all available points when fewer,
  to distinguish stable improvement, noise, flattening, and regression.
- Matched-step/budget comparisons between candidates and their controls.
  Best-so-far from a long run and early results from a short run are not a fair
  final ranking. Do not interpolate nonexistent matched-step measurements.
- Per-dataset regressions and spread across repeated seeds/attempts where
  available. Do not claim statistical significance from one run.

Display `100 * p_err` with `%`. Show changes as percentage points
`100 * (candidate - reference)`; negative is better. Optional relative error
reduction is `1 - candidate / reference` (undefined for a zero reference).
Keep raw ratios and exact steps in the evidence artifact.

Running runs are **provisional**, never completed wins or failures. A missing or
early validation is not evidence that a setting is bad. Check ongoing experiments
before proposing duplicates; recommend waiting when they will answer the same
question. Best observed metrics do not prove a checkpoint was saved.

## 3. Select the Next Hypotheses

1. Select the best supported parent/control within the relevant comparable cohort
   using validation `p_err`, considering both matched-budget results and stability.
   If it is still running, label the choice provisional. Preserve the original
   initialization by default, not an unverified best-step checkpoint.
2. Compare effective parameter differences between controls and neighbors.
   Separate observations from causal hypotheses: changes in data, LR, rollout
   count, and LoRA rank together do not establish which change helped.
3. Prefer a small interpretable batch, as evidence permits:
   - One conservative refinement of a promising, stable setting.
   - One controlled ablation or replication to resolve a specific uncertainty.
   - One bounded exploration motivated by a validation failure or plateau.
   These are roles, not a mandatory grid. No arbitrary parameter sweep is needed.
4. Change one scientific factor per candidate where possible. If coupled changes
   are necessary for validity or resources, explain each and flag the confound.
   Training reward changes are allowed hypotheses, but only validation `p_err`
   can support their benefit. Never train on held-out validation examples or
   alter validation to manufacture a lower score.
5. Check each proposal against all completed/running effective configs and
   existing local proposals, ignoring naming/logging/output-path differences.
   Do not create equivalent experiments under new names. Intentional replication
   requires an explicit seed/repeat distinction and uncertainty-reduction goal.
6. Give every candidate an evidence chain: cited run IDs and steps, per-dataset
   `p_err` observation, limitation, hypothesis, exact config delta, why this test
   is next, resource implications, and a measurable validation-only success rule.
   State expected direction, not a fabricated predicted improvement.

A useful default success rule is lower aggregate `p_err` than the control at
the same validation steps/budget, with no target-dataset regression; require
confirmation at consecutive validations where the budget permits. Record any
user-approved per-dataset tolerance explicitly. Reject or clearly defer trade-off
candidates rather than silently changing the objective. With insufficient evidence
for a setting, state that and propose only tests justified by the available
validation history, or produce no candidates.

## 4. Create Clear, Runnable YAMLs

1. Copy the appropriate existing leaf recipe into a new file in the same project
   directory, preserving the Hydra search path/defaults and established style.
   Reconcile the parent's logged overrides before applying the deliberate delta.
   Do not mutate a shared base or overwrite existing/user-modified recipes.
2. Preserve the algorithm prefix used by `quick_run.sh` (`remax_`, `grpo_`, etc.).
   Use readable tokens describing the model/data family and actual distinguishing
   settings; prefer explicit LR values over ambiguous `lr2`, `new`, `best`, or
   `candidate1`. Suggested structure:

   ```text
   <algorithm>_<model>_<data-style>_s<steps>_bs<batch>_n<rollouts>_r<rank>_g<gpus>_<changed-setting>.yaml
   ```

   Example **naming only, not an evidence-backed recommendation**:
   `remax_2607v1a_earning_batches_tts_s1k_bs256_n2_r256_g16_lr5e-6.yaml`.
   Here `g16` means total GPUs (`nnodes * n_gpus_per_node`), not a batch size;
   `lr5e-6` must resolve to `actor_rollout_ref.actor.optim.lr: 5e-6`.
   For a non-LR hypothesis, append a meaningful factor/value such as `kl5e-4`.
   Retain an explicit LR token when it distinguishes the parent family.
3. Check both filesystem and W&B names for collisions. If the config is already
   represented, reuse/link it rather than overwrite it; choose a meaningful new
   suffix only for a genuinely distinct experiment or documented replication.
4. Do not add `trainer.project_name` or `trainer.experiment_name` to candidate
   recipes. Keep the project inherited from the project base; `quick_run.sh` sets
   the experiment name from the filename at launch. Verify output/checkpoint and
   validation-generation roots after applying that launch override so they
   resolve to the new experiment, not the parent. Preserve the repository's
   resume policy with fresh output roots; never resume the parent's state
   accidentally.
5. Preserve the comparison contract: validation data, scorer, decoding, and
   cadence/budget unless an explicitly justified hypothesis requires a budget
   change. Resource changes must remain valid for global/mini/micro-batches,
   rollout count, LoRA, sequence lengths, and available topology.
6. Compose and resolve every YAML with the lightweight Hydra procedure in
   [evidence and config validation](references/evidence-and-validation.md).
   Diff the **resolved** candidate against the reconstructed parent, not just the
   YAML text. Allow only the documented scientific delta and necessary runtime
   experiment/output identity changes. Correct accidental inherited changes. Do
   not claim GPU execution was tested by a config-only check.

## 5. Report and Verify

Write `report.md` using the [report template](references/report-template.md).
Include a concise recommendation, a complete run inventory, comparable
validation evidence, and a plain-language explanation for **each** created YAML.
Link every recipe, parent, run, and evidence artifact. Keep excluded/incomplete
runs visible and explain why they did not influence selection.

Before completing:

- Every quality conclusion cites validation `p_err`, exact datasets, runs, and steps.
- Every proposed experiment has a distinct, evidence-backed hypothesis and
  reports running-run uncertainty rather than forecasting a final winner.
- Every listed new YAML exists, composes, resolves, matches its name, and has only
  the intended delta; artifact paths are unique and source recipes untouched.
- The report contains the exact metric/set/weights/cohort contract, per-dataset
  results, limitations, candidate priority, and a measurable success criterion.
- `evidence.json` and the report exist and contain no credentials.
- Return links to the report and YAMLs, the top rationale, and validation performed.
  State explicitly that nothing was launched. If blocked, return the blocked
  report instead of claiming candidate creation succeeded.

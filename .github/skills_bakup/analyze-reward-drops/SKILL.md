---
name: analyze-reward-drops
description: 'Analyze verl RL reward drops and validation bad-sample error patterns from training logs, W&B val/generations tables, and val_data_gen JSONL. Correlate reward components, optimization diagnostics, and matched validation samples to distinguish noisy batches from regression. Use when: investigate reward collapse, explain critic/rewards/mean decreases, analyze a validation logging val table, compare bad cases across training steps, or diagnose ReMax/GRPO instability.'
argument-hint: '<log path or W&B run/table or val_data_gen path> [metric] [steps or drop threshold]'
---

# Analyze Reward Drops

Find significant reward regressions and validation bad-sample patterns from a
verl training run and produce an evidence-backed answer to:

1. **When:** training/validation step and source line or table row for the event.
2. **Where:** reward/component decreases or validation error-rate increases.
3. **Why:** likely drivers supported by component, optimization, generation,
   filtering, and error evidence.
4. **Which failures:** recurring validation error patterns, affected samples, and
   whether the same samples worsened compared with earlier validation steps.

For training-log analysis, use the bundled analyzer before manually reading
thousands of rollout lines. Treat its ranked drivers as hypotheses, not proof of
causation. For validation-table or bad-sample requests, read
[Validation bad-sample analysis](references/validation-bad-samples.md) and use
that workflow even if no training log or aggregate reward drop is available.

## Inputs and Defaults

| Input | Default |
|---|---|
| Log | Local `.log` file for training-drop analysis, commonly under `logs/<node>/` |
| Validation evidence | W&B run URL/ID and `val/generations` table/export, or `trainer.validation_data_dir` JSONL |
| Validation comparison | Last available validation before the event, first at/after it, and next available check; record actual steps |
| Primary metric | First available: `critic/rewards/mean`, `reward-core/score/mean`, `critic/score/mean` |
| Rolling baseline | Median of the preceding 5 parsed training steps |
| Drop threshold | Absolute decrease >= `0.05` **or** relative decrease >= `5%` |
| Report count | 10 largest drops |
| Output | Markdown on stdout; optional JSON with `--json` |

The rolling median reduces sensitivity to one noisy preceding batch. A reported
drop is still not automatically a training collapse: check whether the next
three reward points recover. These analyzer defaults apply to training reward
decreases; validation error rates such as `p_err` worsen when they increase.

## Workflow

Choose the entry point from the supplied evidence:

- **Training log:** run steps 1-3, then add validation evidence in step 4 when
  available.
- **Validation table or JSONL:** start at step 4; training-log diagnostics are
  optional supporting evidence, not a prerequisite.
- **Both:** locate reward-drop events first, then compare validation snapshots
  around those events using their actual logged steps.

### 1. Confirm the log and run the analyzer

Preserve the original log. Run:

```bash
python .github/skills/analyze-reward-drops/scripts/analyze_reward_drops.py \
  logs/verl-n2-i2/remax_2609v0_mix_ml_s1k_bs128_n4_r128_g16_kl5_raysubmit_wbjFziFSKst6DkxX_20260930T2027Z.log \
  --json tmp/reward-drop-analysis.json
```

Useful overrides:

```bash
# Inspect smaller movements with a longer baseline.
python .github/skills/analyze-reward-drops/scripts/analyze_reward_drops.py LOG \
  --window 10 --drop-abs 0.03 --drop-pct 3

# Analyze another logged objective explicitly.
python .github/skills/analyze-reward-drops/scripts/analyze_reward_drops.py LOG \
  --metric reward-core/char/mean
```

The parser strips ANSI color sequences, identifies dense `step:<N> - key:value`
metric records, deduplicates steps, rejects non-finite baselines, and reports
exact source line numbers. It does not confuse per-sample `[greedy_reward]`
records with aggregate training-step rewards.

### 2. Interpret “where”

For each ranked drop, report:

- step, line, rolling baseline, current reward, absolute delta, and relative delta;
- whether it is transient, sustained across the next three points, or unknown at
  the end of an incomplete log;
- the largest decreases among `reward-core/*/mean`.

Map component changes to the affected behavior:

| Component | Typical interpretation |
|---|---|
| `char`, `word`, `lex` | transcription/content accuracy worsened |
| `lang` | language detection or language-tag selection worsened |
| `fmt`, `bracket` | required output structure broke |
| `repeat`, `tail_hallu` | repetition or tail hallucination increased |
| `keyword` | required entities/keywords were missed |
| `punc`, `cap` | punctuation or capitalization worsened |

Do not use a component merely because its absolute value is low. Attribute the
drop to components that decreased against the same rolling baseline.
Verify the scorer's direction: error metrics such as `wer` or `n_err` worsen
when they rise, even under `reward-core/`. Do not treat their decreases as
adverse drivers just because the analyzer lists them.

### 3. Diagnose “why”

Use the analyzer’s correlated diagnostics, then inspect the local log context
around the reported line. Rank explanations by directness:

1. **Reward-component evidence:** one or more component means fell with the total.
2. **Batch/filter composition:** `train/gen_kept_frac` fell or
   `train/num_gen_batches` rose. This can indicate a harder batch or filtering
   selection shift rather than model collapse.
3. **Policy instability:** `actor/kl_loss`, `actor/grad_norm`, or
   `actor/pg_clipfrac` rose. Confirm the movement is unusual relative to nearby
   steps before claiming optimization instability.
4. **Generation degeneration:** response clipping or aborted ratio rose; inspect
   response lengths and sample outputs for truncation, empty output, repetition,
   malformed tags, or hallucination.
5. **Runtime/data errors:** search a bounded region before and after the drop for
   `ERROR`, `Traceback`, `NaN`, `OOM`, timeouts, worker loss, bad audio, or reward
   exceptions. Startup warnings far from the drop are not explanations.

Inspect the effective reward formula, weights, clipping, and greedy baseline.
Separate raw component quality from delta/vote contributions: a zero language
delta can mean sampled and greedy outputs both failed. Check actual filtering
settings before interpreting keep-fraction diagnostics; some trainers log them
even when filtering is disabled. Near-zero reward baselines can produce huge
relative changes, so report the absolute decrease prominently.

Example bounded inspection after the analyzer reports line `12780`:

```bash
sed -n '12720,12840p' LOG
```

Also inspect representative rollout records immediately before that aggregate
line. Link failures to IDs, language/data source, ground truth, response, and
component scores when available. Do not paste large transcript blocks into the
final report; summarize the pattern and cite the source lines.

### 4. Analyze validation bad samples

When validation tables or dumps are available, follow the
[validation workflow](references/validation-bad-samples.md) to:

- obtain the run's `val/generations` table and flatten its per-step sample slots;
- compare actual sample identities across validation steps and enrich table
  examples from the matching `val_data_gen/<data_source>/<step>.jsonl`;
- group errors into evidence-backed patterns with counts and short examples;
- check full per-dataset validation metrics before attributing a training drop
  to a model-quality regression.

The current table selects the lowest scores across validation datasets; it is
a selected subset whose membership changes. Neither a slot number nor a generic
audio prompt identifies a sample. Disappearance from the table is not proof of
recovery, and pattern frequency in the table is not validation-wide prevalence.

### 5. Distinguish a noisy batch from collapse

Call a drop **transient batch variance** when it recovers at the next few parsed
steps and the likely driver is a shifting batch/component mix without adverse
optimization diagnostics.

Call it a **possible sustained regression** only when low rewards persist and
the same component or diagnostic remains degraded. Call it **possible reward
collapse** only with a large sustained decline, broad component failure or
degenerate outputs, and supporting policy/runtime evidence.

If the log starts from a resumed checkpoint, state the resume step and source
checkpoint from the entrypoint. Do not interpret the first observed training
step as a fresh baseline. If the log ends during a rollout, state that the last
aggregate step may be missing.

## Required Report

Return a concise report with:

1. **Run context:** config, fresh/resumed status, observed step range, metric,
   baseline window, and thresholds; validation source, scoring direction,
   selected/full coverage, and compared validation steps when applicable.
2. **Drop table:** rank, step, line, baseline, current, absolute/relative change,
   and transient/sustained classification. For validation-only analysis, show
   available per-dataset before/after metrics and source rows instead; do not
   invent a training reward drop.
3. **Cause analysis:** component drivers and correlated diagnostics for each
   important event, with confidence (`high`, `medium`, or `low`).
4. **Sample/error evidence:** bounded examples tied to the event. For validation,
   include a pattern table with count/denominator, affected datasets/languages,
   sample identity, before/after score or error counts, short reference/output
   differences, and table row/slot or JSONL path/line citations. Separate newly
   regressed, persistent, improved, and unmatched cases.
5. **Conclusion:** noisy batch, possible sustained regression, possible collapse,
   persistent validation errors without demonstrated regression, or insufficient
   evidence.
6. **Next action:** a concrete check such as compare adjacent batches, inspect
   W&B unsampled history, validate data-source mix, or reduce/update a specific
   hyperparameter. Do not recommend a hyperparameter change without evidence.

Always separate:

- **Observed:** exact values and log lines.
- **Inferred:** likely explanation based on correlated evidence.
- **Unknown:** facts the log does not establish.

## Failure Handling

- If no aggregate training-step records exist, report that this may be a startup,
  validation-only, failed, or incomplete log; do not analyze per-sample rewards
  as an aggregate time series. Continue validation bad-sample analysis if its
  table or JSONL evidence is available.
- If the requested metric is absent, list the aggregate reward metrics that are
  present and rerun with an exact key.
- If no drop crosses the thresholds, report that fact and optionally rerun with
  explicitly disclosed lower thresholds. Never silently change the threshold.
- If only one or two points exist, report insufficient trend evidence.
- If only one validation snapshot exists, describe observed errors without
  claiming they are new regressions. If tables, ground truth, stable identities,
  or full dumps are unavailable, state the resulting limits explicitly; never
  substitute another run or infer missing samples were correct.
- Never infer causality solely from filename hyperparameters such as `kl5`,
  `bs128`, or `n4`.

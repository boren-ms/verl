---
name: analyze-reward-drops
description: 'Analyze verl RL training logs to find when and where rewards drop and explain likely causes from reward components, filtering, KL, gradients, clipping, response length, and nearby errors. Use when: analyze a training log, investigate reward collapse or regression, find a bad training step, explain critic/rewards/mean decreases, compare reward components, or diagnose ReMax/GRPO instability.'
argument-hint: '<local log path> [reward metric] [drop threshold]'
---

# Analyze Reward Drops

Find significant reward regressions in a verl training log and produce an
evidence-backed answer to:

1. **When:** training step and source line where the decrease occurred.
2. **Where:** primary reward metric and component reward metrics that fell.
3. **Why:** likely drivers supported by component, optimization, generation,
   filtering, and error evidence.

Use the bundled analyzer before manually reading thousands of rollout lines.
Treat its ranked drivers as hypotheses, not proof of causation.

## Inputs and Defaults

| Input | Default |
|---|---|
| Log | Required local `.log` file, commonly under `logs/<node>/` |
| Primary metric | First available: `critic/rewards/mean`, `reward-core/score/mean`, `critic/score/mean` |
| Rolling baseline | Median of the preceding 5 parsed training steps |
| Drop threshold | Absolute decrease >= `0.05` **or** relative decrease >= `5%` |
| Report count | 10 largest drops |
| Output | Markdown on stdout; optional JSON with `--json` |

The rolling median reduces sensitivity to one noisy preceding batch. A reported
drop is still not automatically a training collapse: check whether the next
three reward points recover.

## Workflow

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

Example bounded inspection after the analyzer reports line `12780`:

```bash
sed -n '12720,12840p' LOG
```

Also inspect representative rollout records immediately before that aggregate
line. Link failures to IDs, language/data source, ground truth, response, and
component scores when available. Do not paste large transcript blocks into the
final report; summarize the pattern and cite the source lines.

### 4. Distinguish a noisy batch from collapse

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
   baseline window, and thresholds.
2. **Drop table:** rank, step, line, baseline, current, absolute/relative change,
   and transient/sustained classification.
3. **Cause analysis:** component drivers and correlated diagnostics for each
   important event, with confidence (`high`, `medium`, or `low`).
4. **Sample/error evidence:** only bounded examples tied to the event.
5. **Conclusion:** noisy batch, possible sustained regression, possible collapse,
   or insufficient evidence.
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
  as a time series.
- If the requested metric is absent, list the aggregate reward metrics that are
  present and rerun with an exact key.
- If no drop crosses the thresholds, report that fact and optionally rerun with
  explicitly disclosed lower thresholds. Never silently change the threshold.
- If only one or two points exist, report insufficient trend evidence.
- Never infer causality solely from filename hyperparameters such as `kl5`,
  `bs128`, or `n4`.

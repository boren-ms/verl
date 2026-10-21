# Evidence and Metric Semantics

These are verified conventions in the verl/Phimm checkout. Inspect the code
revision used by historical runs before assuming identical behavior.

## W&B histories and steps

Use the read-only SDK with existing authentication for the resolved host:

```python
import re
import wandb

api = wandb.Api(overrides={"base_url": host}, timeout=60)
run = api.run(f"{entity}/{project}/{run_id}")
reward_key = re.compile(r"^reward-(?:core|delta|vote)/.+/mean$")
p_err_key = re.compile(r"^val-(?:core|aux)/(?P<dataset>.+)/p_err/mean@1$")

for row in run.scan_history(page_size=1000):
    metrics = {
        key: value for key, value in row.items()
        if reward_key.fullmatch(key) or p_err_key.fullmatch(key)
    }
    if metrics:
        observation = {
            "training/global_step": row.get("training/global_step"),
            "step": row.get("step"),
            "_step": row.get("_step"),
            "_timestamp": row.get("_timestamp"),
            "metrics": metrics,
        }
        # Validate values, preserve provenance, and append to evidence.
```

This is an extraction pattern, not a standalone exporter. Avoid passing a union
of sparse train/validation keys to `scan_history(keys=...)`: rows lacking any
requested key may disappear. `run.history()` can be sampled. Summary values
cannot establish the best checkpoint or a complete trajectory. Resolve table
references independently; scalar history does not contain the example text.

Current `Tracking.log(data, step)` in `verl/utils/tracking.py` sends the explicit
training step to W&B. `RayPPOTrainer.fit` in
`verl/trainer/ppo/ray_trainer.py` also emits `training/global_step` on training
rows. Initial validation may only have `_step`; it occurs after checkpoint load
and may therefore be above zero. Verify these mappings against the run's logger
before using `_step` as the optimizer step. Keep all available axes and surface
contradictions. Table rows have their own `step` column. Do not use runtime,
sample count, or the summary's latest step as the historical training axis.

Merge sparse observations only for the same run and verified training step with
nonconflicting values. Preserve repeated/restarted segments and conflicting
values instead of silently selecting the last one. A resumed segment can have
missing earlier history. Record collection start/end and current run state;
running runs do not form an atomic snapshot.

## Reward families

`_compute_reward_metrics` in `verl/trainer/ppo/ray_trainer.py` maps numeric
reward-info fields as follows, averaging only finite numeric entries:

| Raw field | Logged key | Meaning to verify in the scorer |
|---|---|---|
| `char`, `word`, `lang`, `score`, etc. | `reward-core/<field>/mean` | Raw component or final score; direction depends on the field |
| `d_char`, etc. | `reward-delta/char/mean` | Sampled versus greedy/baseline difference, possibly normalized |
| `v_char`, etc. | `reward-vote/char/mean` | Transformed/clipped vote component before its weight |

`recipe/phimm/reward/asr_vote.py` defines configurable per-component behavior.
Raw measures can be accuracy or accuracy minus one (negative error). A delta can
be absolute or relative. A vote can be thresholded or use a direct delta. Vote
means are not automatically win rates or bounded to `[-1, 1]`. Check thresholds,
clipping, weights, baseline policy, missing-component handling, and effective
formula. A zero delta/vote can mean both sampled and greedy outputs failed.
Training families are generally aggregate batch diagnostics, not per-validation-
dataset measures. Different training data mixtures can change them without a
model-quality change.

Compare raw numeric values only when definitions match. Do not interpret `wer`,
`n_err`, or other error measures as higher-is-better merely because they are
under `reward-core`. Do not infer a scientific configuration from filename tokens
alone; prefer logged effective config and the revision used by the run.

## Validation `p_err`

The exact keys are usually `val-core/<data_source>/p_err/mean@1` or
`val-aux/<data_source>/p_err/mean@1`; the namespace can vary with logged sample-count
aggregation, including whether a larger `@N` is present. Discover the actual key.
Preserve aliases and accept them as one observation only when their
same-step values agree. Treat an alias conflict as unresolved evidence.

`process_validation_metrics` and `update_var2metric2val` in
`verl/trainer/ppo/metric_utils.py` control aggregation. For a complete comparable
single-decode corpus, `p_err` normally reflects `sum(n_err) / sum(n_ref)`, not
the arithmetic mean of per-sample error rates. Repeated-decode, UID grouping,
and head-only/missing count fields change how aggregation must be reconstructed;
inspect those functions before recomputing. Do not overwrite official logged
values with a diagnostic rescore.

For each dataset and compatible cohort, distinguish matched-step evidence from
each run's latest or independently best result. If no exact common step exists,
show actual steps and the budget/cadence limitation. Do not fill missing values,
cherry-pick the dataset intersection, or pool datasets to conceal a regression.
Changing the reference style, normalization, parent/chunk scoring unit, or
decoding settings can invalidate a comparison even with the same dataset name.

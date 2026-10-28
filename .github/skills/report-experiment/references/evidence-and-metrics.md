# Evidence and Metric Semantics

Use these repository conventions while verifying the actual revision and
effective config of each historical run.

## W&B histories and optimizer steps

Use the read-only SDK with existing authentication. Scan full history; do not
use sampled `run.history()` output or summary values as trajectories.

```python
import re
import wandb

api = wandb.Api(overrides={"base_url": host}, timeout=60)
run = api.run(f"{entity}/{project}/{run_id}")
reward_key = re.compile(r"^reward-core/.+/mean$")
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
```

Avoid requesting a union of sparse training and validation keys through
`scan_history(keys=...)`; rows missing one requested family can disappear.
Store the source run, original row axis fields, timestamp, and exact metric key
with every observation.

Current tracking normally sends the explicit training step to W&B and emits
`training/global_step` on training rows. Initial validation can have only
`_step` and can occur after checkpoint load. Verify the logger behavior for the
run's revision before treating `_step` as optimizer step. Never use row index,
runtime, sample count, or the summary's latest step as the trajectory axis.

Merge sparse rows only within the same run and verified optimizer step when
values do not conflict. Preserve repeated/resumed segments and conflicting
values instead of silently taking the last row. Record collection time and run
state; a running run is a provisional snapshot.

## Effective settings

Prefer, in order:

1. logged effective/resolved config and launch overrides;
2. config snapshot stored with the run;
3. recipe at the exact code revision used by the run;
4. current recipe only as explicitly labeled fallback evidence.

Record the source for every material setting. Resolve Hydra inheritance and
interpolation before comparison. Filename tokens, display names, current YAML
defaults, and W&B tags are not effective settings by themselves. Sanitize
credentials and signed URLs from saved evidence.

When code revisions differ, inspect the relevant implementation before calling
identically named fields equivalent. Reward scorer behavior, data transforms,
validation aggregation, and rollout defaults can change without a recipe-field
change.

## `reward-core` semantics

`_compute_reward_metrics` in `verl/trainer/ppo/ray_trainer.py` maps finite
numeric reward-info fields such as `char`, `word`, `lang`, and `score` to
`reward-core/<field>/mean`.

The name does not determine direction or unit. Inspect the effective scorer,
commonly under `recipe/phimm/reward/`, and record:

- raw component definition and unit;
- higher- or lower-better direction;
- normalization and missing-value handling;
- component weight and whether the logged value is pre- or post-weight;
- training data mixture and batch construction;
- any revision difference affecting the formula.

Error counts/rates can be lower-better even though they live under
`reward-core`. Accuracy can be represented directly or as accuracy minus one.
Compare numeric values only when definitions and units match. Treat different
definitions as separate panels, not as one line chart.

`reward-core` values are aggregate training-batch diagnostics, not validation
metrics. Changes can reflect batch/data composition without a model-quality
change. Report correlation with validation trajectories only as descriptive
evidence.

## Validation `p_err`

Discover the exact observed keys, usually
`val-core/<data_source>/p_err/mean@1` or
`val-aux/<data_source>/p_err/mean@1`. Same-step aliases can be merged only when
their values agree; otherwise retain the conflict.

Validation aggregation is controlled by the run's metric-processing code. For
a compatible single-decode corpus, `p_err` commonly reflects total errors over
total reference units, not an arithmetic mean of per-sample error rates.
Repeated decoding, UID grouping, parent/chunk aggregation, and missing count
fields can change the interpretation. Do not overwrite official logged values
with a diagnostic rescore.

Before comparing a dataset across runs, verify:

- manifest/subset and data-source identity;
- reference and normalization conventions;
- parent/chunk or segment grouping unit;
- decoding settings and sample count;
- scorer and aggregation revision.

For compatible cohorts, distinguish exact matched-step evidence from each
run's latest or independently best observation. If no exact common step exists,
show actual steps and the cadence/budget limitation. Do not fill missing values,
interpolate sparse validation, cherry-pick only favorable dataset coverage, or
pool datasets to conceal a regression.

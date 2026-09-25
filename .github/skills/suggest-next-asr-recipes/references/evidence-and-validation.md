# Evidence Collection and Config Validation

Run from the repository root. Use existing local dependencies; do not start a
trainer, Ray job, model download, or GPU process to analyze configurations.

## W&B Read-Only Access

Use existing authentication for the specified host. Never print API keys, copy
credentials into commands or artifacts, disable TLS verification, or fall back to
the public W&B host. The example project's URL resolves to:

```python
import wandb

api = wandb.Api(overrides={"base_url": "https://msaip.wandb.io"}, timeout=60)
runs = api.runs("genai/v2607_earning", order="+created_at")
```

Iterate all pages, not just the first page or the newest runs. Record each run's
ID, URL, display name, state, creation time, last activity when available, and
effective config. Use a strict allowlist when saving config values: model path,
training and validation data definitions, algorithm/reward/scoring parameters,
actor optimizer/batching/LoRA/rollout settings, seed, training budget, validation
cadence, topology, and experiment/output identity. Do not dump `run.config`,
`run.summary`, or environment variables wholesale; configs can contain
`trainer.wandb_api_key`, proxy credentials, or signed URLs.

Fetch full history rather than `run.history()`:

```python
import re

p_err_key = re.compile(r"^val-(?:core|aux)/(?P<dataset>.+)/p_err/mean@1$")
for run in runs:
    for row in run.scan_history(page_size=1000):
        metrics = {key: value for key, value in row.items() if p_err_key.fullmatch(key)}
        if metrics:
            observation = {
                "run_id": run.id,
                "step": row.get("step"),
                "_step": row.get("_step"),
                "_timestamp": row.get("_timestamp"),
                "metrics": metrics,
            }
            # Validate and append to this run's evidence history.
```

This is an extraction pattern, not a standalone exporter. Persist the validated
observations with the schema below. Avoid passing the union of all metric keys
to `scan_history(keys=...)`: W&B returns rows containing all requested keys, which
can discard sparse validation rows and initial validation without a `step` field.
Keep snapshot start/end times and each run's observed state; a running project is
not an atomic snapshot.

In this repository, `Tracking.log(data, step)` passes the training step to W&B;
initial validation can have `_step` without a separate `step` metric. Verify the
logging implementation/revision for the runs being compared before using `_step`.
Preserve both fields and flag contradictions. Do not use summary-level `step`
as the step for every historical metric.

### Evidence artifact

The required `evidence.json` should contain:

| Section | Required contents |
|---|---|
| `snapshot` | Start/end UTC, project URL/host/entity/project, source type (API/export/log), local revision/dirty paths, collection completeness and failures. |
| `objective` | Exact original metric keys, target datasets, fixed weights, fraction/percent units, scoring/manifest/decoding identities, cohort rules. |
| `runs` | One entry per run ID with URL/name/state, timestamps, mapped recipe, code revision if known, sanitized effective settings, overrides/drift, cohort, exclusions, validated history and rejected observations with reasons. |
| `comparisons` | Per-run initial/latest/best complete vectors and steps, matched-step comparisons, aggregate arithmetic, provisional flags, missing targets, and repeated-run uncertainty. |
| `candidates` | Priority, path/name, control run/recipe, supporting run IDs/steps, exact dotted-key before/after changes, hypothesis, deduplication result, success rule, config-validation outcome. |

Use JSON `null` plus a reason for missing values, not zero or nonstandard JSON
NaN/Infinity. Preserve rejected raw values as diagnostic strings if necessary.
For unknown revision/manifest provenance, state the gap rather than inventing a
fingerprint. Do not let an unknown comparison contract enter the definitive
winner ranking. A partial API failure must remain visible even if other runs
were collected successfully.

### Worked scoring example (illustrative, not live project evidence)

| Run | Step | Verbatim p_err | Lexical p_err | Same-step mean |
|---|---:|---:|---:|---:|
| Control | 100 | 0.10 | 0.08 | 0.09 |
| Variant | 50 | 0.09 | 0.09 | 0.09 |
| Variant | 100 | 0.08 | 0.10 | 0.09 |

Correct: at matched step 100, both aggregates are 9.00%. The variant improves
verbatim by 2.00 percentage points but regresses lexical by 2.00 points; it is not
an overall improvement and fails a no-regression guardrail. A higher training
reward would not change this conclusion.

Incorrect: averaging the variant's best verbatim value at step 100 with its best
lexical value at step 50 gives 8.50%. No checkpoint achieved that score. Similarly,
dropping lexical and calling the variant the winner changes the fixed objective.

## Lightweight Hydra Check

Read the trusted local recipes before resolving their `eval` interpolations.
This repository registers an arithmetic `eval` resolver in
`recipe/phimm/main_asr_dapo.py`; the check below uses the same resolver name with
builtins disabled and does not import the GPU training entrypoint.

After creating candidates, replace the illustrative paths with **all** candidate
paths. This checks actual composed configuration, not YAML syntax alone:

```bash
python - recipe/phimm/config/<project>/<candidate>.yaml <<'PY'
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"__builtins__": {}}, {}))
for filename in sys.argv[1:]:
    path = Path(filename).resolve(strict=True)
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.stem)
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    trainer = resolved["trainer"]
    assert trainer["experiment_name"] == path.stem, path
    assert trainer["project_name"], path
    assert trainer["val_only"] is False, path
    assert trainer["test_freq"] > 0, path
    assert resolved["data"]["val_data"], path
    for key in ("default_local_dir", "default_hdfs_dir", "validation_data_dir"):
        assert path.stem in trainer[key], (path, key, trainer[key])
    print(f"PASS {path.name}: composed, resolved, training with validation, distinct named roots")
PY
```

Also check the following explicitly; the snippet alone does not prove them:

- The project equals the requested project, validation identities match the
  control, and every name token equals its resolved value.
- Candidate paths and output roots differ from every parent/existing candidate.
  Existing blobs/directories must not be reused when `resume_mode: auto`.
- The resolved diff, including logged parent overrides, contains only changes
  declared in the report. Store the dotted-key diff in the evidence artifact.
- Defaults resolve to the intended base/train-data YAMLs, and reward mappings
  cover the training data sources.
- The settings satisfy the relevant trainer's batching/resource constraints;
  composition alone does not establish VRAM fit or cluster availability.
- A claimed continuation uses an actually saved, complete checkpoint. A lower
  historical `p_err` at step N is not proof that `global_step_N` exists.

Use `git diff --check` and review the scoped diff/untracked files. Do not run
`quick_run.sh`, `submit_job.sh`, or training entrypoints as validation commands.
If dependencies or configuration prevent validation, report the precise failure,
repair the candidate if possible, and leave it explicitly unvalidated otherwise.

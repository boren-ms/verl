# Parallel rollout with fixed reference context

`remax_fl_earnings_short_random_merge23_x4_reference_context` is an alternative
to the FL + Earnings RM2–3×4 context experiment. Each training segment receives
the preceding segment's existing reference text in its user prompt. All segments
in a parent can then generate concurrently. Sampled trajectories and the ReMax
greedy baseline use the same fixed input context.

The preset inherits the thread's `text_history2` recipe, retaining 111,888
parents, 2 epochs, 1,748 optimizer updates, the model, optimizer, rollout sample
count and joint parent reward. It explicitly selects `max_history: 1` for the
requested previous-segment experiment. Override `asr.context_aware.max_history=2`
for a window matching the original ON experiment. The
context source is a training change, so this is not a pure throughput comparison.

Validation and standalone evaluation continue to use their own generated
history. Validation labels are never supplied as prompt context. The original
context-on/off presets retain their behavior.

## Use the published data

The preset reads six published JSONL manifests under:

```text
az://orngwus2cresco/data/boren/data/Training/fl_earnings_short_random_merge23_x4_reference_context_v1/
```

After checking out this branch and installing the recipe on a GPU node, use:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config/v2611_dev/earnings \
  --config-name remax_fl_earnings_short_random_merge23_x4_reference_context
```

The checked-in datasource YAML and the published `training.yaml` snapshot both
use Orange paths. They do not depend on the preparation machine's local data
directory. The model and runtime still require the normal GPU preflight.

## Switch training context by optimizer step

Use the prepared reference data for all phases. `asr.context_aware.training_source`
accepts either one source or a list of step phases. The checked-in schedule
preset uses one epoch, history 2 and three nearly equal phases:

```yaml
asr:
  context_aware:
    enable: true
    mode: text
    max_history: 2
    training_source:
      - start_step: 1
        source: null
      - start_step: 292
        source: reference
      - start_step: 584
        source: generated
```

Launch it with:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config/v2611_dev/earnings \
  --config-name remax_fl_earnings_short_random_merge23_x4_context_schedule_history2_1epoch
```

It trains 874 updates (`111888 // 128`): steps 1–291 use no context, 292–583
use reference context and 584–874 use generated context. History 2 uses up to
two preceding segments in both context phases and in validation. The published
reference data stores all preceding references, so no new data preparation is
needed.

Validation uses complete OpenLong Earnings21, Earnings22 and CORAAL. Checkpoints
are saved every 100 steps and validation runs every 400 steps. The trainer also
saves and validates the final step 874.

The code first selects one source for each decode:

1. If `enable` is false, use no context.
2. For validation and evaluation, use generated context.
3. For training, use `training_source` directly when it is one value. When it is
   a list, use the last phase whose `start_step` is at or before the prompt's
   absolute `global_steps`.

A phase list must start at step 1, and start steps must increase. The final
phase remains active until training ends. Sources are YAML `null` (or
`none`/`None`), `reference` and `generated`. Sampled rollout and its ReMax
greedy baseline use the same prompt step, and resumed training continues from
the restored absolute step. Asynchronous rollout uses the step assigned when
the prompt was issued.

No-context and reference segments decode concurrently. Generated segments
decode in order because each prompt uses earlier generated text. Reference
phases require the offline context field described below; other phases do not
read it. Validation never reads reference labels as prompt context.

## Reproduce the data

Run from the worktree root in an environment containing this recipe:

```bash
python -m recipe.phimm.tools.offline_reference_context \
  --config-dir plugins/phimm/src/recipe/phimm/config/v2611_dev/earnings \
  --config-name remax_fl_earnings_short_random_merge23_x4_text_history2 \
  --output-dir tmp/reference_context/fl_rm23_x4_reference_reproduced \
  +asr.context_aware.training_source=reference \
  asr.context_aware.max_history=1 \
  trainer.experiment_name=remax_fl_earnings_short_random_merge23_x4_reference_context
```

`--input-map mapping.json` accepts a JSON object mapping original manifest
URIs to local cached files. Preparation only writes local output files;
`--blob-root az://...` changes the emitted manifest locations for a separate
upload. Without it, `training.yaml` uses absolute local manifest paths.

Use the emitted snapshot for a subsequent training launch:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path /absolute/path/to/tmp/reference_context/fl_rm23_x4_reference_reproduced \
  --config-name training
```

Reproduction starts from the original `text_history2` data sources because
the new reference-context preset already points at the prepared release.
For a separate remote release, publish its manifests and use its emitted
snapshot with the matching Blob paths. Missing reference contexts raise an
error before any segment generation.

## Data and scoring contract

For a parent containing references `A`, `B`, `C`, the added raw field is:

```json
{"segment_reference_contexts": [[], ["A"], ["A", "B"]]}
```

The dataset assigns each entry to that segment's `asr_reference_context`.
The existing context builder applies `max_history`, `max_terms`, mode and prefix
when constructing prompts. With history 1 the model sees no context for the
first segment, `Previous context: A` for the second and `Previous context: B`
for the third. Current and future references are excluded from each context.
Empty FL labels still advance the history window; they do not carry an older
segment forward when history is 1.

Preparation reads exact per-segment text from `source_metadata_json`, checks
its alignment with `audio_path` and the joint parent reference, and refuses
missing or inconsistent provenance. It never guesses boundaries by splitting
the parent text. Every existing row field, parent order and segment order is
preserved. The generated `preparation.json` records input/output hashes and
parent/segment counts; `train_data.yaml` retains task formatting and passes the
new field through `extra_info`.

Each segment keeps its own audio, prompt, sampled tokens, logprobs and actor
features. Only newly generated responses are scored in original order against
the existing parent reference, and the same joint reward is assigned to all
segments. Fixed context tokens are part of the prompt and are not response
tokens in the actor loss.

## Local checks

```bash
PYTHONPATH=plugins/phimm/src:. python -m pytest -q \
  plugins/phimm/tests/test_reference_context.py \
  plugins/phimm/tests/test_agent_loop.py \
  plugins/phimm/tests/test_context_prompt.py \
  plugins/phimm/tests/test_random_merge_list.py \
  plugins/phimm/tests/test_u2_context_validation.py
```

The concurrency test holds every inference response until all segments have
reached the server, so a sequential rollout fails. Other checks cover greedy
training, validation isolation, empty references, history windows, actor inputs,
joint scoring and JSONL → task formatting → dataset round trips. These CPU
checks do not establish end-to-end GPU throughput or training quality.

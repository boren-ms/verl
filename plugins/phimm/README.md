# Base3a Qwen ASR on main's agent loop trainer

This package keeps the Base3a model, JSONL data mix, scoring and artifact
extensions on Main's V1 PPO/ReMax stack. It retains the `recipe.phimm` import
namespace while leaving Main's `recipe` submodule intact.

The entry point composes `verl.trainer.config.ppo_trainer` directly through
Hydra. A bundled Hydra search-path plugin registers the recipe and verl config
roots, allowing the main YAML files to be inherited directly.
`config/v2611_dev/base.yaml` declares Main worker/model/rollout settings;
`main/remax_main_base3a_2611r2.yaml` declares the training and validation
manifests, source rewards and row/update contract. Evaluation composes the same
native base. CLI overrides are applied once and remain live in interpolations;
unsupported legacy fields fail during Hydra composition. There is no Python
configuration translation or silent field filtering.

`ASRDataset` implements Main's raw-prompt custom dataset contract directly.
Only the JSONL loaders and transforms used by the retained manifests are
supported. Dataset loading always reads the JSONL and applies its transforms;
there is no recipe-level dataset disk cache. Audio processing/tokenization runs
in the agent loop. The ASR loop
uses Main's registration decorator, and its worker inherits Main's session
scheduling and TransferQueue postprocessing. The trainer subclass retains
corpus metrics, durable checkpoint resume and validation artifacts.

See [REFACTOR.md](REFACTOR.md) for the dependency boundaries, removals and local
validation limits. Historical reproduction reports remain unchanged.

Install into the existing GPU environment without replacing its dependencies:

```bash
uv pip install --system --no-deps -e . -e plugins/phimm -e plugins/qwen35_audio
uv pip install --system opencc-python-reimplemented==0.1.7 wetext==0.2.1
python -m pytest -q plugins/phimm/tests/test_adapter.py plugins/phimm/tests/test_native_recipe.py
python -m recipe.phimm.preflight trainer.experiment_name=<unique-name>
```

The two pinned text-normalizer dependencies match the source requirements.
Reinstall `plugins/phimm` in existing environments after this update so its
`hydra_plugins` namespace is discoverable before composing recipe configs.
Preflight exercises the Chinese, Vietnamese, and French scorers before model
initialization, including dependencies loaded lazily by the Chinese scorer.

`ASRAgentLoop` uses main's LLM client and TransferQueue worker. Each training
prompt gets one shared greedy baseline before its eight sampled trajectories.
The ASR scorer computes the delta/absolute vote reward. The
trainer passes the baseline score to the standard ReMax advantage estimator.
Validation groups and sorts segments before scoring full parent references;
corpus error rates use summed error/reference counts.
The default U2 cohort uses
`openlong_all_2parents_error_2609r2b_u2.yaml`: 1,085 unchanged audio clips
from eight parent recordings, with the original full reference repeated on
that parent's rows. Grouping metadata is retained, so WER aligns joined new
outputs against the full reference once per parent. All U2 callers use this canonical configuration. `asr.context_aware`
controls decoding separately and is not changed by the scoring default.
Historical independently scored U2 WER is a different metric convention.

The immutable manifests and `manifest_verified.json` can be reproduced without
loading a model using `python -m recipe.phimm.data.u2_full_reference` on an
existing node with Blob access (`--dry-run` validates without publication).
The original inferred text is retained as `legacy_segment_text` for audit.
Cleaned AA retains its original U2 clips; its two endings precede the canonical
chunk manifest by 0.692s and 10.956s. Running jobs retain their launch snapshots.

Training and validation pass the same ordered list of raw segment responses
to the scorer. They share response parsing and text extraction, preserving
independent boundaries for language/format checks when those measures are
configured. Validation retains its configured benchmark scorers and corpus
counts. Both modes select their scorer from the same top-level `reward_map`
and `rewards`; there is no separate validation reward namespace.
The map uses the sample's `data_source` as its key and a registered scorer
name as its value. Without a mapping, the source name itself is used. All
registrations (`path`, `name`, `reward_kwargs`) live in
[`config/reward/asr.yaml`](src/recipe/phimm/config/reward/asr.yaml), loaded by
the base config. Training presets use a `train_` prefix and benchmark presets
use `val_`; these are names in the shared registry, not runtime mode switches.
Validation source mappings are defined in `v2611_dev/base.yaml` and inherited
by training and standalone evaluation. The ReMax training YAML adds only its
training source mappings. Hydra merges them into one `reward_map`, for example:

```yaml
reward_map:
  noisy_ls: train_librispeech
  earning: val_openasr_en
```

Using the same source, responses, kwargs and baseline produces the same
reward in either mode. Aggregate validation text is only prepared for
display after scoring.

All response-scoring entry points accept either one string or a nonempty
ordered list of strings: measure/vote rewards, language-only reward,
OpenASR (English, multilingual and Hindi lattice),
Mandarin/Vietnamese/French expansion, inhouse DTER/EER, non-speech and
foreign-language evaluation. Empty segment strings are retained for checks;
an empty response list or non-string list entry raises an error. Shared text
normalization and alignment counts live in `reward/asr_metrics.py`.
Non-speech and foreign-language
checks require every response to satisfy the sample's task, while retaining
one parent-level reference count.

Recording-level diarization evaluation uses
[`reward/diarization_eval.py`](src/recipe/phimm/reward/diarization_eval.py),
with the same `solution_str`, `ground_truth`, `extra_info` dispatch interface as
`reward/asr_eval.py`. Install its optional CPU scoring dependencies with
`uv pip install -e 'plugins/phimm[diarization]'` in the evaluation environment.
It runs inside the existing agent-loop evaluation; there is no postprocessing
command. The opt-in
[`config/reward/diarization.yaml`](src/recipe/phimm/config/reward/diarization.yaml)
adds `val_diarization` and maps `dai_asr_i18n_en_public` to it while retaining
the existing ASR scorers.

The consolidated input JSONL keeps all references on each recording's row:

```json
{"id":"example","audio_path":"/data/example.wav","text":"Hello world","language":"English","data_source":"dai_asr_i18n_en_public","evaluation_duration_s":2.0,"reference_segments":[{"speaker":"A","start_s":0.0,"end_s":1.0},{"speaker":"B","start_s":1.0,"end_s":2.0}],"reference_words":[{"speaker":"A","text":"Hello"},{"speaker":"B","text":"world"}]}
```

The DAI preset accepts ordinary ASR text and uses an explicit single-speaker
baseline (`prediction_format: text`, `text_speaker: speaker_0`). Every nonempty
decoded segment is marked as speaker_0 over its full decoding interval from
`svad_bounds`. Empty or `<nonspeech>` outputs produce no speech interval.
All recognized words receive the same label. The original ASR prompt and
response cleanup are reused; the model need not emit JSON, labels or timestamps.
No additional acoustic diarization or forced-alignment model is loaded.

For example, decoded texts `["Hello", "world"]` with
`svad_bounds: [[0, 1], [1, 2]]` yield:

```json
{"segments":[{"speaker":"speaker_0","start_s":0.0,"end_s":1.0},{"speaker":"speaker_0","start_s":1.0,"end_s":2.0}],"words":[{"speaker":"speaker_0","text":"Hello"},{"speaker":"speaker_0","text":"world"}]}
```

Against reference A saying Hello during `[0, 1]` and reference B saying world
during `[1, 2]`, optimal mapping can match speaker_0 to only one reference
speaker: both DER and WDER are 50%. This baseline does not recognize speaker
changes. DER depends on the segment activity and boundaries, not on the
recognized wording apart from whether each segment is empty. Within-segment
silence can count as false alarm, and speaker changes/overlap can incur errors.
`single_speaker_baseline: 1` is recorded in the scoring output.

Labels are mapped optimally once per recording. SVAD-prepared rows retain one
recording per sample, with an ordered `audio_path` list and matching
`svad_bounds: [[start_s, end_s], ...]` on the same JSONL row. The agent loop
returns one response per segment. Words are joined in segment order.

For models that actually emit speaker annotations, set
`rewards.val_diarization.reward_kwargs.prediction_format=structured` and restore
a prompt requesting JSON objects with `segments` and `words`. Structured
timestamps are local to each SVAD segment; scoring clips them to its length and
adds its start offset. Emitted speaker labels must be consistent across segments;
they are never remapped independently per segment.

Every decoding segment must have one response. Text mode can also use
`seg_start`/`seg_end` for one segment, or `evaluation_duration_s` for one full
recording. Without timing metadata DER is unavailable; WDER still works from
the recognized words with their common speaker label. In structured mode,
every segment must supply a valid metric field for that metric to be available.
The original full reference annotations stay in `extra_info` and never enter
the prompt. Reference speech in gaps omitted by SVAD remains in the denominator
and counts as missed speech, rather than disappearing from the evaluation.

The data YAMLs in
[`config/data/val_data/diarization/`](src/recipe/phimm/config/data/val_data/diarization/)
select the Orange `public-en-jsonl/svad30/` manifests, prepared offline with
the bundled Smart VAD and a maximum segment length of 30 seconds:
`dai_asr_i18n_public_en` (48),
`dai_asr_i18n_public_en_validation_smoke` (3), and
`dai_asr_i18n_public_en_internal_test` (45). The internal subsets are disjoint;
the full public view includes both. These data are evaluation only, with no
training split. Each YAML requests standard verbatim transcription and retains
all reference fields from its JSONL row, without annotation sidecars.

For example, the audio field becomes
`"audio_path": ["/data/example.wav#0.0:1.0", "/data/example.wav#1.0:2.0"]`,
with `"svad_bounds": [[0.0, 1.0], [1.0, 2.0]]`. The transcript, reference
segments, reference words, ID and other attached fields remain unchanged.
The existing offline preparation command is reusable:

```bash
python -m recipe.phimm.tools.svad_segment --src data.jsonl \
  --dst data-svad30.jsonl --max-len-sec 30
```

Use the existing evaluation entry point with a merged model and unique run name:

```bash
python -m recipe.phimm.main_asr_eval --config-name eval_diarization \
  actor_rollout_ref.model.path=<merged-hf-checkpoint> \
  trainer.experiment_name=<unique-eval-name>
```

For the smoke subset, add
`data/val_data/diarization=dai_asr_i18n_public_en_validation_smoke`.
The scorer returns fractional `der` and `wder` plus sufficient counts:

- DER uses zero collar, includes overlapping speech and clips to
  `evaluation_duration_s` when supplied. Prepared DAI reference segments already
  include the upstream 200 ms same-speaker word-gap merge. Native model segments
  are used as supplied. Corpus DER pools `der_n_err / der_n_ref`, where the
  denominator is reference speaker-seconds. Miss, false alarm and confusion
  counts are also returned.
- WDER aligns normalized words with Levenshtein, includes substituted pairs,
  excludes insertions/deletions and pools `wder_n_err / wder_n_ref` over aligned
  word pairs. It is a supplemental speaker metric, distinct from WER. Default
  normalization is the existing ASR `hf_english` normalizer applied per word.
  To reproduce the earlier DAI-based WDER normalization, install that toolkit
  and set `rewards.val_diarization.reward_kwargs.normalization=dai_asr_i18n`.
- Reward defaults to `1-DER`. Set
  `rewards.val_diarization.reward_kwargs.reward_metric=wder` for `1-WDER`.
  Both metrics are returned in either case. DER can exceed one and its reward
  can be negative. Missing or invalid predicted annotations give `null` rates,
  an `*_unavailable` flag and zero reward for the selected unavailable metric.
  An explicit empty segment list counts as missed speech. Zero denominators
  give undefined rates while retaining primitive counts for corpus pooling.

Validation logs pooled `val-core/<source>/der_p_err/mean@1` and
`val-core/<source>/wder_p_err/mean@1`. If any recording lacks valid predicted
annotations for a metric, its pooled corpus rate is omitted; the other metric
can still be reported. The DAI preset also enables `compute_wer: true`, reusing
the existing OpenASR English scorer on the joined transcription. Its `n_err`
and `n_ref` counts produce the usual pooled `val-core/<source>/p_err/mean@1`
WER alongside DER/WDER. This does not change the selected diarization reward.

For a concrete DER example, let reference A speak during `[0, 2]`, reference B
during `[1, 3]`, and hypothesis X during `[0, 4]`. Mapping X to A gives one
speaker-second of missed speech during `[1, 2]`, one of confusion during
`[2, 3]`, and one of false alarm during `[3, 4]`. The denominator is four
reference speaker-seconds, including both speakers in the overlap:
`DER = (1 + 1 + 1) / 4 = 75%`.

The adapter retains batch128, LoRA320/640, LR6e-7, two epochs, six-source
mix, validation20/save10, and auto resume. Runtime checks require 55,664 rows
and 868 updates. Main's decoupled token importance correction has cap5 without
batch normalization. Its on-policy loss and gradient are tested against the
source objective; end-to-end quality still requires actual experiment results.

The colocated async Base3a variant keeps the same two-node GPU pool, data,
batch sizes and optimizer settings. It uses one warmup batch and partial
rollout through V1's `colocate_async` scheduler, with the shared ASR metrics
and artifact handling. Select it with:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config/v2611_dev/async \
  --config-name remax_main_base3a_2611r2_async \
  trainer.experiment_name=<unique-async-name>
```

The sync/async comparison presets are grouped under `config/v2611_dev/async/`:

- `remax_main_base3a_2611r2_async`: original async settings.
- `remax_main_base3a_2611r2_async_delay16`: two warmup batches and delay tolerance 16.
- `remax_main_base3a_2611r2_sync_no_ngram`: synchronous control with n-gram blocking disabled.

With TransferQueue checkpoint support (version 0.1.9 or newer), the ASR
callback uploads the queue snapshot before publishing the latest-step marker.
Remote resume restores it so completed trajectories survive and in-flight
prompts can be reissued by V1. This preset places the CPU queue storage actors
and trainer controller on the driver node (`asr.colocate_queue_on_driver=true`),
so checkpoint snapshots work with node-local storage. GPU training and rollout
still use both nodes. If disabling this placement to distribute queue storage,
set `trainer.default_local_dir` to a shared filesystem mounted at the same path
on every node. Restored snapshots are cached under that directory.
Older TransferQueue versions retain V1's
fallback behavior without queue recovery. GPU throughput and quality still
need validation in an actual training run.

Random segment variants are grouped under `config/v2611_dev/rand_segment/`.
Select them with `--config-path config/v2611_dev/rand_segment` and `--config-name`:

- `remax_main_base3a_2611r2_sc`: `segment_count: [1, 2]`.
- `remax_main_base3a_2611r2_sc1`: `segment_count: [2, 5]`.
- `remax_main_base3a_2611r2_sc2`: `segment_count: [1, 4]` (inherits the SC preset).
- `remax_main_base3a_2611r2_sl`: `segment_len: [5, 30]` seconds.
- `remax_main_base3a_2611r2_sl1`: `segment_len: [1, 10]` seconds.
- `remax_main_base3a_2611r2_sl2`: `segment_len: [2, 5]` seconds.
- `remax_main_base3a_2611r2_sl3`: `segment_len: [10, 20]` seconds.
- `remax_main_base3a_2611r2_sl4`: `segment_len: [2, 10]` seconds.

These presets inherit `main/remax_main_base3a_2611r2.yaml`, disable duration
cropping, and load prepared offline manifests directly. All 40 concrete segment
presets, including Earnings/FL, async, long-form, merge-then-segment and
continuation variants, use the datasource groups in
`config/data/train_data/offline_audio/`. Identical datasets share a group;
`preparation.json` records each preset's original generation options, ordered
sources, row counts, parity hashes and remote publication checksums. Their
training YAMLs select these manifests directly and contain no `offline_data`
section. Context is controlled by `asr.context_aware`. Preparation plans are
used only when explicitly invoking the offline tool. The manifests are
available under:

```text
az://orngwus2cresco/data/boren/data/Training/offline_audio_alignment_20261117_v1/<task>/source_000.jsonl.gz
```

The SC2, SC2a, FL5, synchronous SC1a and async SC/batch64/PPO4 recipes
also use prepared data. Shared variants reuse existing manifests; additional
partitions are published under
`az://orngwus2cresco/data/boren/data/Training/offline_audio_main_merge_20261117_v1/`.
FL5 retains its minimum of two segments, including for short recordings.

Every generated training row was compared with the former online processing
order. Uploaded manifests were downloaded and checked against both compressed
and original JSONL SHA256 hashes. Gzip changes only storage; `audio_path`
remains an ordered list. Raw input manifests, original effective configs,
duration provenance and preparation scripts are archived under the same root's
`provenance/` directory. Existing offline Random Merge presets continue using
their prepared datasets.

Smart VAD can also prepare an ordered `audio_path` list for each long recording,
keeping the original full reference and writing no audio files. Use
`python -m recipe.phimm.tools.svad_segment --src input.jsonl --dst segmented.jsonl`
with the PHIMM environment, or select `offline_data.svad_segment` in the existing
preparation workflow. See [offline SVAD segmentation](src/recipe/phimm/utils/svad/README.md)
for dependencies, boundary behavior, and configuration. It uses the supported
`file#start:end` selector syntax.

Random Segment, Smart VAD, and Random Merge run only in offline tools, never in the
training loader's `pre_process` or `post_process` blocks. For a **new** dataset
or segmentation range, start from the raw base with a preparation plan:

```bash
PYTHONPATH=plugins/phimm/src:plugins/qwen35_audio/src:. \
uv run --no-project python -m recipe.phimm.tools.offline_audio \
  --config-dir plugins/phimm/src/recipe/phimm/config/v2611_dev/main \
  --config-name remax_main_base3a_2611r2 \
  --output-dir /shared/prepared/base3a_sc1 \
  +asr_training=context_aware offline_data.random_segment.segment_count=[2,2]
```

Launch training using the generated snapshot:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path /shared/prepared/base3a_sc1 --config-name training
```

The output directory must be new. It contains fixed `source_NNN.jsonl`
manifests, `train_data.yaml`, `training.yaml`, and `preparation.json` with
settings, counts and hashes. The tool applies raw path/text normalization,
generates the audio lists, then saves rows **before** task information,
foreign-language masking and prompt/reference formatting. Those steps still
run when training loads the prepared rows. Validation inputs, optimizer
budgets, and `asr.context_aware` are preserved. Prepare new datasets with this
tool before training, then launch the generated snapshot or update the preset
to read the prepared manifests directly.

Use a directory accessible on all training nodes. `--blob-root az://...`
sets emitted manifest URIs; upload the generated JSONL files separately before
launching. For new mixes, review merged-parent budgets and expected row counts.
To reproduce an existing preset's data, use its original effective config from
`provenance/original_configs/`; the ready-to-train preset has no active
generation plan. Changing segmentation settings requires regenerating the
manifests, rather than overriding a prepared preset's runtime settings.
For more control, pass Hydra overrides after the CLI options, for example
`offline_data.random_segment.even_split=true`. `+asr_training=multi_segment`,
`+asr_training=context_aware` and `+asr_training=random_merge` select offline
preparation plans. Already-materialized merge presets need no extra preparation.

Each original recording remains one sample with its full reference. An offline
segment plan replaces `audio_path` with an ordered list of selectors:

```yaml
offline_data:
  random_segment:
    segment_count: [1, 2]
    segment_len: null
    seed: 42
```

For exactly two segments, set `segment_count: [2, 2]`. Cut points are random;
for example, `A.wav` can become `[A.wav#0%:35%, A.wav#35%:100%]`, using the
existing audio range format.
Segment counts and cut points are deterministic
for a seed and source-row index. They are sampled during dataset preparation,
so all rollout trajectories and the greedy baseline use the same partition.
Segments cover the entire recording without gaps or overlap. Audio decoding
and range reads happen in the agent worker, not during dataset preparation.
Selectors can also partition an existing `#start:end` audio range or an
archive-backed audio entry.

For `audio_path: a.wav;b.wav`, the worker first concatenates the decoded
waveforms in that order, then slices the combined waveform. A segment can span
the original file boundary. Generated selectors include a full inner range
(`a.wav;b.wav#0%:100%#start%:end%`) so the segment range applies to the complete
concatenation. Dataset preparation does not create a merged audio file.

Alternatively, select duration mode instead of count mode:

```yaml
random_segment:
  segment_count: null
  segment_len: [10, 20]  # Seconds; draw a new length for every segment.
  seed: 42
```

`segment_len` defaults to `null`, selecting count mode. If neither range is
provided, count mode defaults to `segment_count: [1, 2]`. Two non-null ranges
raise an error; when switching the training config to duration mode, set
`segment_count: null` and supply `segment_len`. The old `range` and `max_segment_length` options
are removed. Count mode samples an inclusive integer count and needs no duration
lookup. Cut points are random by default. Set `even_split: true` in the
`random_segment` block to divide the sampled count into equal-duration segments:

```yaml
random_segment:
  segment_count: [2, 2]
  segment_len: null
  even_split: true  # Two halves: [0%, 50%] and [50%, 100%].
  seed: 42
```

The count is still sampled uniformly when its bounds differ. Even splitting
uses the full recording, preserves the original reference, and needs no duration
lookup. It is supported only in count mode; combining it with `segment_len`
raises an error. For the earnings-short SC preset, enable it with
`offline_data.random_segment.even_split=true`.

`min_segment_fraction` is an optional
count-mode-only floor relative to the full recording,
defaulting to `0.1` for random cuts and `0` for even splits, and limited to
`1 / segment_count[1]`. For more than ten possible segments, explicitly lower
this fraction so the requested counts remain feasible.

Duration mode independently draws a new length from `segment_len` for every
segment, rather than choosing one length for the entire recording. Sampling
continues until the remaining audio fits within the upper bound. The final
remainder is retained and may be shorter than the lower bound; a recording
shorter than the lower bound stays as one segment. Segment count has no configured
upper limit in this mode. A degenerate range such as `[20, 20]` explicitly requests 20-second segments
plus the final remainder.

Duration mode first uses the row's `audio_length` (seconds); if absent, it reads
audio header metadata without waveform decoding. Ordinary remote files use
seekable metadata reads; archive and legacy chunk sources inspect the selected
encoded entry using the byte index rather than ChunkManager. Building an index
for an uncached legacy chunk may first localize the encoded chunk to disk;
supplying `audio_length` avoids that IO as well. Existing time selectors are
accounted for when reading duration metadata. `audio_length`, when supplied,
must describe the selected source's duration.

The training loader consumes the saved list unchanged. All sampled rollouts,
greedy baselines and epochs reuse the same saved boundaries; validation is
never randomly augmented. The normal per-segment token limits still apply.

### Random Merge: ordered audio lists

Use `+asr_training=random_merge` with `recipe.phimm.tools.offline_audio`.
This selects ordered-list merging before task information is added, enables
`asr.context_aware=true`, reserves a 2048-token prompt budget and disables
random cuts. The offline plan is:

```yaml
offline_data:
  random_merge:
    max_audio_length: 180
    buffer_size: 100
    seed: 42
    merge_count: [2, 3]
    text_separator: " "
    group_by: [data_source, prompt]
  random_segment:
    enabled: false
```

A random starting utterance and compatible utterances that fit the cap form
one parent, up to a uniformly sampled target count in `merge_count` (inclusive).
Omit `merge_count` for duration-only packing. Packing uses bounded lookahead and chooses the longest fitting
candidate, with random ties. Every input appears once; a leftover remains a
singleton list when the buffer or duration cap prevents a pair. The minimum
count is a target, never a reason to drop a source. Duration uses `audio_length` or audio header metadata; no
waveform is decoded or rewritten. Inputs must be individual path strings;
already concatenated or segmented lists are rejected. An input at or above
the cap raises rather than truncating its reference.

For example, merging two rows produces the following structure (order is the
sampled audio order):

```json
{
  "audio_path": ["A.wav", "B.wav"],
  "text": "hello bonjour",
  "keywords": ["Alice", "Paris"],
  "language": "English French",
  "components": [
    {"language": "English", "target_language": "English", "text": "hello"},
    {"language": "French", "target_language": "French", "text": "bonjour"}
  ]
}
```

The ASR dataset expands this list into two segments, preserving the original
utterance boundaries. B is decoded with A's generated response as an assistant
prefix. The full joined text is scored once per trajectory; the prefix is not
scored twice. Sampled and greedy trajectories retain their own histories.
Do not combine this block with enabled random segmentation.

Text variants follow the same order, keywords are unioned without duplicates,
and language names/codes are normalized and unioned. Existing components are
flattened in order. Missing languages use the source's configured hint, or
English when no hint is given. Merged language unions survive task formatting.
To keep packs monolingual, include `language` in `group_by`.
`source_audio_lengths`, `source_languages`, and `random_merge_source_indices`
align with the audio list. Other producer fields become ordered lists of
JSON-encoded values; `source_metadata_json` retains complete original rows.
Stale producer `gt_output` is regenerated from the merged text/components.
Use `verl_format.extra_keys` for provenance fields needed in rollout details.

Packing changes the number of dataset rows. Mix budgets now count merged
parents; choose an appropriate budget (for example `epoch: 1` per datasource)
and training step target for a new experiment. The merge plan clears inherited fixed row/step assertions. Existing jobs and
their targets are unchanged.

For source-spec-based offline packing, use the same raw preprocessing
before task formatting, then load the resulting manifests without merging again:

```bash
python -m recipe.phimm.tools.offline_random_merge \
  --sources sources.json --output-dir merged_data \
  --audio-format list --merge-count 2 3 --cap 180 --seed 42
```

`sources.json` is a list of `{name, config}` source blocks. The command records
source/manifest hashes, exact-once coverage, duration conservation, count
distributions and JSONL loader roundtrips. Generated YAML runs `add_task_info`
on the merged raw metadata. Enabled merging in `post_process` is rejected.
The `remax_earnings_short_random_merge23_context` Earnings preset uses the
materialized list dataset, 394 optimizer updates and U2 full-parent references
with history decoding. This matches the SC1 context run's update budget; audio
exposure per update differs because a parent contains multiple utterances.

For a matched context-off experiment, use
`remax_earnings_short_random_merge23_no_context`. It reads the exact same
materialized manifests and sets `asr.context_aware: false` for sampled training,
the ReMax greedy baseline AND validation. All three decode with empty assistant
prefixes. U2 grouping and full-parent references are retained for WER scoring.
The old preset name `remax_earnings_short_random_merge23_no_train_context` is
an alias with the same shared policy. There is no separate training override.
Use only `asr.context_aware` in configurations and CLI overrides.

### Offline consecutive utterance windows

`offline_contiguous_merge` builds deterministic overlapping windows of two or
three utterances from the same original recording. It groups by
`original_audio_url` and orders clips by `start_seconds + part_start_seconds`.
For FY27 Earnings batches without timestamps, `--order-by auto` groups by the
recording name in `id` and sorts numeric `_segN`/`_partN` suffixes. It uses
timestamps whenever those fields are supplied. Missing segment numbers remain
gaps in the source; no missing utterances or timestamps are invented.
Each recording contributes exactly as many merged samples as original rows;
neighboring utterances are reused, but complete windows are unique. Groups
with fewer than three rows are rejected because they cannot satisfy both
the sample budget and unique-window constraint.

```bash
PYTHONPATH=plugins/phimm/src:plugins/qwen35_audio/src:. \
uv run --no-project python -m recipe.phimm.tools.offline_contiguous_merge \
  --source-config plugins/phimm/src/recipe/phimm/config/data/train_data/quartr_1st23.yaml \
  --output-dir /tmp/quartr_first23_contiguous23
```

Raw path/text normalization runs before merging and task information is added
when training loads the result. `audio_path` is a list; references and language
components follow that list, keywords are a stable union, and source IDs,
timestamps and original records remain available for auditing. Consecutive
means neighboring transcribed clips; silence gaps in the original recording
are not inserted between files. No audio bytes are rewritten.

The full prepared datasource is
`data/train_data/offline_audio/earnings_short_contiguous23.yaml`. It contains
both FY27 Earnings batches and Quartr First23, Second48 and Final38: 25,240
original utterances from 180 recordings produce exactly 25,240 merged samples.
Each source and each recording retains its original sample count. The
standalone `quartr_1st23_contiguous23.yaml` shares the First23 partition of this
bundle. The paired training presets select all five prepared manifests,
retaining the original mix, full-reference U2 validation, batch128 and
394-step budget.

All Earnings presets inherit from `base_earning.yaml`, directly or through
another Earnings preset. This shared base inherits from `../base.yaml` and
defines the Earnings checkpoint, rewards and U2 validation without importing
or selecting training data. Each experiment selects its own training sources
and sampling mix. The long-form presets also define their own training budget.

The paired multi-segment ReMax presets are
`remax_earnings_short_contiguous23_context` and
`remax_earnings_short_contiguous23_no_context` under `config/v2611_dev/earnings/`.
They share all five prepared sources and the same ReMax recipe (`n=8`), and set
`asr.context_aware` to true/false across sampled rollout, greedy baseline and
validation. Both retain ordered 2/3-utterance lists and full-parent U2 scoring;
the context-off preset decodes each segment without previous transcript text.

Set `data.max_audio_dur: null` when using
these complete lists; `asr.context_aware` controls history for both training and
validation. The sample count is unchanged, while total audio exposure grows
because each output contains two or three utterances.

The tool writes `data.jsonl`, `data.jsonl.gz`, `train_data.yaml` and
`preparation.json`. Multiple `--source-config` paths, or a file containing
multiple datasource blocks, produce `source_000/` and subsequent partitions
plus a combined datasource and preparation record. `--input-jsonl` accepts a
single local source; `--input-map` accepts a JSON map of source URIs to local
manifest files. FY27 Earnings requires `--duration-cache` with normalized
audio-path keys, since the raw manifests omit durations. `--blob-root` selects
the URI written into the datasource. Uploads are a separate operation. The
published bundle and audit are under
`az://orngwus2cresco/data/boren/data/Training/earnings_short_contiguous23_20261117_v1/`.

### Offline packing with concatenated audio

The existing offline preparation command retains its semicolon-based format
for recipes that concatenate and then resegment audio. Use:


```bash
python -m recipe.phimm.tools.offline_random_merge \
  --sources sources.json --output-dir /path/to/output \
  --blob-root az://orngwus2cresco/data/boren/data/verl/my_offline_dataset \
  --cap 180 --buffer-size 100 --seed 42 --workers 32
```

`sources.json` is a list of `{name, config}` objects; each `config` is an
ordinary raw JSONL training-data block. Every entry is packed independently.
Input limits, path mapping, field renaming and tagged-text cleaning retain
the original recipe's behavior. Duration uses `audio_length`,
`audio_duration_seconds` or `duration` when present; otherwise the command
reads audio headers, with a resumable duration cache. No merged audio is
written. All parents are strictly below the cap; leftovers are retained and
oversized individual sources fail preparation.

Each output directory contains `data.jsonl`, `statistics.json` and a
`train_data.yaml` with statistics in comments. The command also creates a
combined data YAML for the regular dataset loader.
Text and transcript variants use the audio order, keywords are unioned, and
components flattened. Other metadata is an ordered list rather than the
first constituent's value. Nullable or structured ancillary values are
JSON-encoded inside those lists; their field names are recorded in the
statistics. `source_metadata_json` additionally retains each original record,
including producer references and original timestamps. `source_audio_lengths`
and `random_merge_source_indices` align with the semicolon-separated paths.
The command audits exact row coverage, total duration and full JSONL loader
roundtrip. For training, use one epoch per merged source rather than retaining
the original short-row sample budgets.

Each segment is transcribed independently with its task prompt and its own
audio: `prompt + segment1 -> response1`, `prompt + segment2 -> response2`.
No previous response or segment is included in the next request. The reward
scorer receives an ordered list of raw segment responses. It extracts and
joins their text for lexical, keyword, punctuation and capitalization
metrics against the original full reference. Language and format checks run
separately on each response and are averaged with equal segment weights;
when a reference lists multiple languages (for example `English_Chinese`),
each predicted target language name may match any listed language. Repeated
labels on separate lines are valid.

Responses use JSONL: one `{"language":"English","text":"Hello."}` object per
segment, with one or more lines in transcript order. The scorer joins the
`text` fields in order. A malformed line contributes its raw text, gets
no language credit, and fails the format check for that response. Combined metrics
(`word_lang`, `char_lang`, `word_fmt`) use the same per-segment checks.
ReMax likewise generates all greedy segment responses and scores them as one
sample before sampled trajectories start. The agent returns independent
outputs through Main's multi-output interface (`uid_session_segment`), each
with the shared sample reward and its original generated IDs, logprobs and
audio features. Actor loss uses each segment's independent context and the
existing sequence/token aggregation; samples with more segments contribute
more segment loss terms. Dataset length, source mixing budgets and
`max_samples` continue to count original recordings. Segment count can thus
increase actor work and optimizer minibatches without increasing prompt count.

Sampling passes `ngram_size=15` and `window_size=512` to the registered
DeepSeek processor. The configured window now takes effect for sampled
rollouts, greedy baselines and validation. This deliberately corrects the
source's `ngram_window` spelling, which was ignored and resulted in a
100-token window. Validation inherits the training no-repeat settings unless
its custom settings are overridden; greedy baselines use the validation path
and its 512-token response limit.
It also preserves the source LLM's seed0 on every replica and raw rollout
log probabilities, rather than inheriting main's seed42 and processed scores.

For an already synced two-node Ray pool:

```bash
python plugins/phimm/submit.py --submission-id <unique-id> \
  trainer.experiment_name=<unique-name>
```

The submission ID is also the default experiment name; an explicit
`trainer.experiment_name=` override takes precedence. Select a grouped preset
with, for example, `--config-path config/v2611_dev/long --config-name
remax_main_base3a_2611r2_sl_long5`. Config paths are relative to the recipe
module. Sync and install editable packages on every GPU node before submitting
(see installation above); the submitter sets source paths for driver and
workers but does not install dependencies or verify cluster capacity.

Initial validation can be checked separately with `trainer.val_only=true`
and a separate experiment name. It does not replace the training run.
After the final checkpoint is exported and verified, full evaluation uses:

```bash
python plugins/phimm/submit.py --evaluate --submission-id <unique-eval-id> \
  trainer.experiment_name=<unique-eval-name> \
  actor_rollout_ref.model.path=<verified-merged-HF-path>
```

The evaluation manifest retains the reference run's 35 datasets, including
`eslo_telephony_fr` and `accueil_ubs_fr` from its producer revision
`965a025ac220c91305aa762bdb176697b55e0fb5`. The current development branch's
generic manifest omits these two reference datasets.

For context-aware OpenLong evaluation, select `eval_openlong_context_zip`:

```bash
python plugins/phimm/submit.py --evaluate --submission-id <unique-context-eval-id> \
  --config-name eval_openlong_context_zip \
  actor_rollout_ref.model.path=<verified-HF-path>
```

This evaluates the complete Earnings21, Earnings22 and CORAAL parent groups.
Segments run in timestamp order within each recording. The immediately
preceding response's transcript is passed as context in the next user prompt.
The assistant turn starts empty for fresh JSONL. The first segment has no
history. Histories reset between recordings, and recordings can run
concurrently. Only newly generated objects are scored. Saved
`asr_detail.context_prefixes` aligns with the raw `segment_outputs` for auditing.
The preset reserves 2048 prompt tokens for audio and context together and
raises an error on overflow.

The shared option `asr.context_aware=true` enables this behavior in both
training and validation. For training, use `+asr_training=context_aware` on
the normal training config and set a unique `trainer.experiment_name`. This
selects the offline multi-segment plan (one or two random segments per sample)
and reserves 2048 prompt tokens. Materialize it before launching training. Context stays within each sample's
ordered `asr_segments`; it never crosses unrelated or shuffled training
examples. Set `offline_data.random_segment.segment_count` to choose another supported
segment count, or enable the flag directly on an existing multi-segment config.
Single-segment samples have no preceding context.

For Earnings Short SC1, select
`--config-path config/v2611_dev/earnings --config-name remax_earnings_short_context_aware`.
This inherits the Earnings Short baseline data and training budget, enables
segmentation explicitly, and uses exactly two segments with a random cut per
full clip. Context is enabled for both training and validation.

Sampled rollouts and ReMax greedy baselines each build their own context from
their own generated outputs. The prefix belongs to the prompt for both the
rollout server and actor: only new response tokens retain response masks and
rollout logprobs and contribute to the policy loss. All segments share the
single parent-level outcome reward. The default remains independent decoding
(`asr.context_aware=false`).

The recipe selects the installed NVIDIA CUDA runtime matching PyTorch's CUDA
major version. The source branch's explicit-runtime helper is reused in the
Qwen general plugin because TileLang can load an incomplete `libcudart_stub.so`
that vLLM 0.24 would otherwise select for its sleep allocator.
The HF package initializes optional FlashInfer IPC before importing the model,
because spawned workers unpickle its config before vLLM loads general plugins.
`plugins/qwen35_audio/tests/test_cuda_import.py` exercises that spawn order and
checks both CUDA consumers against the installed CUDA major version.

Azure model downloads are cached atomically per node. Every FSDP rank uploads
its own checkpoint files before the driver publishes the latest-step marker.
The callback also uploads dataloader state; auto resume prefers the committed
remote step. Validation JSONL and metric JSON files are uploaded to the unique
experiment's `val_data_gen` directory.

Required quality gate: at every compared dataset, the reproduced error rate
may regress by at most **0.1 percentage point**. Missing datasets or mismatched
reference denominators do not count as a pass. Reference scores must use the
same normalization as reproduced outputs.

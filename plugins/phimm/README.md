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

Two segment variants can also be selected with `--config-name`:

- `remax_main_base3a_2611r2_sc`: `segment_count: [1, 2]`.
- `remax_main_base3a_2611r2_sl`: `segment_len: [5, 30]` seconds.

Both directly inherit `remax_main_base3a_2611r2.yaml`, disable duration cropping,
and use separate experiment names. For other training configs,
multi-segment training can be enabled with `+asr_training=multi_segment` and
an explicit `trainer.experiment_name=<unique-name>`. Each original recording
remains one dataset sample with its full reference. The `random_segment`
dataset block replaces its `audio_path` with an ordered list of selectors:

```yaml
# Inside a dataset's post_process block:
random_segment:
  segment_count: [1, 2]  # Inclusive segment count range.
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
lookup and always randomizes cut points. `min_segment_fraction` is an optional
count-mode-only floor relative to the full recording,
defaulting to `0.1` (10% of the full recording) and limited to
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

`data.random_segment` applies the block to every training source; a source's
own `post_process.random_segment` takes precedence, including `enabled: false`.
The opt-in config above sets this global default and disables duration
cropping. The standard per-segment prompt/response token limits still apply.
Without the opt-in config, a listed segment exceeding `max_audio_dur` fails
instead of silently cropping audio against a full reference. Validation is
not augmented by the global training option.

Prepare merged parents offline before training. Dataset preprocessing supports
loading prepared manifests and random segmentation. Use the preparation command:

```bash
python -m recipe.phimm.data.offline_random_merge \
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
each header may match any listed language, with matching `src` and `tgt`.
An independent response need not contain all reference languages. Mixed
responses check every header; unsupported or mismatched languages lose credit.
Single-language references retain their existing extra-header penalties.
For all references,
missing or malformed responses remain visible to these checks. Combined
metrics (`word_lang`, `char_lang`, `word_fmt`) use those averaged checks.
Only model response format 2609 is supported; omitted versions default to
2609 and other configured versions raise an error.
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

Use the curated [Docker image](../../docker/remax/README.md) and sync this checkout
to both nodes with `bpush`. The submitter prepares editable packages on every
GPU node, composes the config, checks topology and records a receipt before
submitting. It uses `--name` as both submission ID and default experiment name.
Use `--config <installed-name>` for another recipe or `--dry-run` to preview.

For an already synced two-node Ray pool:

```bash
python plugins/phimm/submit.py --name <unique-name>
```

Initial validation can be checked separately with `trainer.val_only=true`
and a separate experiment name. It does not replace the training run.
After the final checkpoint is exported and verified, full evaluation uses:

```bash
python plugins/phimm/submit.py --evaluate --name <unique-eval-name> \
  actor_rollout_ref.model.path=<verified-merged-HF-path>
```

The evaluation manifest retains the reference run's 35 datasets, including
`eslo_telephony_fr` and `accueil_ubs_fr` from its producer revision
`965a025ac220c91305aa762bdb176697b55e0fb5`. The current development branch's
generic manifest omits these two reference datasets.

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

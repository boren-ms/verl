# FY24Q3-1 long-form English

Last updated: 11/27/2026

<!-- FY24Q3-1 consecutive-duration filter statistics, verified 2026-11-27:
Zero duration d=0 and long duration d>5 seconds are counted separately per original sample.
Reject the whole sample at a same-category consecutive run >=4; runs 1-3 are allowed.
757,156 parents = 179,004 alignment failures + 160,338 consecutive-run rejects + 417,814 retained originals.
All 16,487 v2 accepted originals remain; 401,327 more are restored.
Retention of the same 757,156 original samples: v1 0.780949%; v2 2.177491%; v3 55.182023%.
Originals: 417,814 samples / 43772.3843872222 hours / 398,025,213 human words.
Segments: 417,814 parents / 5,388,363 units / 42420.2224533333 hours / 398,025,213 human words.
Pairs: 4,970,549 samples / 79705.4087700000 hours / 747,067,974 word occurrences.
Every original/raw/v1/v2 source, alignment timestamp and human reference is preserved.
All eight outputs passed independent validation and Orange SHA256 readback; sampled native audio passed. Three full filtered-original shards passed actual ASRDataset checks; remaining loader checks were explicitly waived by the user. No training or cropped audio.
-->

## Latest filter: allow 1–3 consecutive abnormal words (aligned_v3)

Within **each original sample**, classify words into two separate categories:
**zero duration (`d == 0`)** and **long duration (`d > 5` seconds)**.
A whole sample is rejected only when either category has **at least four consecutive words**.
Same-category runs of 1, 2 or 3 are retained, even if a sample has many separate runs.
A normal word or a switch to the other category breaks a run. Exactly 5 seconds is normal.
The normalized reference has no punctuation, so no sentence splitting is used.
Negative/nonfinite timings, incomplete coverage and pre-existing alignment failures remain excluded.
No Qwen inference is rerun and no timestamp, segment, human word or reference is altered.

| Version | Samples / units | Hours | Human words / occurrences |
| --- | ---: | ---: | ---: |
| Retained full originals | 417,814 samples | 43772.384387 | 398,025,213 |
| Word-timed segments | 417,814 parents / 5,388,363 segments | 42420.222453 | 398,025,213 |
| Sliding adjacent reference pairs | 4,970,549 pairs | 79705.408770 | 747,067,974 |

The new filter retains **417,814 / 757,156 original samples (55.18%)**,
including every previously retained v1 and v2 sample. Compared with v2, it restores
**401,327 samples** and **43510.157800 original-audio hours**.
There are 179,004 unchanged alignment failures and 160,338 run-based rejects:
158,450 zero-run only, 200 long-run only, and 1,688 both categories.
403,121 parents produce pairs; 14,693 retained originals
have fewer than two segments and remain in the retained-original and segmented manifests.
All 354,035 pair exclusions retain reasons and original timing flags.
Original hours are full parent waveform durations; segment hours sum timed selector spans.
Sliding pairs repeat segments, so pair hours and word occurrences include duplication.
Words are normalized human whitespace-delimited words, not tokenizer tokens.

### Retention comparison across policies

Each rate below uses the same **757,156 original samples**, including the unchanged
179,004 alignment failures. Counts are retained whole original samples, rather than
segment units or sliding pair rows. Every preceding version remains available.

| Policy | Rule | Retained original samples | % of all original samples | % of successfully aligned samples | Index and Orange directory |
| --- | --- | ---: | ---: | ---: | --- |
| v1 | Reject any word outside inclusive [0.04, 3] seconds | 5,913 | 0.7809% | 1.0227% | [v1 index](hc_fy24q3_1.aligned_v1.index.json); `aligned_v1/` |
| v2 | Reject at least two words with d <= 0 or d >= 5 anywhere in a sample | 16,487 | 2.1775% | 2.8517% | [v2 index](hc_fy24q3_1.aligned_v2_word_0_5_max1.index.json); `aligned_v2_word_0_5_max1/` |
| v3 | Reject a same-category consecutive run >=4 of d=0 or d>5; allow runs 1-3 | 417,814 | 55.1820% | 72.2672% | [v3 index](hc_fy24q3_1.aligned_v3_consecutive_max3.index.json); `aligned_v3_consecutive_max3/` |

The aligned-only denominator is **578,152 samples**. All directories share the Orange
`training/` prefix printed below. Each contains `shard-0000/` through `shard-0007/`, with
retained full originals in `filtered_originals.jsonl.gz`, timed segment rows in
`segments.jsonl.gz`, and sliding reference pairs in `pairs.jsonl.gz`.
Each index records the complete published paths and verified hashes.

### Consecutive-run distribution

The statistics below use all **578,152 successfully aligned original samples**, before this new filter.
An occurrence is a **maximal contiguous same-category run**, rather than overlapping windows of 1/2/3 words.
Run percentages divide by all runs in that category; word percentages divide by all abnormal words in that category.

| Category | Run length | Runs | % of category runs | Words in runs | % of category words |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero duration | 1 | 28,970,532 | 88.6372% | 28,970,532 | 70.8862% |
| Zero duration | 2 | 2,422,386 | 7.4114% | 4,844,772 | 11.8544% |
| Zero duration | 3 | 635,711 | 1.9450% | 1,907,133 | 4.6664% |
| Zero duration | >=4 | 655,779 | 2.0064% | 5,146,608 | 12.5929% |
| Duration >5s | 1 | 181,743 | 91.0659% | 181,743 | 80.3629% |
| Duration >5s | 2 | 12,273 | 6.1496% | 24,546 | 10.8537% |
| Duration >5s | 3 | 3,593 | 1.8003% | 10,779 | 4.7662% |
| Duration >5s | >=4 | 1,964 | 0.9841% | 9,085 | 4.0172% |

A sample can contain several runs, so run counts differ from sample counts.
This second table assigns each sample to its **maximum run length**, independently for each category.
Percentages use all successfully aligned samples; columns cannot be added across categories.

| Category | Maximum run in sample | Samples | % of aligned samples |
| --- | ---: | ---: | ---: |
| Zero duration | 0 | 6,104 | 1.0558% |
| Zero duration | 1 | 196,631 | 34.0103% |
| Zero duration | 2 | 141,173 | 24.4180% |
| Zero duration | 3 | 74,106 | 12.8177% |
| Zero duration | >=4 | 160,138 | 27.6983% |
| Duration >5s | 0 | 489,708 | 84.7023% |
| Duration >5s | 1 | 74,093 | 12.8155% |
| Duration >5s | 2 | 9,274 | 1.6041% |
| Duration >5s | 3 | 3,189 | 0.5516% |
| Duration >5s | >=4 | 1,888 | 0.3266% |

Full length histograms, word totals and denominators are retained in
[the v3 index](hc_fy24q3_1.aligned_v3_consecutive_max3.index.json), `RUN_STATISTICS.json` in each shard,
and the published root `RUN_STATISTICS.json` / `RUN_SUMMARY.json`.

### Preserved data and configurations

The v3 index links the unchanged original JSON, all eight immutable raw alignment shards,
[v1 index](hc_fy24q3_1.aligned_v1.index.json) and
[v2 index](hc_fy24q3_1.aligned_v2_word_0_5_max1.index.json), with source SHA256 hashes.
V1 and v2 manifests/configurations stay available. Large new data is retained locally in
`aligned_v3_consecutive_max3/shard-0000/` through `shard-0007/`, ignored by Git, and published under:

```text
az://orngwus2cresco/data/speech/users/keqideng/DataSpecs/human_caption_v3b/FY24Q3-1/WER_3600/training/aligned_v3_consecutive_max3/shard-000i/
```

Each shard contains `filtered_originals.jsonl.gz`, `segments.jsonl.gz`, `pairs.jsonl.gz`,
`quality_decisions.jsonl.gz`, `pair_exclusions.jsonl.gz`, `RUN_STATISTICS.json` and `COMPLETE.json`.
Decisions account for every original, retain previous filter decisions and record all outlier
words/timestamps, maximal runs and run-based rejection flags. Raw predictions and alignment failures
remain referenced by immutable source path and SHA256.

- [Filtered originals](hc_fy24q3_1_filtered_consecutive_max3.yaml).
- [Word-timed segments](hc_fy24q3_1_segments_consecutive_max3.yaml).
- [Two-segment reference pairs](hc_fy24q3_1_2segments_reference_context_consecutive_max3.yaml).
- [Reference-context training configuration](hc_fy24q3_1_reference_context_training_consecutive_max3.yaml).
- [Actual Hydra training entry](../../../v2611_dev/long/hc_fy24q3_1_reference_context_consecutive_max3.yaml).

The actual `ASRDataset` loader completed three whole published filtered-original shards
(shards 0000–0002; 156,584 samples), including newly retained maximum-zero-run-3 examples.
The user explicitly requested stopping the remaining loader checks and proceeding to delivery.
The complete segmented and paired versions were not loaded in this final probe, and no final
cross-shard schema-unification or simultaneous full-mix loader success is claimed.
All eight shards of the derived versions passed independent full parent/reference/timing/coverage/segment/pair validation
and exact Orange manifest SHA256 readback. The final Hydra configuration includes all eight paths,
uses human reference context with history one and generated validation, and leaves the model unset.
The earlier actual reference-context/full-agent QA remains available; reader/context/agent source is unchanged.
The native JSON reader builds Arrow tables in memory, so a training host must provide RAM for the
complete mixed dataset. The loader log, actual completed checks and explicit waiver are retained in
published `FINAL_LOADER_SCOPE.json` and the v3 index.
Full `ASRAgentLoop` import is unavailable locally because of existing `torchvision` / `transfer_queue`
dependencies; its unchanged implementation retains the earlier full-agent QA. This does not claim
a new full-agent import. A native Orange timed selector from a newly restored parent exactly matches
the existing full parent waveform slice at 16 kHz. This is sampled audio QA, not a hash of all audio.
No cropped audio or training/model-checkpoint launch occurs.

Print the composed configuration without starting training:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config \
  --config-name v2611_dev/long/hc_fy24q3_1_reference_context_consecutive_max3 \
  "hydra.searchpath=[file://${PWD}/verl/trainer/config]" \
  --cfg job
```

The checkpoint remains unset. Matching ASR model/tokenizer/audio processor/chat template,
registered model extensions and prompt/response/context/resource limits must meet the constraints
in the training section below before a separately authorized run.

Reproduce one shard from retained predictions, without GPU inference:

```bash
python scripts/refilter_consecutive_human_captions.py \
  --input <aligned_v1/shard-000i/raw_segments.jsonl> \
  --source-receipt <aligned_v1/shard-000i/ALIGNMENT_COMPLETE.json> \
  --output-dir <fresh-new-output-directory> \
  --maximum-long-word-seconds 5 --maximum-consecutive-words 3
```

<!-- FY24Q3-1 final processing statistics, verified 2026-11-26:
8 shards; 4,045 disjoint source chunks; 757,156 original/raw parents.
179,004 alignment failures + 572,239 whole-original word-duration rejects
+ 5,913 accepted originals = 757,156 accounted parents (0.78% retained).
5,913 segmented parent rows contain 12,591 timed segment units, each <=30s.
6,678 sliding adjacent pairs come from 3,878 eligible parents;
2,035 accepted parents have fewer than two segments; 753,278 parents excluded from pairing.
Reject the entire original if any selected word is outside inclusive [0.04, 3.0] seconds.
Originals, raw predictions, timestamps and rejection reasons are retained locally and on Orange.
All published manifests passed SHA256 readback and actual Hydra/ASRDataset checks;
sampled native Orange timed audio reads passed. No cropped audio or training launch.
Source: alignment_verified_aggregate_counts.json and alignment_final_deliverables_verified.json.
-->

All eight Qwen alignment shards and their Orange readbacks pass complete
parent/reference/filter/segment/pair validation: 757,156 parents from 4,045
source chunks. All four complete published versions also pass real Hydra/ASRDataset,
reference-context and native Orange timed-selector verification.
No training has been launched.

<!-- FY24Q3-1 relaxed word-duration statistics, verified 2026-11-26:
Normal word duration: strictly 0 < seconds < 5. Reject a whole parent at >=2
outlier words (duration <=0 or >=5); allow and record at most one outlier.
757,156 parents = 179,004 alignment failures + 561,665 duration-count rejects
+ 16,487 accepted originals. All 5,913 old accepted originals remain;
10,574 additional originals are restored. Accepted outliers: 6,030 zero / 10,457 one.
Originals: 16,487 samples / 262.2265869965 hours / 1,963,896 human words.
Segments: 16,487 parents / 38,148 units / 244.3730888889 hours / 1,963,896 words.
Pairs: 21,661 samples / 300.1109069444 hours / 2,347,714 word occurrences.
11,852 pair-eligible parents; 4,635 accepted parents with fewer than two segments.
745,304 pair-excluded parents. Pair hours/words include sliding-window duplication.
Raw predictions, timestamps and exact human labels unchanged; no audio files or training.
Source: hc_fy24q3_1.aligned_v2_word_0_5_max1.index.json and its verification receipts.
-->

## Refiltered version: strict (0, 5) seconds

The new filter allows **at most one abnormal word per original sample**. A word
is abnormal if its predicted duration is **<=0 or >=5 seconds**. Two or more
abnormal words reject the entire original. Alignment failures and incomplete
word coverage remain excluded. Allowed outliers retain their original word,
timestamps and explicit flags; no duration is clamped, no human word is dropped,
and no Qwen alignment is rerun.

| Version | Original strict filter | New filter | New hours | New human words |
| --- | ---: | ---: | ---: | ---: |
| Filtered full originals | 5,913 samples | 16,487 samples | 262.226587 | 1,963,896 |
| Word-timed segments | 5,913 parents / 12,591 units | 16,487 parents / 38,148 units | 244.373089 | 1,963,896 |
| Sliding adjacent pairs | 6,678 pairs | 21,661 pairs | 300.110907 | 2,347,714 |

Full-original audio grows from **83.805113 to 262.226587 hours**, adding
**178.421474 hours and 10,574 samples**. Retained human words grow from 606,599
to 1,963,896. The new set includes every previously accepted original. Of the
16,487 accepted originals, 6,030 have zero outlier words and 10,457 have one.
There are 179,004 alignment failures and 561,665 duration-count rejects, accounting
for all 757,156 parents. The 11,852 eligible parents produce 21,661 pairs;
4,635 accepted parents have fewer than two segments and remain in both preceding
versions. All 745,304 pair exclusions retain exact reasons and outlier timestamps.

Word counts are whitespace-delimited normalized human reference words, not
model tokens. Original hours use the previously decoded parent waveform duration;
segment hours sum native selector spans. Pair hours and word occurrences count
repeated segments from sliding pairing. No cropped audio is saved.

[The new shard index](hc_fy24q3_1.aligned_v2_word_0_5_max1.index.json) records all
counts, hours, words, policies, manifests and verification hashes. It links to the
unchanged original JSON and every existing raw alignment shard in `aligned_v1`.
The original JSON, old filters, raw predictions, model provenance and rejection
records are preserved. New large manifests are retained locally under
`aligned_v2_word_0_5_max1/shard-0000/` through `shard-0007/` and ignored by Git.

```text
az://orngwus2cresco/data/speech/users/keqideng/DataSpecs/human_caption_v3b/FY24Q3-1/WER_3600/training/aligned_v2_word_0_5_max1/shard-000i/
```

Each new shard contains `filtered_originals.jsonl.gz`, `segments.jsonl.gz`,
`pairs.jsonl.gz`, `quality_decisions.jsonl.gz`, `pair_exclusions.jsonl.gz` and
`COMPLETE.json`. Decisions cover every original and retain the previous filter
result. Raw alignment is referenced by path and SHA256 rather than duplicated.
All eight source/output hashes, original human references, parent IDs, disjoint
chunk sets, complete word coverage, positive <=30s segment spans, ordered pairs
and exclusions passed independent validation and exact Orange SHA256 readback.

The actual `ASRDataset` loaded all complete published new manifests through the
new Hydra configuration. The actual `ContextPrompt` builder verified empty first
reference context and the preceding human label for the second segment, with
reference training / generated validation / history one. Full `ASRAgentLoop`
import was unavailable in this local QA environment because of existing
`torchvision` and `transfer_queue` dependencies; its unchanged implementation
had passed the earlier full-agent QA. Two native Orange timed selectors from a
newly restored one-outlier parent exactly matched decoded parent slices: 423,680
and 367,360 frames at 16 kHz. This is sampled audio QA, not a hash of all recordings.
The QA-only Blob buffer was 16 KiB; shared packages and reader semantics were preserved.

New configurations:

- [Filtered originals](hc_fy24q3_1_filtered_word_0_5_max1.yaml).
- [Word-timed segments](hc_fy24q3_1_segments_word_0_5_max1.yaml).
- [Two-segment reference pairs](hc_fy24q3_1_2segments_reference_context_word_0_5_max1.yaml).
- [Composed reference-context configuration](hc_fy24q3_1_reference_context_training_word_0_5_max1.yaml).
- [Training entry](../../../v2611_dev/long/hc_fy24q3_1_reference_context_word_0_5_max1.yaml).

Print the new training configuration from the repository root in an ASR environment:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config \
  --config-name v2611_dev/long/hc_fy24q3_1_reference_context_word_0_5_max1 \
  "hydra.searchpath=[file://${PWD}/verl/trainer/config]" \
  --cfg job
```

The entry expects 21,661 training rows and leaves the ASR checkpoint unset.
The checkpoint, matching tokenizer/audio processor/chat template, registered
`hf_qwen35_audio` model extensions, prompt/response limits and resource settings
must follow the constraints documented below. This change does not launch training.

Reproduce one shard with the immutable alignment runner present:

```bash
python scripts/refilter_long_human_captions.py \
  --input <aligned_v1/shard-000i/raw_segments.jsonl> \
  --source-receipt <aligned_v1/shard-000i/ALIGNMENT_COMPLETE.json> \
  --output-dir <fresh-new-output-directory> \
  --minimum-word-seconds 0 --maximum-word-seconds 5 --maximum-outlier-words 1
```

## Original strict-filter version (aligned_v1)

| Version / configuration | Rows | Contents |
| --- | ---: | --- |
| `hc_fy24q3_1.yaml` | 757,156 | Original full audio selectors and exact normalized adjusted human captions |
| Raw `aligned_v1/shard-000i/raw_segments.jsonl` | 757,156 | Every original, word timestamps, model-window predictions, seams, alignment errors and quality reasons |
| `hc_fy24q3_1_filtered.yaml` | 5,913 | Accepted originals with full parent audio and human references |
| `hc_fy24q3_1_segments.yaml` | 5,913 | Accepted parent rows containing 12,591 ordered word-timed segments |
| `hc_fy24q3_1_2segments_reference_context.yaml` | 6,678 | Sliding adjacent pairs from 3,878 parents; exactly two ordered paths and own human labels |

There are 179,004 alignment failures and 572,239 whole-original word-duration
exclusions. These categories plus the 5,913 accepted originals account for all
757,156 parents. Of the accepted originals, 2,035 have fewer than two segments;
they remain in the original, filtered-original and segmented versions and are
excluded from pairing. `pair_exclusions.jsonl` retains 753,278 pair exclusions
with exact reasons and quality/outlier timestamps. Quality retention is about
0.78%; alignment and duration thresholds were not relaxed.

[The shard index](hc_fy24q3_1.aligned_v1.index.json) records verified counts,
SHA256 hashes, exact Orange locations, local shard paths and provenance. Large
original/raw/version files and generated receipts are ignored by Git. The local
original is `hc_fy24q3_1.jsonl`; all eight raw and derived shards are retained in
`aligned_v1/shard-0000/` through `aligned_v1/shard-0007/`.

```text
az://orngwus2cresco/data/speech/users/keqideng/DataSpecs/human_caption_v3b/FY24Q3-1/WER_3600/training/hc_fy24q3_1.jsonl.gz
az://orngwus2cresco/data/speech/users/keqideng/DataSpecs/human_caption_v3b/FY24Q3-1/WER_3600/training/aligned_v1/shard-000i/
```

Each shard retains `raw_segments.jsonl`, `filtered_originals.jsonl.gz`,
`segments.jsonl.gz`, `pairs.jsonl.gz`, `pair_exclusions.jsonl`,
`ALIGNMENT_COMPLETE.json`, `COMPLETE.json`, run/model-runtime receipts and the
owned-cache release ledger. The original plain manifest is 6,494,121,026 bytes,
SHA256 `84fd8a19498f19bf119778e85a29a65c7dcde05bcde3dc26c19336f22ab3ca1e`.

## References and original strict-filter selection policy

References are the exact normalized `tn_transcription` from locally read `.info`,
with `transcription_source=display_human_caption_adjusted`. Stored binary
`.transcription` text remains a source artifact; ASR hypotheses never replace
human captions. Source `.info` chunks were not transferred to Orange.

Reject the **entire original** if any selected word is below 0.04 seconds or above
3 seconds; the allowed bounds are inclusive. Keep all rejected originals,
timestamps and reasons in raw alignment and exclusion files. Alignment errors,
word mismatch, unreliable caption mapping and invalid/overlapping timings also
remain recorded. Forced alignment estimates timings; it does not prove the
transcript's truth or perceptual boundary quality.

Segments contain whole aligned words and span at most 30 seconds. Native
`chunk.audio:count:index#start:end` selectors resolve the parent record and then
slice its waveform in memory. No cropped audio files were saved. Internal model
windows are at most 180 seconds, anchored by `.info.segmentation_adjusted` and
unchanged human captions. Neighboring human context and ordered seams use actual
retained Qwen predictions; timestamps are never clamped or replaced by guesses.

Pairs are `(0,1), (1,2), ...` within one original. `audio_path` is an ordered list;
`extra_info.segment_reference_texts` holds each segment's own human label. The
first segment has empty label context; the second sees only the first segment's
human reference.

## Reproducibility and verification

The runner was frozen at Git revision
`229defa0e9c3884b779823f509dfb5fc6cb547f1`, SHA256
`f3d0f60f79045cd20ba4e033b188866dc35921461786f1b0426486d8d71b1f17`, seed 42.
Model: `Qwen/Qwen3-ForcedAligner-0.6B`, revision
`c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
Official Qwen3-ASR revision: `7c6daf77a2421100f5fb066495372c00129d39ff`.
The verified `qwen-asr==0.0.6` wheel has SHA256
`20dab74fb1c7599ae15cfb2488d36740f04f3125cf60f1670ae3a8bcd280a5e3`.
The isolated UV environment pins Transformers 4.57.6, Hub 0.36.2 and Accelerate
1.12.0, preserving shared Torch 2.11/cu130 and Ray 2.54. English-only Nagisa
handling is scoped to `scripts/english_alignment_shims`; optional automatic
package installation is disabled with `OAIPKG_DISABLE_META_MISSING=1`.

All received model files and installed Qwen source/dictionary bytes match pinned
receipts. All eight actual loaded runtimes compared tokenizer vocabulary/merges
and sampled parameter values against the pinned snapshot. These are sampled
in-memory parameter checks, not a hash of every in-memory tensor.

Whole chunks were assigned to disjoint shards. Each worker had one exclusive
owned cache root and released a chunk only after all its parent rows were
committed. The post-run audit verified 4,045 unique release entries totaling
16,032,004,867,660 bytes and zero remaining whole-chunk copies in those eight
worker caches. Source Orange inventory verifies exact names and sizes for
4,045 audio objects / 16,032,004,867,660 bytes and 4,045 transcription objects /
6,309,288,980 bytes. Audio blob contents were not all hashed. All published
alignment/version manifests and metadata have SHA256 readback verification.

All raw parent IDs, human references and source paths were compared with the
verified normalized inputs. Validation recomputed the original-level filter,
word coverage, segments, complete sliding pairs and exact exclusions; source
chunk sets are disjoint. The retained real 200-parent pilot accounted for
50 alignment failures, 147 duration rejects, three accepted parents and one
pair. Its accepted pair passed the real Hydra/ASRDataset/reference-context and
native timed-reader checks. Earlier QA-only rejected parents remain rejected.
The runner's 29 CPU checks passed; unrelated baseline repository hook failures
remain recorded in the task receipts.

Detailed receipts are in
`/home/boren/.codex/artifacts/fy24q3_1_transfer_20261124/`.
`alignment_full_collection_status.json` confirms all eight full-shard checks;
`alignment_verified_aggregate_counts.json` records final aggregate counts.
`alignment_final_loader_verified.json` records the real final-loader result; its
Orange readback SHA256 is
`17e7181dae2fffa2932d79befd13d03f6abc6044871f27b7ff3d4af61d6edcee`.
The accepted pair selectors produced 477,440 and 207,360 frames at 16 kHz,
exactly matching parent slices. All complete manifests were loaded; these two
audio reads are a sample, not a hash of every source recording.

## Reference-context training configuration

The training entry point is
[`config/v2611_dev/long/hc_fy24q3_1_reference_context.yaml`](../../../v2611_dev/long/hc_fy24q3_1_reference_context.yaml).
It composes the verified configuration below and checks for 6,678 training pair
rows. From the repository root in a configured ASR environment, print the
composed config without launching training:

```bash
python -m recipe.phimm.main_asr_remax \
  --config-path config \
  --config-name v2611_dev/long/hc_fy24q3_1_reference_context \
  "hydra.searchpath=[file://${PWD}/verl/trainer/config]" \
  --cfg job
```

`hc_fy24q3_1_reference_context_training.yaml` composes the actual Hydra ASR
configuration with paired training data and the existing OpenLong validation
source. It sets `training_source=reference`, `max_history=1`,
`validation_source=generated` and `data.max_audio_dur=null`.

A training run requires a compatible ASR checkpoint in
`actor_rollout_ref.model.path` (currently null), its matching tokenizer, audio
processor and chat template, and explicit trainer/resources settings. The base
selects `hf_qwen35_audio` and the `qwen35_audio` vLLM plugin; use the matching
registered model extensions. The forced-aligner checkpoint is for timing.
Ensure acoustic features, one previous reference and generated response fit
both configured prompt/response limits and the checkpoint's context limit;
the inherited 1,024-token prompt limit may need adjustment for 30-second audio.
The pinned alignment environment is separate from the ASR training environment.
No arbitrary checkpoint or training job is selected by this dataset task.

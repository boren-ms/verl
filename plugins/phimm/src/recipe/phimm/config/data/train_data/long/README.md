# FY24Q3-1 long-form English

Last updated: 11/26/2026

All eight Qwen alignment shards and their Orange readbacks pass complete
parent/reference/filter/segment/pair validation: 757,156 parents from 4,045
source chunks. All four complete published versions also pass real Hydra/ASRDataset,
reference-context and native Orange timed-selector verification.
No training has been launched.

## Preserved versions

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

## References and selection policy

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

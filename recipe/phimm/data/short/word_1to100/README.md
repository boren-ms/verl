# LibriSpeech 1–100-word dataset

This directory contains a planner, a separate plan auditor, and a WAV renderer.
Candidate plans themselves are not audio. The planning inputs are
Qwen3-ForcedAligner full-transcript outputs from `short_2to5_v1/aligned`;
model revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, model weights
SHA-256 `47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7`.
Word boundaries are estimated and do not guarantee phonetic completeness or VAD.

`plan.py` accepts only full alignments with matching transcript, one token per
alignment unit and strictly positive, monotonic word intervals; it records all
exclusion counts and input hashes. Seed 20261104 creates 100 examples for each
word count 1–100 in training. Each test split has 100 examples per ten-word
bin, balanced over its ten word counts. Sources are drawn randomly; each sample
has distinct sources; each test version uses any one recording at most once.
Test versions preferentially use sources with the fewest previous uses; overlap
between versions is recorded, never hidden. The per-test-bin count is a chosen
working target, as no number was specified. Random concatenation can cross
speakers and unrelated sentences.

Reproduction, from repository root (choose a fresh output directory):

```sh
python recipe/phimm/data/short/word_1to100/plan.py \
  --aligned-dir tmp/librispeech_concat_20261104/aligned \
  --output-dir tmp/librispeech_word_1to100_20261104/plans
python recipe/phimm/data/short/word_1to100/audit_plan.py \
  --aligned-dir tmp/librispeech_concat_20261104/aligned \
  --plans tmp/librispeech_word_1to100_20261104/plans
```

Rendering must occur in a fresh directory on a verified available remote node.
`render.py` validates each source's actual 16-kHz mono format and sample bounds,
records sample-accurate source and output offsets, writes PCM WAV and hashes,
and rejects out-of-range boundaries. Transfer failures interrupt the run.
It does not replace rejected candidates or assert dataset completeness. Verify
all resulting files, source/text consistency, per-version source uniqueness,
requested counts, duration distribution, alignment risk, and sample audio;
backfill if feasible without changing constraints. Upload only after verification
and read back remote manifests and sampled audio; check the whole inventory.

The Base0 default `data.max_audio_dur=40` truncates audio during loading; plans
include examples longer than 40 seconds. No duration cap was imposed. Choose
and validate the final configuration only after measuring the rendered audio,
including model input context and batch memory, and never silently truncate.

`verify.py` independently checks completed rendering against plan hashes and
IDs, manifest/rejection accounting, per-test-version source uniqueness, WAV
inventory and SHA-256, source sample ranges, and every WAV's PCM against its
concatenated source slices. It reports actual word counts, maximum duration,
counts over 40 seconds and cross-version source overlaps. It exits without a
success report on a mismatch. Successful checks establish sample consistency,
not that the aligner's estimated word boundaries avoid clipping phonemes.
For example, after a run use a fresh report path:

```sh
python -m recipe.phimm.data.short.word_1to100.verify \
  --plans /path/to/plans --rendered /path/to/rendered \
  --report /path/to/new-audit.json --workers 8
```

## Completed dataset, 2026-11-04

Final root: `az://orngwus2cresco/data/boren/data/LibriSpeech/word_1to100_20261104/`.
The dataset contains 9,982 training rows and 20 test datasets with 100 rows each.
Training counts per 10-word range from 1–10 through 91–100 are 1000, 999,
998, 1000, 997, 999, 998, 1000, 994, and 997. Each test group has ten samples
per word count. Eighteen training candidates whose estimated source word end
exceeded the physical file were rejected rather than clipped; records and reasons
are in `train/rejected.jsonl`. Training and test splits are disjoint, test source
recordings are unique within each version, and pairwise cross-version overlap
ranges 0–137 for clean and 0–150 for other; exact pairwise results are in
`provenance/merged_a1.audit.json`.

All 11,982 WAVs were checked locally on the task node against source PCM and
SHA-256, with original word boundaries and source paths recorded. After upload,
all final audio inventories and all 42 data/rejection manifests plus COMPLETE
were checked; one audio per dataset was independently downloaded and hashed.
These checks do not claim every remote audio byte was read back. The source
alignments, original plans, input hashes, model identity, merged audit, and
loader evidence are in `provenance/`.

The automatic edge audit flags 947 of 18,202 joins with an absolute consecutive
sample jump above 0.1 full scale; 8,236 of 60,368 segment edges have 10-ms RMS
above 0.05 full scale. These are inspection heuristics, not proof of clipping or
silence. `review/human_review_a1.jsonl` has a random and longest sample from each
of 21 datasets; `review/boundary_flagged_review_a1.jsonl` adds the highest flagged
join in each of 20 eligible groups. Their 62 distinct audio objects were all
read back and hashed. The samples have not been manually listened to; Qwen
alignment is an estimate and cannot guarantee phonetic completeness.

The example `recipe/phimm/config/v2611_dev/short/word_1to100/remax_base0.yaml`
uses the matching train and 20 val configs. Actual `RLHFDataset` and matching
Base0 tokenizer/processor loaded the longest audio in each group; the longest
is 54.24 s, with 718 prompt tokens (including 678 audio placeholders), below
the configured 60 s and 2048 tokens. `truncation: error` prevents silent prompt
truncation. PPO/reference per-GPU token budgets are 4096 and model length 2816.
The example reduces microbatch/validation batch for longer samples, but no GPU
training or memory-capacity test has been run. Batch size 64 with `drop_last`
leaves 62 of 9,982 training rows outside the final full batch per epoch.

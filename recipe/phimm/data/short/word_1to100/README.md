# LibriSpeech 1–100-word dataset work in progress

This directory contains a planner, a separate plan auditor, and a WAV renderer.
The candidate plans are not published or verified audio. The planning inputs are
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

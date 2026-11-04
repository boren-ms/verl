# LibriSpeech word-boundary concatenation

This dataset uses source transcripts and existing Qwen3-ForcedAligner-0.6B
word timestamps from the earlier `short_2to5_v1/aligned/` run. It does not
claim a new forced-alignment run was performed for this dataset. Model revision:
`c7cbfc2048c462b0d63a45797104fc9db3ad62b7`; model weights SHA-256:
`47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7`.
The original alignment files and hashes are retained under the dataset's
`provenance/` directory. Timestamps are model estimates, not guaranteed VAD or
phonetic boundaries; human listening is necessary to assess clipping or silence.

Training sources are limited to official train-clean-100, train-clean-360 and
train-other-500; evaluation sources come from test-clean or test-other only.
Source utterance IDs are disjoint between training and test. Sampling is with
replacement, including reuse of source utterances between different durations
within the same test split, so validation subsets are separate but not
source-disjoint or statistically independent.

Candidate seed is 20261104. Each training target is the midpoint of a 1-second
bin from [1,2) through [29,30), with 500 candidates per bin; validation targets
are 5, 10 and 20 seconds, 200 candidates each. The allowed target deviation is
0.5 seconds and every output is strictly under 30 seconds. A source must have a
full alignment with positive, non-overlapping word durations and a matching
transcript. Each output contains one or two word-bounded source windows; the
segments are sampled, never joined across train/test splits. Rejected candidates
and their reasons remain in `rejected.jsonl`; planned counts and actual outcomes
must be read from the generated reports and `COMPLETE.json`.

The planning script and provenance manifest describe candidate selection. The
rendering script loads original audio, validates source sample bounds, writes
16-kHz mono WAV, and records the source windows, text, sample offsets and SHA-256.
Source-boundary failures are rejected, not clamped. Transfer errors stop the
batch. The verifier checks all output audio and metadata before upload. The
first full render stopped on an out-of-range timestamp; it is retained separately
from the corrected batch and is not part of the published dataset.

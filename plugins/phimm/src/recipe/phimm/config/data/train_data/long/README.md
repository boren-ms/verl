# FY24Q3-1 long-form English

Last updated: 11/24/2026

`hc_fy24q3_1.yaml` loads the generated JSONL from Orange as `hc_fy24q3_1.jsonl.gz`.
The existing JSONL dataset loader supports gzip-compressed manifests.
The local JSONL is in this directory and is excluded from Git because it is a generated multi-gigabyte artifact.
`hc_fy24q3_1.preparation.json` records the row counts, source index hash, and output SHA-256.

The source partition has 4,045 chunks and 757,156 utterances. Its audio blobs occupy
16,032,004,867,660 bytes and its transcription blobs occupy 6,309,288,980 bytes.
Both chunk types are transferred through `orngtransfer/data/speech/` to:

```text
az://orngwus2cresco/data/speech/users/keqideng/DataSpecs/human_caption_v3b/FY24Q3-1/WER_3600/ChunkFiles/
```

Training references use `tn_transcription` from `.info`, requiring
`transcription_source=display_human_caption_adjusted`. The stored `.transcription`
text differs from this human caption. Metadata is read locally for preparation;
the Orange chunk transfer contains `.audio` and `.transcription` files.

Each row contains the existing `audio_chunk` locator (`file.audio:count:index`),
human reference text, source filename, original chunk index, and normalized WER.
`audio_length` comes from `.info.duration`; these metadata durations are distinct
from the aggregate durations in `file_set.json`. The preparation keeps the full
partition and its normalized human captions. Optional duration and WER filtering
is available through the builder's CLI arguments.

For full-parent training, set `data.max_audio_dur=null`, or use an ASR segmentation
configuration that pairs each segment with its corresponding reference. Load the
data config using `/data/train_data/long/hc_fy24q3_1@data.train_data`.

Reproduce the JSONL from a local source index and locally downloaded `.info` files:

```bash
.venv/bin/python scripts/prepare_long_chunk_jsonl.py \
  --file-set /path/to/file_set.json \
  --info-dir /path/to/info \
  --audio-root az://orngwus2cresco/data/speech/users/keqideng/DataSpecs/human_caption_v3b/FY24Q3-1/WER_3600/ChunkFiles/ \
  --output plugins/phimm/src/recipe/phimm/config/data/train_data/long/hc_fy24q3_1.jsonl
```

The current transfer's durable job states and sanitized logs are in
`/home/boren/.codex/artifacts/fy24q3_1_transfer_20261124/`.
`transcription_verification.json` and `audio_verification.json` record destination
file counts and sizes. `jsonl_complete.json` records the published training
manifest. Training can start after the audio verification and JSONL publication
finish. The recovery worker resumes failed transfers after the audio jobs end.

Validation: eight CPU conversion tests and Ruff checks pass. The full pre-commit
run passes its generated-config, type, compilation, and naming checks, and reports
existing documentation-date, license, and test-layout failures outside these changes.

## Qwen alignment and preserved versions

Preparation is active; alignment starts only after the audio and transcription
inventory and original human-caption manifest are verified on Orange.
The original `hc_fy24q3_1.jsonl` remains unchanged.

- Unfiltered alignment: each original receives `word_timestamps`, `segments`
  (path/text pairs), `segment_metadata`, and a `word_duration_filter` report.
  Alignment failures also remain in this raw JSONL.
- Filtered originals: full original audio and human text, plus alignment/quality
  provenance. No segments replace the parent audio.
- Filtered segments: accepted parents with an ordered `audio_path` list,
  `segment_reference_texts`, and retained `segments` pairs. Each segment spans
  at most 30 seconds and contains whole aligned words; outer silence is omitted.
- Reference context pairs: `(0,1), (1,2), ...` within each accepted parent.
  Exactly two paths and own labels per sample. Parents with fewer than two
  segments remain in the preceding versions and appear in pair exclusions.

Exclude an entire original from all filtered training versions if any word is
**below 0.04 seconds or above 3 seconds**. The bounds are inclusive. Unfiltered
raw rows retain offending word indices, durations, and reasons. Alignment errors,
word mismatch, overlapping/out-of-bounds times, and unreliable long-input
caption mappings cannot enter the filtered versions. Counts distinguish model
failures, quality exclusions, accepted parents, and generated pairs.

Example retained selector:
`az://orngwus2cresco/data/speech/.../chunk_UUID.audio:187:0#1.000:29.500`.
The existing audio reader resolves the chunk record, then the time selector in
memory. **No cropped WAV/FLAC files are written.**

Pinned model: `Qwen/Qwen3-ForcedAligner-0.6B` at
`c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
Official Qwen3-ASR source revision:
`7c6daf77a2421100f5fb066495372c00129d39ff` (`qwen-asr==0.0.6`).
Use an isolated UV environment with `transformers==4.57.6` and
`huggingface-hub==0.36.2`; retain all recorded package versions and runner hash.
Long recordings use existing `.info.segmentation_adjusted` timings to form
windows at most 180 seconds (the official inference helper's conservative
limit). Matching anchors map exact normalized human words to these windows;
text is never divided by guessed speech rate. Caption metadata is only read for
preparation; `.info` chunk files are not transferred to Orange.

```bash
.venv/bin/python scripts/align_long_human_captions.py prepare \
  --input plugins/phimm/src/recipe/phimm/config/data/train_data/long/hc_fy24q3_1.jsonl \
  --info-dir /home/boren/.codex/artifacts/fy24q3_1_transfer_20261124/info \
  --shards 8 --output-dir /path/to/new/alignment-inputs
# On reserved GPU resources, one process per GPU with its own input/output.
python scripts/align_long_human_captions.py align \
  --input /path/to/alignment-inputs/input-0000.jsonl.gz \
  --output-dir /path/to/new/aligned-0000
python scripts/align_long_human_captions.py pairs \
  --input /path/to/aligned-0000/raw_segments.jsonl \
  --output-dir /path/to/new/versions-0000
```

The `pairs` command writes `filtered_originals.jsonl.gz`, `segments.jsonl.gz`,
`pairs.jsonl.gz`, `pair_exclusions.jsonl`, and checksum/count metadata. Preserve
all eight raw shard JSONLs as well. Source sharding keeps entire audio chunks
in one worker to reuse the bounded chunk index. Resume refuses changed inputs,
runner/environment settings, corrupted JSONL, and completed outputs.
Run a real-audio pilot in a separate directory before processing full shards.

The new dataset YAMLs and `hc_fy24q3_1_reference_context_training.yaml` are
**prepared examples with publication pending**. The training example uses
previous-reference text for training and generated context for validation.
Supply the desired training model and resources. Before marking ready, publish
all expected versions, verify hashes/counts and actual time-selector reads, and
compose/test the real ASR dataset loader. No training job is requested or started.

Local checks: 24 CPU preparation/segmentation/filter/version tests passed;
repository-pinned Ruff, mypy, changed-file license checks, and `git diff --check`
passed. Hydra composition passed with recipe plus verl trainer config search
paths. An isolated native audio-reader check resolved an indexed chunk followed
by `#1:3` to the exact 32,000 expected frames at 16 kHz; local training package
initialization lacks Ray. Real Qwen and Orange loader validation remains pending
and is required before the new versions are marked ready. The broader repository
license scan also reports the existing `scripts/build_ls_speed_only_dataset.py`;
no unrelated files were changed to resolve baseline checks.

Automatic continuation is bound to this chat every 30 minutes by local cron:
`/home/boren/.codex/thread-schedules/01a20f4d-26aa-7b51-8ebb-efd322ec9694/fy24q3-1-align/config.json`.
The full executable instructions are in the sibling `prompt.txt`. The schedule
stops after transfers, alignment/filtering, preserved versions, publication and
actual loader/audio checks finish, or on cancellation. The local machine and
cron must remain running; installation/thread binding and scheduled continuation
delivery in this chat are verified.

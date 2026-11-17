# Offline Smart VAD segmentation

Prepare long-form recordings as ordered `audio_path` lists using the existing
PHIMM reader and full-reference multi-segment contract. One input recording
remains one output row. Original audio files, full transcripts, IDs, and other
parent metadata are preserved. No audio files are written.

## Output format

```json
{"audio_path":["/data/A.wav#0.0:1.0","/data/A.wav#1.1:3.0"],"text":"Full recording reference","svad_source":"/data/A.wav","svad_bounds":[[0.0,1.0],[1.1,3.0]],"svad_parent_duration":3.0,"svad_method":"svad"}
```

Times are seconds relative to the source recording. The existing time selector
is `file#start:end`. The syntax `file:count:index` has a different meaning: it
addresses samples in an indexed chunk. No new path syntax or reader is added.
The current `ASRDataset` expands the ordered list into `asr_segments` and keeps
one full parent reference for reward/evaluation.

## Install inference dependencies

Use an existing PHIMM environment, or install the plugin with its `svad` extra:

```bash
uv pip install -e 'plugins/phimm[svad]'
```

The local model is bundled. CPU inference needs no network access or GPU. The
existing PHIMM audio reader supplies local, remote, archive, chunk, and time
selector support; accessing remote inputs still needs the normal storage access.

## Prepare a manifest directly

Run from the repository root with the PHIMM environment's Python:

```bash
PYTHONPATH=plugins/phimm/src:plugins/qwen35_audio/src:. \
python -m recipe.phimm.tools.svad_segment \
  --src /data/long_form/input.jsonl \
  --dst /data/long_form/segmented.jsonl \
  --max-len-sec 30 --n-workers 4
```

The input must contain one scalar `audio_path` per parent. For a different input
field, add `--audio-key WavPath`; output always uses `audio_path` as a list.
Plain relative paths resolve against the manifest directory. Production sources
with selectors or URI schemes are used as supplied. Blank lines are ignored.

For an audio directory without a manifest:

```bash
PYTHONPATH=plugins/phimm/src:plugins/qwen35_audio/src:. \
python -m recipe.phimm.tools.svad_segment \
  --audio-dir /data/recordings \
  --dst /data/recordings_segmented.jsonl \
  --max-len-sec 30
```

Folder discovery is recursive and sorted. WAV, FLAC, OGG, MP3 and AIFF are
considered, subject to the installed libsndfile's codec support. No WAV export
option exists. Output must be a new local JSONL file. The tool preserves input
order across worker counts and publishes the manifest atomically after all
parents succeed. Unreadable/invalid audio aborts preparation with its source
path; partial manifests are removed. A final JSON summary reports parent,
segment and fallback counts. Each worker decodes a complete parent in memory.

## Use the existing offline preparation workflow

Configure exactly one enabled offline transform:

```yaml
offline_data:
  svad_segment:
    max_len_sec: 30
    min_seg_sec: 0.1
    num_threads: 1
    num_proc: 4
    empty_policy: fixed
```

Disable or remove other `offline_data.random_segment` / `random_merge` plans.
Then run the existing `recipe.phimm.tools.offline_audio` command with your
config. It writes the usual `source_NNN.jsonl`, `train_data.yaml`,
`training.yaml`, and `preparation.json`, including normalized-input and output
hashes. Raw path/text normalization happens before SVAD. Prompt/reference
formatting still happens when the resulting dataset is loaded. Validation
sources and training settings are preserved. Segmentation runs only offline;
training reads the saved lists without running SVAD again.

## Boundary behavior

- Audio is downmixed by the existing reader and resampled to 16 kHz only in
  memory for detection. Selectors address the original source audio.
- `max_len_sec` defaults to 30 and must be at least 0.2. Retained speech spans
  are bounded by that maximum; unusually long spans are split again. Context
  collars are clamped to the source and overlaps are removed so the same source
  samples are not decoded twice. Gaps identified as silence can remain.
- `min_seg_sec` defaults to 0.1. Shorter detected pieces are discarded. The
  inherited chunker passes recordings no longer than the maximum through as one
  span, including short silent recordings.
- If no span survives, `empty_policy: fixed` retains the parent with contiguous
  fixed windows and `svad_method: fixed_fallback`. Its final remainder can be
  shorter than `min_seg_sec`. Use `empty_policy: error` to reject these parents.
  Empty audio always fails. Empty lists are never emitted.
- The full reference stays in `text` (and any other original transcript field).
  This is the existing full-reference multi-segment format. It does not produce
  word-aligned per-segment transcripts.
- Inline `audio` arrays are removed from prepared rows so readers use the saved
  selectors. Other parent metadata is preserved. `svad_*` fields record the
  preparation provenance.

## Source and verification

Model, featurizer and chunker were ported from `/home/boren/code/verl-mirror`,
branch `boren/rl_2611_dev`, commit
`c6e1ce87f8ec9f44bee4003fe000f35e513cfeb8`, under
`recipe/phimm/utils/svad/`. Original Smart VAD attribution is preserved.
The port uses relative imports, explicit CPU thread limits, scalar hysteresis
thresholds, stable softmax, and handles empty audio / one probability frame.
`mel_filters.npz` is unnecessary because the featurizer builds its filter bank.

```bash
PYTHONPATH=plugins/phimm/src:plugins/qwen35_audio/src:. \
python -m pytest -q plugins/phimm/tests/test_svad_segment.py
```

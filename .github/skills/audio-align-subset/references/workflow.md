# Alignment, crop, and publication workflow

## Data contract

`audio_subset.py` reads UTF-8 JSONL. `id`, `text`, and `audio_path` must be
nonempty strings; `word_timestamps` is an array or JSON-encoded array of
`{"text":"word","start":0.32,"end":0.64}`. It also accepts `start_time` and
`end_time`. Times are absolute seconds within the referenced physical audio.
The planner rejects mismatched text, invalid or overlapping bounds, and a
window that crosses an alignment unit. It retains original text and source path.
`--start-word` chooses a zero-based initial word for every row, defaulting to
zero; per-row start positions and arbitrary time intervals require another
explicitly verified transformation.
The utility currently handles physical file paths, not time-range, ZIP, or
multi-source references; resolve those explicitly before using it.

Qwen's Python package and model have version dependencies. On this workspace a
working isolation was `qwen-asr==0.0.6`, `transformers==4.57.6` and
`huggingface-hub==0.36.2`; check compatibility rather than downgrading the shared
system environment. Qwen's package imports `nagisa` unconditionally. Only for
an explicitly English-only environment without DyNet, the repository provides
`recipe/phimm/data/short/english_shims/nagisa.py`; it raises if Japanese
segmentation is attempted. Never use that shim to claim Japanese support.

## Commands

The commands below are examples; substitute actual paths and allocated GPUs.
`$SKILL` is the directory containing this skill. Copies of the script run on
any machine with Python, soundfile, and, for az:// sources, this repository's
`verl.audio_cache` and its dependencies. Use local staging disks, not a slow
network-mounted directory, for intensive audio I/O.

```bash
python "$SKILL/scripts/audio_subset.py" shard --input input.jsonl \
  --output-dir input-shards --count 2 --seed 42

# On each GPU node, use its own input shard and unique output path. The Qwen
# runner writes alignment_error for failures and exits nonzero if any occurred.
python -m recipe.phimm.data.short.run_alignment \
  --input shard-0000.jsonl --output aligned-0000.jsonl \
  --model /path/to/Qwen3-ForcedAligner-0.6B --language English \
  --n-gpus 8 --batch-size 8 --workers-per-gpu 1 --io-threads 3

python "$SKILL/scripts/audio_subset.py" crop \
  --input aligned-0000.jsonl --output-dir crop-0000 \
  --min-words 2 --max-words 5 --seed 42 --workers 8 \
  --zero-duration flag --repo-root /root/code/verl \
  --audio-base az://ACCOUNT/CONTAINER/prefix/audio

# Rerun the same crop command with --resume for an incomplete shard.

# Stage the small manifest, run.json, COMPLETE.json, and rejection files from
# every node to one host. Upload the audio separately to its referenced paths.
python "$SKILL/scripts/audio_subset.py" merge \
  --inputs crop-0000 crop-0001 --output-dir merged
python "$SKILL/scripts/audio_subset.py" review \
  --input merged/data.jsonl --output review.jsonl --count 10 --seed 42
```

`shard` records input SHA-256 and per-shard counts in `COMPLETE.json`.
`crop` records input checksum and relevant settings in `run.json`. It keeps
successes and rejects in separate JSONLs and sets `COMPLETE.json` only after
all rows are accounted for. A crash may leave orphan audio files; a same-input
resume replaces only the tool's deterministic orphan filename. It validates
already recorded clip hashes, and refuses changed input, changed settings,
completed outputs, duplicates, and corrupted partial JSONL. `merge` checks
completion, settings and counts, and detects duplicate IDs; it does not check
remote audio bytes or upload anything. For local output omit `--audio-base`.

`--zero-duration flag` retains and flags selected zero-duration words;
`exclude` rejects them; `error` stops on the first. `--min-clip-seconds` can
exclude implausibly short clips if explicitly appropriate. Select the policy
separately for training and held-out validation. The default 10-item review is a
deterministic sample, not automatically stratified: add flagged examples when
quality review calls for them, and state how the sample was selected.

## Scaling and resource safety

One Qwen runner processes one node. For N nodes, create N disjoint shards and
submit one resource-reserved job per node with node affinity and the node's GPU
count; do not run two workers on the same GPU by accident. The runner defaults
to one model per GPU, batch 8, and up to three concurrent audio cache requests
per worker. Tune batch size, I/O threads, worker count and CPU reservation using
measured throughput, GPU memory, utilization, storage limits and job contention.
`workers-per-gpu > 1` loads additional model copies and can increase VRAM use;
it is not automatically faster. The cache uses per-file locks for concurrent
prefetch. The existing `recipe/phimm/cache_eval_audio.py` can prefetch ahead in
parallel; avoid doubling prefetch concurrency without measuring storage load.

During the 2026-11 LibriSpeech run, two 8-GPU jobs plus 24-thread per-node
prefetch handled disjoint shards. Observed throughput on one node increased from
roughly 20–30 to ~70 examples/sec after prefetch began; these were operational
snapshots, not a controlled benchmark. GPU utilization still varied and was not
continuously saturated. This is a starting point, not a performance promise.
An independent 1,024-clip cached-audio smoke on one Brix node took 10.40 s with
one crop thread and 9.59 s with eight; outputs matched by ID, text, bounds and
audio SHA-256. This single small run includes process startup and is not evidence
that eight threads are optimal for other storage or batch sizes.

This skill lives in the repository under `.github/skills/audio-align-subset`.
After verifying the pool and checkout, use `bpush` to sync the repository to
Brix and compare the local and remote script hashes. The `remote-development`
skill governs Brix access and non-code blob transfers. Preserve isolation and
use exact node-bound jobs, job IDs and output directories.

## Quality, review, and storage

Run a small real-audio end-to-end smoke on every environment before full work.
Compare selected text to the source prefix; inspect clips and alignment edge
cases, source duration, sample rates, zero-duration entries, and boundary
warnings. Automated checks cannot replace listening. Inspect the requested
number of review records from each split and link their accessible audio.

Remote publication should retain source/full alignment manifests, output and
rejection manifests, audio, configuration and audit evidence under an
independent destination. Verify the destination inventory against all intended
clip paths and read back and hash the manifests and review audio. For stronger
requirements verify every remote clip's content, not only its size or listing.
Write dataset YAMLs only against verified final manifests and route the correct
reward/data source. Re-compose the actual configuration and use the real loader
and at least one audio per split before calling the config ready.

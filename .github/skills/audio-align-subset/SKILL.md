---
name: audio-align-subset
description: Align existing speech transcripts to audio with Qwen3 ForcedAligner and create reproducible timestamp-based audio subsets. Use for word-timestamp alignment, short-prefix clips, dataset sharding, audit and human-review manifests; use ASR detail comparison for recognition-error reports instead.
---

# Audio alignment and subsets

Turn existing audio/transcript pairs into word timestamps and, when requested,
real cropped audio with matching text. Preserve dataset split provenance and the
user's selection policy. Do not equate forced alignment with recognition or
claim that timestamps guarantee correct boundaries.

## Decide the job

Inspect real input records and audio locations first. Establish source and target
splits, target size, selection/seed, word range or explicit crop rule, destination,
and the requested quality policy. Use independent train/test sources and check
for overlap. If the user leaves a nonessential value open, make and document a
reasonable choice; ask only when the choice materially changes the dataset.

Use the official [Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR)
and `Qwen/Qwen3-ForcedAligner-0.6B` unless another aligner is requested. Pin and
record model revision/hash, package versions, input hashes, selection seed, and
all failures. The model accepts a known transcript and returns estimated word or
character timestamps; it does not validate the transcript's truth.

For the available commands, metadata schema, remote GPU scaling, storage
verification and example invocations, read [references/workflow.md](references/workflow.md).
The reusable utility is [scripts/audio_subset.py](scripts/audio_subset.py); the
repository's Qwen runner is `recipe/phimm/data/short/run_alignment.py`.

## Execute and validate

1. Normalize the input to JSONL with unique string `id`, `text`, and `audio_path`.
   Verify paths and source split membership. Use the script's deterministic
   `shard` command to partition independent work; save its completion metadata.
2. Align each shard on an appropriately reserved GPU node, then use `crop` on
   each aligned shard. `crop` uses bounded, parallel I/O, writes physical FLAC
   or WAV clips, records source and selected timestamps, and can resume an
   incomplete run only if input and settings match. `merge` combines completed
   shard manifests but does not copy their audio. Use a fresh output location;
   never overwrite an unrelated dataset.
3. For prefix subsets, choose a reproducible 2–5-word target if that is the
   request; retain all available words if there are fewer than the target. Cut
   from the predicted first-word start through the chosen last-word end. This
   removes audio outside estimated boundaries but is not a VAD guarantee. The
   bundled word-window planner assumes whitespace-delimited words and checks aligned
   units against the transcript; use an explicitly defined language-appropriate
   segmenter for languages without whitespace-delimited words.
4. Check input/output/reject counts, clip duration and source boundaries,
   duplicate identities, alignment errors and zero-duration words. Decide
   whether to flag, exclude, or inspect questionable alignments according to
   the user and dataset purpose; do not silently discard validation examples.
   Report all denominators and retain exclusion/rejection records. Generate
   requested review JSONLs, include source text/paths/timestamps, and sample
   both ordinary and flagged examples when applicable.
5. Upload only when requested. Verify remote manifest content, audio inventory,
   and read-back hashes for an appropriate sample; check the actual training
   loader and configuration if YAMLs were requested. Do not claim every audio
   blob was content-hashed unless that was done. Give exact paths, counts,
   performance evidence, and unresolved quality limitations.

For Brix, use the `remote-development` skill and verify pool ownership, all
nodes, GPUs, Ray jobs, placement groups, processes and disk before reserving
resources. Do not disturb unrelated work or retry repeated infrastructure
failures without authorization. Never infer that an idle GPU is unowned. Do not
start training unless explicitly requested.

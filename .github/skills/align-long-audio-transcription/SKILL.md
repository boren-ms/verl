---
name: align-long-audio-transcription
description: 'Align one full long-audio reference transcript to ordered chunk-level ASR hypotheses and produce a validation-ready presegment JSONL/YAML. Use when: split a parent transcript across audio chunks, create per-segment references from `gts` and `clean_output`, align Earnings22 or other long recordings by `parent_audio_path` and `seg_index`, replace repeated parent references with chunk references, publish a presegment manifest, or configure verl evaluation for aligned long audio.'
argument-hint: '<input JSONL> [output JSONL or dataset name]'
---

# Align Long Audio Transcription

Partition a full recording transcript into lossless, monotonic per-segment
references by aligning it against ordered segment hypotheses. Produce a compact
dataset manifest and, when requested, matching verl validation and evaluation
YAML files.

This skill prepares reference data. Use
[asr-detail-compare](../asr-detail-compare/SKILL.md) for result comparisons or
recording-level WER without changing the dataset, and
[verl-asr-run](../verl-asr-run/SKILL.md) only when an evaluation run is requested.

## Expected Input

The input is JSONL with one row per audio segment. By default, each row has:

- `gts`: the same full parent transcript repeated across all segments.
- `clean_output`: the hypothesis for this segment only.
- `parent_audio_path`: stable parent-recording key.
- `seg_index`: zero-based segment order within the parent.
- `audio_path`, `id`, `language`, `n_segments`, `seg_start`, and `seg_end`.

If field names differ, pass the corresponding CLI overrides. Do not align rows
independently against the full reference; alignment must span all ordered
hypotheses for one parent recording.

## Workflow

1. Inspect the input before editing or generating data.
   - Parse every non-empty line as JSON.
   - Count rows and unique parent recordings.
   - Require one consistent full reference per parent.
   - Require unique `seg_index` values per parent and sort by that field.
   - Confirm hypotheses are segment-local rather than repeated full transcripts.

2. Run the bundled [alignment script](./scripts/align_long_audio_transcript.py)
   with the configured project interpreter (`/home/boren/.virtualenvs/openai/bin/python`
   when available):

   ```bash
   /home/boren/.virtualenvs/openai/bin/python \
     .github/skills/align-long-audio-transcription/scripts/align_long_audio_transcript.py \
     INPUT.jsonl OUTPUT_aligned.jsonl
   ```

   Useful overrides:

   ```bash
   --group-field parent_audio_path \
   --segment-index-field seg_index \
   --reference-field gts \
   --hypothesis-field clean_output \
   --aligned-reference-field gts \
   --parent-reference-field parent_gts
   ```

3. Understand the alignment behavior.
   - Concatenate hypotheses in segment order while retaining each hypothesis
     word's segment owner.
   - Canonicalize words with Unicode NFKC, case folding, and punctuation removal.
   - Globally align reference and hypothesis words with `kaldialign`.
   - Assign each aligned reference word to the owner of its hypothesis word.
   - Distribute deleted reference spans monotonically between neighboring segment
     owners.
   - Write each segment reference to `gts` and preserve the original full
     transcript in `parent_gts`.

4. Create a compact validation source manifest when the user needs a dataset.
   Keep only:

   ```text
   url, transcript, id, language, parent_audio_path, seg_index,
   n_segments, seg_start, seg_end
   ```

   Map `audio_path` to `url` and aligned `gts` to `transcript`. Remove generation
   and stale scoring fields such as `input`, `output`, `clean_output`, `score`,
   `reward`, `wer`, `n_err`, `n_ref`, `step`, `prefix`, and `keywords`.

5. Publish only when remote workers must consume the dataset or the user asks
   for upload. Use `bbb cp` to a stable `az://orngwus2cresco/...` path and verify
   the exact object with `bbb ls`. Keep a validated local copy and report both
   paths. Never print credentials or SAS query strings.

6. When a dataset YAML is requested, copy the nearest established config,
   usually `recipe/phimm/config/data/val_data/earnings_chunked.yaml`.
   - Map `audio_path: url` and `text: transcript`.
   - Preserve `task`, language, and prefix behavior requested by the user.
   - Use a distinct `data_source` for the aligned dataset.
   - Include segment metadata under `verl_format.extra_keys` when downstream
     aggregation needs it.

7. Create an evaluation YAML only when requested.
   - Start from the matching model/version config.
   - Replace the validation dataset default with the new dataset YAML.
   - Map the exact new `data_source` to the intended reward function.
   - For the built-in validation manager, set
     `val_reward.reward_manager: naive` (`naive` is the registered name).
   - Preserve the original model checkpoint and unrelated evaluation settings.

## Decision Points

- If rows do not repeat one consistent parent reference, stop and determine the
  actual reference source before alignment.
- If segment indexes are duplicated, fail rather than silently choosing an order.
- If hypotheses are absent for every segment, text-only alignment cannot infer
  reliable boundaries. Ask whether to run an audio forced aligner instead.
- If only a local artifact is requested, do not upload it or create Azure paths.
- If a YAML is for remote evaluation, do not reference a local `tmp/` path;
  publish the manifest first.
- If the input is pretty-printed JSON objects rather than JSONL, decode it as a
  JSON object stream and rewrite the derived artifact as one object per line.

## Completion Checks

For every parent recording:

1. Concatenate aligned segment references in `seg_index` order.
2. Compare `.split()` tokens with the original full transcript.
3. Require exact equality so no reference word is lost or duplicated.

For the final manifest:

- Parse every line with `json.loads`.
- Require the expected row count and exact field set.
- Require non-empty `url` values and string `transcript` values. A segment may
  legitimately receive an empty reference (for example silence or hallucinated
  speech); flag it for review, but never invent words or drop it silently to
  satisfy a non-empty check.
- Confirm parent count and segment-index uniqueness.

For YAML files:

- Parse with `yaml.safe_load`.
- Verify rename mappings against a real manifest row.
- Compose the final Hydra evaluation config and assert the dataset source,
  reward mapping, reward manager, and checkpoint path resolve correctly.

## Deliverables

Report clickable absolute local paths for:

- The aligned JSONL containing per-segment references.
- The compact validation source JSONL, if created.
- The dataset YAML, if created.
- The evaluation YAML, if created.

Also report row count, parent count, validation results, and any remote blob URL.
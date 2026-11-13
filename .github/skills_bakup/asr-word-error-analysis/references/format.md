# Alignment Format & JSONL Schema

## Alignment Notation

The alignment uses standard Levenshtein word-level edit distance. Each word position is labeled:

- **(space)** – correct match
- **S** – substitution (ref word ≠ hyp word)
- **D** – deletion (ref word missing in hyp)
- **I** – insertion (extra word in hyp not in ref)

Example from `alignment_samples.txt`:

```text
REF: the   cat   sat   on    the   mat
HYP: a     cat   ***   on    the   mat   today
OPS:  S           D                       I
```

In `error_details.csv`, the `alignment_ops` column uses a compact format:

```text
S(the/a) cat sat D(on/-) the mat I(-/today)
```

Correct words appear bare; errors appear as `OP(ref_word/hyp_word)`.

With `--normalizer openasr_en`, compound matches are grouped into one alignment cell while the original normalized reference-word count is retained for WER. For example, `ice cream` and `icecream` align without an error, but the reference denominator remains two words, matching `openasr_en_eval`.

## HTML Error Context

The HTML report shows five equal words on the left and right of each error by default. Overlapping error contexts are merged, and each hidden equal run is represented by one ellipsis. Use `--context-words` to change the context size. Each utterance also provides a **Show full alignment** disclosure.

Only HTML is condensed. `summary.json`, all CSV artifacts, and `alignment_samples.txt` retain complete alignment data.

## JSONL Input Schema

Each line of the input JSONL must have at least:

| Field | Required | Description |
| --- | --- | --- |
| `ref` | Yes (or `--ref-column` override) | Reference transcription text |
| `hyp` | Yes (or `--hyp-column` override) | Hypothesis transcription text |

For verl `val_data_gen` files, the script auto-remaps `gts` to `ref` and `clean_output` to `hyp` when the requested columns are missing. The original `output` column can be shown in HTML reports with `--raw-output-column output`.

When rows contain `parent_audio_path`, `seg_start`, and `n_segments`, the script verifies that every recording group is complete and has one repeated full reference. It then orders rows by `seg_start`, joins segment hypotheses, and analyzes one row per parent recording. The report links the parent audio rather than unsupported `#start:end` fragment paths.

### Optional ID fields (auto-detected in this priority order)

1. `audio_file_stem` (derived from `audio_file` if present)
2. `audio_file`
3. `utt_id`
4. `utterance_id`
5. `example_id`
6. `item_id`
7. `segment_id`
8. `id`
9. `key`
10. `audio_path`

### Nested `meta` promotion

If a row contains a `meta` dict, these fields are automatically promoted to top-level columns: `audio_file`, `audio_path`, `audio_length_s`, `dataset`, `duration`, `id`, `sampling_rate`, `text`.

### Example JSONL

```json
{"id": "utt_001", "ref": "the cat sat on the mat", "hyp": "a cat sat on the mat"}
{"id": "utt_002", "ref": "hello world", "hyp": "hello world"}
{"meta": {"audio_file": "/data/audio/003.wav", "id": "utt_003"}, "ref": "good morning", "hyp": "good moaning"}
```

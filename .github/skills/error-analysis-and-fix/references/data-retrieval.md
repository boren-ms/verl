# Source discovery and reproducible retrieval

## Lineage before copying

Search effective configs, DataSpec aliases, source builders and prior report manifests with `rg`. Follow references recursively: a JSON may list other JSON files, ZIPs, chunks, or corp storage paths rather than utterances. If local evidence is incomplete, inspect the related source codebase (for example MOE) and read bounded remote listings/manifests, including pagination. Verify account/container mappings instead of assuming similarly named mounts point to the same storage. Keep credentials and signed URLs out of artifacts.

For each source record its URI/version, parent manifests, split, transcript origin, IDs, count/duration and hash. Establish actual training inclusion from resolved configs and reader behavior; a source file on storage does not prove a checkpoint saw it. Preserve human references unless label correction is explicitly part of the task, with separate provenance for every correction.

Use a source adapter rather than rewriting the training format into a guessed schema:

- JSONL/parquet: discover real reference/audio/ID columns and retain loader fields.
- ZIP: preserve archive/member or byte-range selectors and the requested ZIP DataSpec style.
- Binary chunks: inspect the writer and reader, including header case, sequential records and UTF-8 metadata. Preserve parent IDs and byte/record selectors; verify a real audio record can be read.
- Long recordings: group segments by parent before splitting; report segment and parent coverage.

Use the applicable installed transfer skill for copies. Prove the source listing and a sample copy before a large recursive transfer. Copy only fields/files needed for the downstream loader.

## Pattern definitions

A pattern should specify version, semantic description, searched fields, normalization, match rule, positive/negative examples, false positives and evidence type. A hypothesis-only feature needs a decode; an acoustic feature may need an external classifier or manual review. Save those derived annotations with their provenance. The helper below deliberately supports only regexes on explicit scalar fields; complex detectors should emit auditable annotations through a separate adapter.

Example for the helper (regex uses Python Unicode semantics; text is searched as supplied):

```json
{
  "version": "case-lexemes-v1",
  "mode": "any",
  "patterns": [
    {"id": "example-lexeme", "field": "text", "regex": "示例词"}
  ]
}
```

Keep semantic examples and matcher normalization in the case ledger or an accompanying definition. For confusions, pair evidence can be a derived scalar annotation, not an arbitrary regex on the reference alone.

## Local selection helper

`scripts/select_subset.py` selects from a canonical JSONL adapter; it does not decode, score, copy audio, create DataSpecs, validate audio or submit jobs. Use it after the adapter and canonical scorer are verified. Preserve original loader fields in the adapter and generate the repository's native DataSpec separately.

Required source fields:

| Field | Contract |
| --- | --- |
| `id` | Unique nonempty string, namespaced when corpora reuse IDs |
| `text` | Human reference string; may be empty only if valid for this source |
| `audio_path` | Nonempty string identifying actual audio, not just its containing ZIP/chunk |
| `source_split` | `train`, `dev`, `validation`, or `test` |
| `parent_id` | Optional parent recording identity, namespaced across sources |
| `source_row` | Optional nonnegative integer for deterministic ranking ties |
| other fields | Retained verbatim, e.g. byte range, language, duration, audio hash |

If the native audio path denotes a container, the adapter must build an utterance selector including member/record/byte position; preserve the separate native fields for loading. Canonical string identities cannot detect aliases or re-encoded copies: record that limitation and add audio-content hashes or corpus-specific checks upstream.

For error ranking, supply a **complete** score JSONL for the declared source universe. Each score row must have identical `id`, `text`, `audio_path`, a string `hypothesis`, and nonnegative integer `n_ref`, `n_err`, `n_ins`, `n_del`, `n_sub`; `n_err` must equal the three edit counts. Supply `--scorer-id` identifying the pinned canonical scoring implementation and settings. The helper checks joins/counts, not the correctness of the scorer's metric. Absolute `n_err` can rank long examples higher; that is intentional for an error-count request.

```bash
python .github/skills/error-analysis-and-fix/scripts/select_subset.py \
  --source source_train.jsonl --expected-source-count 120098 \
  --scores complete_scores.jsonl --scorer-id pinned-canonical-scorer \
  --rank errors --min-errors 1 --top-k 20000 --role train \
  --heldout full_dev.jsonl full_test.jsonl --output-dir selected_train
```

Those counts are illustrative. Pass the count verified for this case. For reference-feature retrieval use `--rank deterministic --patterns patterns.json --seed case-v1`; output is labeled as candidates. Patterns on other adapter annotations produce `feature_candidate`, without assuming those annotations prove a recognition failure. Supplying scores and patterns yields model-confirmed feature matches only when the error threshold is positive; zero-error selections remain candidates in the audit.

Training selection requires heldout manifests and canonical `train` rows. Declare heldout scope in the case ledger (e.g. full dev and test plus guards); one small heldout manifest is not proof of global leakage freedom. The helper excludes exact ID/audio/parent matches; optional `--exclude-same-text` excludes whitespace-collapsed reference matches and may be unsuitable for repeated read speech. Validation selection accepts only dev/validation; test hard cases use `--role diagnostic`. Do not reuse a training candidate universe as a heldout diagnostic.

Outputs are a new directory containing `manifest.jsonl` (unchanged source rows), `audit.jsonl` (rank, feature matches, evidence type and optional scores/hypotheses), and `summary.json` (hashes, counts, exclusions, missing parent coverage, selection settings and membership hash). No silent padding: insufficient eligible rows, duplicate IDs, incomplete scores, incompatible labels/selectors and wrong splits fail before publishing output. An expected count plus matching IDs only validates the supplied universe; verify its lineage upstream before claiming full-corpus coverage.

Run the local synthetic invariant checks with `python scripts/test_select_subset.py` from this skill directory. They need no audio, cloud credentials or GPU.

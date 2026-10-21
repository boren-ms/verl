# Validation Examples and Fixing Directions

Use the actual table and matching validation dumps for every requested run.
This workflow supports both multi-run and within-run checkpoint comparisons.

## Retrieve and flatten tables

The current W&B key is **`val/generations`**, implemented by
`ValidationGenerationsLogger` in `verl/utils/tracking.py`. Resolve history or
summary references to their exact `media/table/*.table.json` run file, or to the
referenced artifact/version. If necessary, enumerate `run.files()` and inspect
the schemas/steps of candidate table files. Prefer a supplied local export when
available. Preserve run ID, file/artifact version, local path, and step coverage.
Report missing/disabled tables explicitly and use matching dumps when available.

Current table files contain `columns` and `data`, for example:

```json
{"columns":["step","input_1","output_1","score_1","ground_truth_1"],
 "data":[[40,"transcribe audio","model output",-0.2,"reference text"]]}
```

Flatten by column names and numeric `output_i` suffix, preserving step, slot,
input, output, score, optional ground truth, and file/row provenance. Older
tables can omit references; some exports have one sample per row. Distinguish
empty text from missing fields and surface malformed/non-finite observations.

Each update contains cumulative rows from that logger instance. Prefer one
verified snapshot covering the requested steps. When combining snapshots/resumes,
deduplicate identical observations and preserve conflicting versions rather than
counting cumulative copies as new cases.

`RayPPOTrainer._maybe_log_val_generations` sorts all validation samples by
`(score, input)` ascending and takes `trainer.log_val_generations` examples;
zero disables logging. These are a selected subset across datasets. Verify score
direction and scale before calling them worst cases. Different scorers can bias
selection. Slots are ranks, not persistent IDs. The current table has no explicit
dataset, language, audio path, sample ID, or component columns.

## Enrich and match identities

Resolve the run's actual `trainer.validation_data_dir`; `_dump_generations` in
`verl/trainer/ppo/ray_trainer.py` writes:

```text
<validation_data_dir>/<data_source>/<step>.jsonl
```

Common fields are `input`, raw `output`, `gts`, `score`, `step`, and `data_source`;
optional fields include `clean_output`, reward/error counts, and flattened
`extra_info` such as `id`, `audio_file`, language, and segment metadata. Confirm
actual files and steps. Use existing authenticated blob tooling for `az://`
sources; do not substitute a benchmark dump from another run/checkpoint.

Join each table example to its same-step dump using available input, reference,
raw output, and score together. Table `ground_truth` corresponds to dump `gts`.
Check uniqueness and retain ambiguity. Across steps/runs prefer unique stable
`(data_source, id)` or audio path plus segment boundaries. Confirm unchanged
reference and decoding multiplicity. Dataset+prompt+reference is a fallback only
when unique; generic prompts, rank, row order, and random UIDs are not identity.

Fetch counterpart outputs from the full dump even when not selected in the other
table. Report matched, baseline-only, target-only, and ambiguous counts per
dataset. Disappearance from a selected table is not recovery. If neither IDs nor
counterparts can be recovered, report cross-sectional patterns and leave change
unknown. Cover missing table datasets from full dumps; a table with no examples
from a dataset does not establish that dataset has no errors.

For grouped long audio, check `recipe/phimm/reward/long_audio_grouped.py`:
chunks are grouped by dataset/parent, ordered by `seg_start`, concatenated, and
scored once. The parent score can be broadcast to chunks while error counts exist
only at the group head. Group by stable `parent_audio_path` (or verified equivalent),
check complete unique segments and consistent references, and count each parent
once. Null count fields are not zero. Never compare one chunk with a full-parent
reference or count broadcast scores as independent failures.

## Identify patterns and estimate their impact

Inspect raw output for tags/format and cleaned output for content. Prefer logged
alignments/error counts and the run's actual scorer/normalizer. Repeated
normalization can change digits or remove content. Inspect discarded spans when
large deletions may come from parentheses or malformed angle tags, especially
after long-audio concatenation. Diagnostic rescoring must be separate, use the
same alternative convention on all compared references/hypotheses, and disclose
scorer/denominator changes. An empty output can be correct for non-speech.

| Pattern | Concrete evidence | Useful direction when supported |
|---|---|---|
| Substitution/deletion/insertion | Matched reference/output spans and counts | Inspect affected vocabulary, domain, language, or data coverage |
| Digits/entities/keywords | Number/name changes, entity or keyword components | Targeted examples or coverage checks; verify spoken/display normalization |
| Repetition/tail hallucination | Unsupported repeated spans/suffixes, detector output | Check decoding termination and reward behavior; verify reference/audio repetitions |
| Empty/truncated output | Nonempty speech reference, missing suffix, finish/token evidence | Check extraction, chunking, response limit, and generation termination |
| Language/format failure | Raw tags, expected schema, parse failures | Separate parser/header defects from transcription language errors |
| Punctuation/capitalization | Raw differences and configured scoring treatment | Verify task convention and normalization before altering reward weights |
| Reference/scoring/data issue | Reproducible extraction mismatch or verified reference/audio defect | Fix the demonstrated data/scorer issue, then rerun the same comparison |

Use multiple labels when warranted and disclose overlap. Report counts and
denominators for selected-table, matched-cohort, or full-dataset scope. Include
one or two short examples per important pattern, with dataset, stable identity,
run/step, score/counts, and table row/slot or JSONL line. Text alone cannot prove
bad audio, acoustic causes, or the language spoken; mark those as hypotheses.

Define any bad-case threshold. For matched examples distinguish newly failed
(crosses that threshold), worsening persistent, unchanged persistent, and
improved cases. A single snapshot establishes presence, not emergence. With
full comparable counts and unchanged references, show added errors
`target n_err - baseline n_err`. For a complete fixed single-decode corpus with
unchanged total `n_ref`, dividing added errors by total `n_ref` estimates a
pattern's contribution to the dataset's `p_err` change. Do not sum overlapping
labels or infer corpus contribution from selected-table counts.

Prioritize frequent or high-impact patterns corroborated by per-dataset `p_err`
and matched examples. If a spike is dominated by one recording, quantify its
share while retaining it in official metrics. If dataset metrics are stable,
describe persistent tail errors without calling an overall regression. Tie each
fix suggestion to a falsifiable diagnostic or one-factor experiment, affected
datasets, and guardrails; reward correlations alone do not establish a cause.

# Validation Bad-Sample Analysis

Use this workflow for training validation logging tables, recurring bad cases,
or validation evidence near a training reward drop. It also works without a
training log. Analyze existing artifacts; no new training or evaluation job is
needed.

## Resolve the run and validation timeline

Confirm the run ID/URL, host, experiment, effective validation datasets, reward
dispatch/scorer, decoding settings, and validation cadence. Inspect the code
revision used by the run when available. Current logging behavior is defined in
`RayPPOTrainer._maybe_log_val_generations` and `_dump_generations` in
`verl/trainer/ppo/ray_trainer.py`, and `ValidationGenerationsLogger` in
`verl/utils/tracking.py`; older runs may differ.

For each important training drop at step S, select the last validation before S,
the first at/after S, and the next available validation to check persistence.
Record actual steps and gaps; a table at step 120 does not establish what
happened at training step 113. For a validation-only request, compare the
requested steps or available adjacent snapshots. Keep pre-training step 0 and
resumed segments distinct; disclose missing baselines.

Use unsampled `run.scan_history(page_size=1000)` for aggregate trajectories.
Avoid a union of sparse train/validation keys in `scan_history(keys=...)`, which
can exclude rows missing any requested key. Preserve `step` and `_step`; verify
their mapping against the logger. The table's own `step` column identifies its
sample rows. Do not assign the run summary's latest step to historical data.

Track each dataset's exact `val-core/<dataset>/p_err/mean@1` key when present,
and relevant `val-core`/`val-aux` reward and component keys. Discover actual keys
rather than assuming a namespace or sample count. Compare the same dataset,
scorer/normalization, metric, and `@N` across steps. Reward/accuracy decreases
and error-rate increases are adverse; verify `score` semantics from the scorer.
Report fraction versus percent units, absolute change (percentage points for
rates), and relative change separately. Disclose any validation-specific
threshold; the training analyzer only detects decreases and requires aggregate
training reward records, so it cannot directly analyze validation-only history
or increasing error rates.

## Obtain the actual validation table

Use an existing local export when provided. Otherwise use the run's configured
W&B host and existing authentication for read-only access. For this checkout
the usual host is `https://msaip.wandb.io`; verify it from the run URL. Do not
dump credentials or whole configs, switch hosts, or disable TLS verification.

The logged key is **`val/generations`**. A UI panel titled `val` may refer to
that table; inspect the key rather than assuming an artifact name. Table
references can appear in history or summary and contain a `path` to a
`media/table/*.table.json` file. Download that exact run file, or resolve the
referenced table artifact/version if the reference uses an artifact URI. If
necessary, enumerate `run.files()` for table files and inspect their schemas
and steps before selecting one. A file listing or screenshot alone is not the
complete table data. Record the run, original file/artifact version, local
path, and captured step coverage. Report access or missing-file errors.

Current W&B exports have `columns` and `data`:

```json
{
  "columns": ["step", "input_1", "output_1", "score_1", "ground_truth_1", "input_2", "output_2", "score_2", "ground_truth_2"],
  "data": [[20, "prompt A", "response A", -0.4, "reference A", "prompt B", "response B", 0.0, "reference B"]]
}
```

Flatten by column names, not positional assumptions: find numeric `output_i`
suffixes and map each row/slot to `step`, `input`, `output`, `score`, optional
`ground_truth`, and provenance (file, row number, slot). Retain raw values and
distinguish missing fields from empty strings. Older tables can omit ground
truth; other logger exports may already have one sample per row. Validate row
widths and numeric steps/scores, and surface malformed/non-finite entries rather
than silently coercing them to zero.

Each W&B update contains prior table rows accumulated in that logger instance.
Prefer one verified snapshot covering the needed steps. If combining versions
or resumed sessions, deduplicate identical observations while preserving
provenance; flag conflicting versions at the same step/slot. Do not count every
cumulative copy as a new sample or assume a latest snapshot spans all resumes.

Current selection sorts all validation samples by `(score, input)` ascending
and retains `trainer.log_val_generations` entries (often 20; zero disables the
table). Slots are ranks, not persistent sample IDs. The table has no dedicated
dataset, language, audio path, ID, or component-score columns. Do not invent
those fields from slot order or generic prompts. Mixed scorer scales can skew
selection toward a dataset; some scorers even return positive error counts as
`score`, making ascending order select fewer errors. Confirm direction before
calling all selected rows the worst cases.

## Enrich and match samples across steps

Resolve `trainer.validation_data_dir` from the effective config; it is commonly
`<trainer.default_hdfs_dir>/val_data_gen`. List the actual dataset directories
and verify exact numeric step files before fetching:

```text
<validation_data_dir>/<data_source>/<step>.jsonl
```

Use the training run's dumps, not a separate benchmark run or the latest file
from another checkpoint. Remote `az://` paths are supported by the repository's
existing blobfile tooling. The log's `Dumped ... generations` records can also
identify exact paths.

JSONL contains `input`, raw `output`, `gts`, `score`, `step`, and usually
`data_source`; optional fields include `clean_output`, reward component/error
counts, and flattened `extra_info` such as `id`, `audio_file`, or language.
Actual fields depend on the scorer/dataset. Table `ground_truth` corresponds to
dump `gts`; inspect structured references using the scorer's interpretation.
Join table examples to the same step's dump using available fields together
(input, reference, raw output, score), check uniqueness, and retain ambiguous
candidates instead of arbitrarily choosing one.

Across steps, prefer a unique `(data_source, id)` or an audio path plus segment
offsets. Check duplicates, reference consistency, and repeated generations
before pairing; never zip samples by rank or row order. A fallback composite
of dataset, exact prompt, and reference is usable only if unique and stable.
Generic audio prompts, repeated reference text, and randomly generated `uid`s
do not establish identity. Report matched, baseline-only, target-only, and
ambiguous counts. Fetch a sample's counterpart from the full dump even if it
was not selected in the other table. Without that counterpart, table entry or
exit means unmatched, not newly failed or recovered.

Use raw output to inspect language/format tags and cleaned output for content.
Apply the configured extraction and text normalization once; prefer logged
counts and alignments. If rescoring is needed, use the same scorer/normalizer
and identify any mismatch with logged results. An empty cleaned hypothesis can
be correct for non-speech; missing ground truth cannot establish substitutions,
deletions, or hallucination.

For grouped long audio, inspect the reward manager: it can broadcast one parent
score to every chunk while storing error counts only on the group head. Group
by dataset and stable parent identity, concatenate hypotheses in segment order,
and compare against the parent reference once. Null count fields are not zero;
do not compare one chunk to the full reference or count a repeated parent score
as multiple independent failures. Separate repeated decodes when identifiable.
Verify complete, unique segment indices, consistent segment counts and full
references within each parent. For the usual schema, group on
`(data_source, parent_audio_path)` and order by `seg_index`, checking that this
matches the scorer's `seg_start` order. Per-segment IDs are not parent IDs.

### Reuse checkpoint comparison reports

Read the sibling [asr-detail-compare skill](../../asr-detail-compare/SKILL.md)
and reuse its script instead of duplicating normalization or WER logic:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  .github/skills/asr-detail-compare/scripts/compare_result_details.py \
  --baseline-path tmp/run/val_data_gen/openml_bad/0.jsonl \
  --target-path tmp/run/val_data_gen/openml_bad/320.jsonl \
  --baseline-name step0 --target-name step320 --dataset openml_bad \
  --ref-column gts --hyp-column clean_output --join-columns id \
  --normalizer auto --top-n 20 --write-html --write-full-csv \
  --audio-blob-root '' --output-dir tmp/run/review/openml_bad
```

Use actual confirmed step files and the available Python environment. For a
single long-audio dataset, add `--aggregate-segments-by parent_audio_path`
and replace the join key with `--join-columns parent_audio_path`. Aggregate
first before constructing a perfect reference baseline for a single-checkpoint
report, or the full reference will be repeated as every segment's hypothesis.

Pass explicit `--ref-column gts --hyp-column clean_output` for verl dumps:
derived normalized columns must not overwrite original scoring inputs.
Normalization can be non-idempotent, especially for digits. Check full ID
coverage independently because the comparison uses an inner join. Read the
generated summary for actual HTML/CSV paths and inspect reference-match flags.
Label top-N reports as ranked subsets; use the full CSV for population counts.

### Check extraction and normalization artifacts

Inspect text discarded by the normalizer before interpreting large deletion
counts as missing speech. Some normalizers remove parenthesized content: the
same spoken words may score much better when the model replaces parentheses
with commas. Compare raw content and bracket frequency across checkpoints.
A diagnostic rescore that preserves parenthetical words must apply the same
alternative convention to both references and hypotheses at every checkpoint;
label it separately from official metrics and report any denominator change.

For grouped long audio, a malformed opening angle tag can survive extraction.
If normalization strips angle-bracket spans across newlines, a closing angle
bracket in a later chunk can cause many otherwise correct chunks to disappear
from the scored hypothesis. Locate the discarded span and its start/end
segment IDs, then compare parent-level normalization with segment-local
normalization or the run's verified extraction fix. Reproduce the logged score
first. These counterfactuals diagnose scoring sensitivity; they do not replace
official validation results or establish acoustic correctness. Record the
scorer revision when current cleanup differs from the run's dumped text.

## Identify patterns and test regression claims

Classify examples using text differences and available logged components:

| Pattern | Evidence to inspect |
|---|---|
| Substitutions, deletions, insertions | Logged alignment/counts; short reference/hypothesis differences |
| Empty output or truncation | Nonempty speech reference; missing suffix; token limit/finish evidence if available |
| Repetition or tail hallucination | Repeated spans or unsupported suffix; `repeat`/`tail_hallu` changes |
| Language or format failure | Wrong/missing language tags, malformed brackets, parsing failures; raw output |
| Digits, entities, keywords | Number/date/currency/name changes or missing required terms; matching counts/components |
| Punctuation or capitalization | Raw differences and configured normalization/reward treatment |
| Non-speech, reference, or scoring issue | Speech on silence, wrong reference, normalization disagreement; audio check when available |

Permit multiple labels but state that counts overlap. Keep suspected label or
audio problems as hypotheses unless verified; a large WER alone cannot prove
bad audio. For each pattern report count/denominator, distinct identities,
dataset/language when known, and one or two short examples with source
row/slot or JSONL line. Label counts as selected-table, matched-cohort, or
full-dataset counts. Deduplicate parent failures and cumulative copies first.
Distinguish missing/malformed headers from valid headers with the wrong target
language. The tag-based language reward alone does not establish the language
of the actual transcription. Treat repetition/tail detectors as review
candidates; repeated phrases can be present in the reference or audio.

For matched examples, separate newly regressed, persistent, improved, and
unmatched/unknown cases using adverse score/error changes and output evidence.
Report baseline/target steps, scores, error counts/components, and a short
reference versus before/after hypothesis. A low score at both steps is a
persistent difficult example, not evidence of a new training regression.
Define the bad-case criterion from the scorer or disclose the chosen threshold.
Reserve newly failed for matched cases crossing that criterion; a previously
bad case whose errors increase is a worsening persistent failure. With only one
snapshot, report observed patterns and leave change over time unknown.

Prioritize patterns by both frequency and impact. When matched samples have
unchanged references and comparable logged counts, report their added errors
(`target n_err - baseline n_err`) alongside case counts. For a complete fixed
single-decode corpus with unchanged total `n_ref`, a pattern's contribution to
the corpus error-rate change is its added errors divided by total corpus
`n_ref`. Overlapping labels must not be summed as independent contributions;
table-only counts cannot establish corpus impact.

Check whether these patterns also increase in full validation data and whether
dataset metrics worsen over subsequent checks. For single-decode complete
datasets, corpus `p_err` is normally `sum(n_err) / sum(n_ref)`, not mean sample
WER. Confirm `process_validation_metrics` and `update_var2metric2val` semantics
for repeated decodes, missing values, and parent-level counts before
reconstructing aggregates. Keep dataset metrics separate unless fixed dataset
weights and comparable scoring justify a combined number.
If one recording dominates a spike, quantify its share of added errors and
show the metric with that recording excluded while retaining it in official
totals. Check the same segment before, during, and after the spike. Disclose
logged versus recomputed totals; compound normalization can make the effective
reference-word count depend on the hypothesis, and later scorer revisions can
change cleanup behavior.

Relate confirmed validation regressions to the nearby training component,
filtering, optimization, and runtime evidence. Co-occurrence supports a
hypothesis, not causality. If full metrics are stable, report recurring tail
errors without claiming overall collapse. If only a selected table exists,
report its error patterns and explicitly leave validation-wide regression and
prevalence unknown. Recommend the specific missing comparison or data/scoring
check supported by the evidence.

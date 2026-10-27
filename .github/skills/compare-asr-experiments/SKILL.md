---
name: compare-asr-experiments
description: 'Compare verl ASR experiments in a concise PDF: key config/code differences, validation p_err figures, training reward mean figures, and brief bad-case analysis. Use for comparing runs and explaining quality tradeoffs; does not launch experiments.'
---

# Compare ASR Experiments

Keep the PDF focused on these sections, in order:

1. **Key config/code differences:** what changed and the expected effect.
2. **Validation `p_err` figures:** each dataset separately, lower is better.
3. **Training reward mean-value figures:** comparable `reward-core/*`,
   `reward-delta/*`, and `reward-vote/*` means.
4. **Brief bad-case analysis:** a few matched improvements/regressions and the
   error patterns they support.

Attempt all four for every requested run. Missing reward families or example
artifacts are coverage gaps, not zero values. Keep dataset tradeoffs visible;
do not reduce them to an overall average or winner unless explicitly requested.
Analyze existing evidence; creating recipes or submitting jobs requires a
separate user request.

## Resolve the jobs and evidence

Accept W&B run/project URLs, run IDs or experiment names, recipes, training logs,
and local exports. Resolve the requested runs from supplied context before
asking for missing identity information. A broad project comparison may include
running and completed runs; retain failed/interrupted runs with usable evidence
and label their limits. Do not silently substitute a similarly named run.

Record run ID/URL, display name, state, code/config revision when known,
fresh/resumed status, observed training/validation steps, and source timestamps.
Names identify recipes imperfectly. Use actual run IDs, distinguish retries and
seeds, and preserve resumed segments within a run that reuses its W&B identity.
Record model initialization, data mix, algorithm, reward formula, and relevant
decoding/scoring differences to explain comparability.

Use the supplied W&B host and existing authentication. For this repository,
`https://msaip.wandb.io` and entity `genai` are common defaults; verify them from
the run or project configuration. Read
[evidence and metric semantics](references/evidence-and-metrics.md) for sparse
history retrieval, exact keys, step mapping, and reward interpretation. Use full
unsampled histories rather than summary values or chart screenshots. Preserve
source provenance and collection gaps; save only relevant sanitized config
fields, excluding credentials and signed URL tokens.

## Identify key config/code changes

Compare effective run configs, launch overrides, and the code revisions actually
used. Show a compact before/after table of material changes with source pointers
and expected effects. Trace relevant code paths when revisions differ; distinguish
verified behavior from hypotheses. Exclude unrelated diffs and stale config
snapshots. If provenance is incomplete, state the gap rather than claiming a
single-variable comparison.

## Compare reward components

Discover the actual keys in each run and build a coverage matrix for all three
families. Preserve exact component names and units. Summarize important shared
components in the report and retain the full observed metric inventory in the
evidence. A missing family, component, or step is `N/A` with a reason.

Plot the logged training `*/mean` components against actual optimizer step, with
one run per line and separate panels for different components/units. Include a
compact figure comparing their arithmetic means over a stated shared step window;
label window bounds and observation counts. Raw noisy traces or a numeric table
alone do not replace these mean-value figures. Disclose any smoothing separately
and preserve the unsmoothed values in evidence.

For comparable metrics, show values and changes at shared actual training steps,
plus trajectory and recent stability. A short window of shared steps can help
with noisy training batches; disclose its boundaries, number of observations,
and statistic. Keep unmatched latest values separate from matched comparisons.
Do not interpolate sparse validation or silently align steps by nearest neighbor.
If step schedules have no overlap, disclose separate intervals rather than
claiming matched-budget evidence. Changes in batch size, data mix, or reward
definition limit what equal training steps establish.

Interpret `reward-core` as raw measures, `reward-delta` as comparison against a
baseline such as greedy decoding, and `reward-vote` as transformed contributions
only after checking the run's scorer. Error counts/rates can live under
`reward-core` and improve when they decrease. Delta and vote values depend on
normalization, thresholds, clipping, weights, and baseline quality; a larger
value does not by itself establish better transcription. Compare differing
formulas separately. Relate reward movements to validation evidence without
claiming causality from correlation.

## Compare validation `p_err` dataset by dataset

Discover exact `val-core/<dataset>/p_err/mean@1` and
`val-aux/<dataset>/p_err/mean@1` keys. Preserve original keys and verify alias
agreement at the same step. Do not substitute `score`, `p_edge`, `pb_err`,
`dter_p_err`, training reward, or `best@N` for the requested metric. Different
decoding/sample-count variants can be reported separately when explicitly
requested, using their exact definitions.

List the union of requested/observed datasets so missing coverage remains
visible. Verify manifest/subset, reference convention, scorer/normalizer,
chunk-versus-parent aggregation, and decoding settings for each dataset.
Separate incompatible cohorts; unknown provenance permits a descriptive
comparison with a caveat, not a definitive ranking.

For **each dataset**, show each run's initial available, latest, and best observed
`p_err` with its actual step, and compare at the latest shared validation step
when available. A resumed first observation is not necessarily a pretraining
baseline. Best observations are descriptive and may occur at different steps
across datasets; never combine them into an imaginary best checkpoint.
Disclose unequal training budgets and validation opportunities. Running-run
conclusions are provisional.

Keep raw fractions in evidence; display `100 * p_err` as percent. Show changes
as `100 * (target - baseline)` percentage points, with negative better. If useful,
relative error reduction is `100 * (baseline - target) / baseline`; leave it
undefined for a zero baseline. Preserve valid zeros and rates above 100%; reject
non-finite or negative observations rather than replacing them. Use the user's
control run when specified; otherwise name any chosen reference explicitly.

Do not average across datasets or pool their error counts by default. Conclude
which runs improve, regress, or trade off **on each dataset**. Dataset-specific
best steps do not identify one checkpoint that achieves all those results.

## Inspect validation examples and find patterns

Read [validation examples and fixing directions](references/validation-examples.md)
whenever using this skill. Obtain actual `val/generations` table data for every
run, not only a UI screenshot. Resolve a user reference to `val/generation` or
`val` to the run's actual logged key. Pair examples at shared validation steps
and around important per-dataset regressions when available. Cover each dataset
using matching `val_data_gen` dumps when the table lacks coverage or identity.

Match stable sample identities and references, never table slots or row order.
Fetch counterpart outputs from the full dump even when they were not selected
in the other table. Separate newly failed, worsening persistent, persistent,
improved, and unmatched cases. With one snapshot, describe observed patterns
without inventing a regression. Deduplicate cumulative table copies and grouped
long-audio parents before counting.

Report pattern counts with their denominator and scope (selected table, matched
subset, or full dataset), example identities, short reference/output differences,
and source rows or JSONL lines. Selected examples do not establish population
prevalence. Where full comparable counts exist, quantify added errors and corpus
impact; otherwise leave that unknown. Distinguish text/scoring artifacts from
acoustic hypotheses that need listening or additional evidence.

## Recommend the next fixing directions

Prioritize a short list of specific next checks or fixes. For each, connect:

- **Evidence:** dataset, run/step, `p_err` movement, pattern counts/scope, and
  representative matched examples; reward components are supporting diagnostics.
- **Hypothesis:** the likely mechanism and its uncertainty, distinguishing model,
  data/reference, extraction/scoring, reward, and decoding causes.
- **Next action:** a bounded diagnostic or one-factor experiment that can test
  the hypothesis, with exact config/code targets only when verified.
- **Success check:** the affected dataset's `p_err` at a comparable budget,
  reduction of the observed pattern, and guardrails for other datasets.

Do not suggest increasing a reward weight or changing a hyperparameter solely
because its logged component is low. If evidence is selected, incompatible, or
missing, prioritize the comparison or scoring check needed to resolve it. A
suggestion does not authorize modifying configs or launching jobs.

## Deliver the comparison

Generate a **slide-style `report.pdf` with embedded figures** by default, alongside
a concise `report.md` and a machine-readable `evidence.json`, in a unique
output directory (default `tmp/experiment_comparisons/<timestamp>/` in the working
project). Include source IDs/paths, exact keys, histories/steps used, sanitized
comparison settings, missing/conflicting observations, example identities and
pattern evidence. Use JSON `null` plus a reason for missing values.

Use the four-section structure above. Keep run identity and comparison limits
brief; put detailed inventories and provenance in `evidence.json` or `report.md`.
For bad cases, show a few stable-ID matched reference/output excerpts, the error
change, and a short interpretation. Include counts and denominators when available,
and at most a short next-check recommendation. Avoid extra background slides or
long per-checkpoint tables unless requested.

Read [slide-style PDF reporting](references/pdf-slides.md) when preparing the
deliverables. Build landscape 16:9 slides with one main finding per page,
readable charts and tables, and evidence-backed captions. Include per-dataset
`p_err` trajectories and training reward mean-value figures when histories exist,
plus figures for quantified validation error patterns when supported. Save the
editable deck source and reusable figure files with the report. Export and
inspect the actual PDF; an HTML deck or export instructions alone do not fulfill
the PDF requirement.

Link the PDF, Markdown report, and evidence in the answer and lead with the
dataset tradeoffs and best-supported next direction. If access or PDF rendering
fails, complete the available deliverables and state exactly which requested
dimensions or artifact remain unavailable; do not claim PDF generation succeeded.

## Obsidian handoff

When the user requests an Obsidian update, read and apply [obsidian-research-notes](../obsidian-research-notes/SKILL.md) with the verified comparison results, provenance, limitations, PDF, figures, and evidence. A comparison-only request does not trigger vault edits or publication.

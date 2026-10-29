---
name: report-experiment
description: 'Build a compact, evidence-backed PDF comparison report for verl ASR experiments focused on configuration differences, validation, and reward-core, with reproducible data and optional Obsidian updates. Use asr-detail-compare for utterance-level error review. Does not create slide decks or launch or modify experiments.'
---

# Report Experiment — Evidence Report

Compare two or more existing verl ASR experiments and produce a concise,
evidence-backed **report package** centered on three questions:

1. **What settings differ?**
2. **How does validation quality change by dataset and optimizer step?**
3. **How do comparable `reward-core/*` components change over optimizer steps?**

Keep experiment access read-only: do not edit recipes, launch jobs, or stop
runs. The default deliverable is a verified, compact `report.pdf` containing
only configuration differences, validation, and `reward-core`, plus the full
evidence package, key figures, and non-redundant comparison tables. Read
[compact PDF layout](references/compact-pdf-layout.md) before rendering the
PDF. This skill does not generate PPTX or depend on a presentation workflow. Use
[obsidian-research-notes](../obsidian-research-notes/SKILL.md) to update the
research note and affected daily homepage automatically, unless the user opts
out of vault updates or requests report artifacts only. These handoffs do not
authorize Git commit/push or other remote publication; those require separate
user authorization. When asked only to edit this skill, change its instructions
without generating reports or editing the vault.

## Inputs and run identity

Accept W&B run/project URLs, run IDs, experiment names, recipe paths, local
history exports, or training logs. Resolve the exact runs before comparison.
Do not silently substitute similarly named retries, seeds, or resumed lineages.

For every run, record:

- display name, run ID/URL, state, project, and experiment name;
- recipe and effective config source;
- code revision and launch overrides when available;
- fresh/resumed status and checkpoint lineage;
- observed optimizer-step and validation-step ranges;
- evidence collection time and missing provenance.

Use full unsampled histories rather than summary values or chart screenshots.
Read [evidence and metric semantics](references/evidence-and-metrics.md) before
collecting W&B histories or interpreting steps and `reward-core` values.

## Comparison contract

Use the user's control/reference experiment when specified. Otherwise use the
first requested experiment as the reference and label that choice explicitly.
Keep one stable experiment order and color mapping across every figure and table.

Comparative evidence must place experiments together:

- overlay comparable experiments on the same axes for each dataset/component;
- show values side by side in the same table row;
- compute deltas only at exact shared steps or over an explicitly shared window;
- express validation delta as target minus reference percentage points;
- express reward delta in the component's native unit with its direction stated.

Separate single-experiment charts do not satisfy the comparison unless a run is
incompatible and appears only in a clearly labeled coverage/provenance panel.

## 1. Put configuration differences in tables

Compare effective settings, not filename tokens or stale YAML defaults. Render
the result as comparison tables; do not describe configuration differences only
in prose. Start with the union of relevant fields, then render only material
differences, missing values, and unresolved values in the report table. Do not
include rows whose effective values are identical across every compared
experiment. Preserve identical controls in `evidence.json` under a clearly
named `shared_controls` section so they remain auditable without repeating them
in the human-facing table.

Cover these groups when present:

| Group | Typical effective config keys |
| --- | --- |
| Model | base checkpoint/revision; LoRA rank, alpha, dropout, and targets |
| Training | algorithm, epochs/steps, batch sizes, accumulation, optimizer, learning rate, warmup |
| Rollout | engine, sample count, temperature, top-p/top-k, sequence limits |
| Data | train mixture, sampling/order, subset sizes, validation manifests |
| Reward | scorer implementation, `reward-core` definitions, weights, normalization |
| Runtime | node/GPU topology, precision, resume behavior, validation/save cadence |

Use this table schema, adding one value column per experiment:

| Group | Effective config key | Experiment A | Experiment B | Difference / expected effect | Evidence |
| --- | --- | --- | --- | --- | --- |
| Training | `actor_rollout_ref.actor.optim.lr` | value | value | concise verified difference | resolved config/run source |

For three or more experiments, retain the same schema with additional experiment
columns. The “Difference / expected effect” cell summarizes the row across all
runs; split the table by group when the report table would become too wide.

Use fully qualified effective config keys whenever available. Normalize simple
booleans, numbers, paths, and units for readability while preserving raw values
in `evidence.json`. Summarize long lists/dictionaries in the report cell and
link them to the complete structured value in the evidence. Mark absent
settings as `N/A` and unresolved settings as `Unknown`; never infer values from
experiment names. A row with identical values in every experiment is not a
report-table row and belongs only in `shared_controls` evidence.

For each difference, include its source and a brief expected effect only when
supported by code/config semantics. Visually emphasize changed cells, not the
control cells. If several settings changed together, add a table-level note that
the comparison is multivariable. Save only the difference/unresolved row set as
`data/config_differences.csv` or JSON. Save shared controls separately as
`data/shared_controls.csv` or JSON when they are useful for audit. Do not repeat
the same setting in both tables.

## 2. Plot validation trajectories

Discover exact `val-core/<dataset>/p_err/mean@1` and
`val-aux/<dataset>/p_err/mean@1` keys. Preserve the observed key and verify
same-step aliases before merging them. Do not substitute reward, `score`,
`p_edge`, `pb_err`, `dter_p_err`, or `best@N` for `p_err`.

Create validation figures with:

- actual optimizer step on the x-axis;
- `p_err` percentage on the y-axis, lower is better;
- all comparable experiments overlaid in each dataset panel, using one
  consistent color per experiment and markers at observed points;
- one panel per dataset, splitting dense panels across figures/pages;
- gaps for missing observations and resumed history rather than interpolation.

Alongside the figures, report for each dataset and run:

- first observed value and step;
- latest observed value and step;
- best observed value and step;
- value at the latest exact shared validation step when one exists.

Use a side-by-side comparison table such as:

| Dataset | Exact shared step | Reference `p_err` | Experiment B `p_err` | B − reference (pp) | Coverage / caveat |
| --- | --- | --- | --- | --- | --- |

For additional experiments, add one value and one delta column per experiment.
Keep unmatched latest/best values in separate labeled columns or a continuation
table; never mix them with the exact-shared-step delta.

Keep datasets separate. Do not average them into an overall winner unless the
user explicitly requests a defined aggregation. Verify manifest, reference,
normalization, grouping unit, decoding settings, and sample count before calling
two series comparable. Unknown provenance supports a descriptive comparison
with a caveat, not a definitive ranking.

Display raw fractions as percentages. Express target-minus-control changes in
percentage points, where negative is better. Dataset-specific best points can
occur at different steps and do not identify one checkpoint that is best on all
datasets.

## 3. Plot `reward-core` over optimizer steps

Discover every logged `reward-core/<component>/mean` key for each run. Build a
coverage matrix first, then select the shared comparable components that are
most decision-relevant for the key figures. Group related components into
readable panels instead of producing one repetitive figure per metric. Preserve
all discovered components in data/evidence, including run-only and unplotted
components, and label why they were not used for cross-run figures.

For every plotted component:

- use the verified optimizer-step axis, never W&B row index or wall time;
- overlay all comparable experiments on the same component axes and show raw
  observed values, with optional smoothing only when disclosed;
- keep components with different units or directionality in separate panels;
- state whether higher or lower is better after checking the scorer;
- preserve gaps, resumed segments, conflicts, and non-finite observations;
- label the exact metric key and observation count.

When runs overlap in optimizer-step budget, add a compact shared-window summary
with window bounds, arithmetic mean, observation count, and latest exact shared
step. Do not use nearest-neighbor alignment or interpolate missing values. If
there is no common step/window, show the separate trajectories and state the
budget limitation.

Use a side-by-side reward table such as:

| `reward-core` component | Shared window | Reference mean (n) | Experiment B mean (n) | B − reference | Direction / unit |
| --- | --- | --- | --- | --- | --- |

Add value/delta columns for additional experiments. Keep incompatible
definitions in the coverage table rather than placing them in numeric delta
columns.

`reward-core` is a training-batch diagnostic. Its direction depends on the raw
field: accuracy-like scores can be higher-better, while error counts/rates can
be lower-better. Check the effective scorer and data mixture before comparing
numeric values. Do not treat reward movement as validation improvement or infer
causality from correlation.

## 4. Explain the comparison

Lead with dataset-level validation tradeoffs, then relate them to settings and
`reward-core` behavior. Separate:

- **observed facts:** settings, exact metrics, steps, and trajectories;
- **supported interpretation:** behavior consistent with a known setting change;
- **hypothesis:** a plausible explanation requiring another controlled run.

Call out unequal budgets, different validation cadence, resume boundaries,
changed data mixtures, reward-definition changes, and code revisions. Do not
declare one overall winner when datasets trade off or evidence is incompatible.
Do not recommend changing a reward weight merely because its component mean is
small.

## Evidence package and default deliverables

Write a unique report directory, defaulting to
`tmp/experiment_reports/<timestamp>/`. Complete and verify:

- `report.md`: the authoritative comparison report with concise findings,
  key-figure captions, difference-only table rows, and caveats;
- `report.pdf`: the self-contained human-readable report rendered from the
  verified Markdown, figures, and canonical tables;
- `evidence.json`: run identity, complete settings, `shared_controls`, exact
  metric keys, raw observations, comparison windows, transformations, claim
  provenance, and missing-data reasons;
- `figures/`: reusable key validation and `reward-core` figures;
- `data/`: plotted CSV/JSON data and canonical numeric tables so every figure
  and value is reproducible.

The default compact PDF must include these key visuals when evidence exists:

- one validation trajectory figure per comparable dataset, or a readable set
  of dataset panels, with all comparable runs overlaid;
- a validation exact-shared-step comparison figure or table when aligned points
  exist;
- reward-core trajectory figures grouped by compatible units and direction,
  without duplicate panels that communicate the same comparison;
- a compact reward shared-window table when a common window exists;
- a difference-only effective-settings table.

Do not add a cover, executive-summary page, run-identity/coverage page,
standalone interpretation page, or evidence-limits page to the PDF. Preserve
that material in `report.md` and `evidence.json`, where it remains auditable
without expanding the reading surface.

Each table must have a distinct analytical purpose. Do not repeat the same rows
or values in multiple tables merely with different formatting. In the PDF,
prefer one canonical table for effective-setting differences, one for exact
shared-step validation, and one for reward shared-window summaries. Keep run
coverage and provenance in the evidence package unless a missing-coverage fact
must appear as a compact caveat beside the affected figure. Omit any table whose
content is already fully represented elsewhere and adds no new
decision-relevant information.

Every figure must be generated from a saved file under `data/`, and every table
cell must trace to `evidence.json` or be marked `Unknown`/`N/A` with a reason.
Never use decorative charts, invented values, or screenshots as substitutes for
the underlying numeric evidence. If an input dimension is unavailable, preserve
the gap as `null` with a reason and use a compact coverage table instead of an
empty or fabricated figure.

Return links to `report.pdf`, `report.md`, `evidence.json`, the data directory,
the figure directory, and any updated research note/homepage. Lead with the
best-supported validation findings.

## In-chat presentation

The report is also a Chat deliverable, not only a PDF artifact. After the
figures and canonical tables have been generated and verified, show the key
Validation figure(s) directly in the current Chat using Markdown image embeds
with absolute local paths, for example:

```markdown
![Validation trajectories](/absolute/path/to/figures/validation.png)
```

Show the effective Config Difference in the Chat as a concise Markdown table
using the verified `data/config_differences.csv` or JSON values. If the table
is too wide to read reliably, render that same saved data as a dedicated
config-difference image and embed it with an absolute local path. Do not
recreate values by hand, use screenshots as the evidence source, or show a
figure/table that is not backed by `evidence.json` and the saved `data/`
artifacts. Include links to the underlying figure, table data, and full PDF
after the inline presentation. The inline display is required even when the
same content is already embedded in `report.pdf`.

### Generate and verify `report.pdf`

Render the PDF only after `report.md`, `evidence.json`, figures, and canonical
tables agree. Use a deterministic local PDF path appropriate to the available
environment, such as a document renderer or a direct plotting/PDF library. Do
not route through presentation software or create an intermediate slide deck.

The PDF must:

- use compact document-oriented pages, normally A4 landscape, rather than
  slide-sized pages;
- contain only the three report sections defined in
  [compact PDF layout](references/compact-pdf-layout.md): difference-only
  settings, validation, and `reward-core`;
- place short interpretation and caveats beside the relevant table or figure
  instead of creating separate narrative pages;
- embed the saved key figures at readable resolution and preserve the stable
  run colors and labels;
- keep tables readable across page boundaries, repeating headers when the
  renderer supports it and splitting wide tables rather than shrinking text;
- contain no invented values, duplicate tables, empty placeholder pages, or
  links standing in for required visuals;
- record its path and rendering method in `evidence.json` without making the
  PDF an independent source of claims.

Verify the finished PDF by checking that it opens, has a nonzero page count,
contains extractable report text when supported, and renders every page without
clipping, blank figures, or unreadably small tables. Render representative pages
and a contact sheet for visual inspection when tools are available. Cross-check
all numeric tables, figure captions, findings, and caveats against
`evidence.json`. If PDF rendering fails, retain the verified evidence package,
report the exact failure, and do not claim that `report.pdf` exists.

After PDF verification, perform the in-chat presentation before delivery:

- embed the verified Validation trajectory figure (or the smallest readable
  set of per-dataset Validation figures) directly in the Chat;
- show the verified effective Config Difference table inline, or embed its
  rendered image when width makes a Markdown table unreadable;
- cross-check the inline values and image paths against `evidence.json` and
  the saved figure/data files, and still return the normal artifact links.

## 5. Update the research note and daily homepage

After `report.pdf`, the evidence package, and key figures pass verification, invoke
[obsidian-research-notes](../obsidian-research-notes/SKILL.md). Keep vault
organization and publication rules in that skill rather than duplicating them
here. Supply:

- exact compared run identities, control/reference choice, checkpoint/step
  coverage, and evidence collection time;
- verified setting differences, per-dataset validation findings and shared-step
  deltas, comparable `reward-core` windows/means with observation counts, and
  material provenance or coverage limits;
- `report.pdf`, `report.md`, `evidence.json`, reusable key figures, plotted data,
  and any local dependencies needed to open the copied report;
- an existing comparison-note/daily-entry identity when known, so a refresh
  updates the same report rather than creating duplicate notes.

Use only the collected evidence. Utterance-level matches and bad cases are
optional inputs when already available; their absence must not trigger new
evaluation or invented examples. The note skill owns selecting figures,
copying linked assets, and updating the existing detailed note and daily entry.

Before delivery, verify the note links and copied report dependencies resolve,
the note metrics agree with the verified PDF/evidence, and the affected daily
entry points to the refreshed report. Include both note links with the report
artifacts and state whether vault changes are local, committed, or pushed.
If the user opted out, skip the handoff; if the vault is unavailable, deliver
the verified report and explicitly record the incomplete note handoff rather
than claiming it succeeded.

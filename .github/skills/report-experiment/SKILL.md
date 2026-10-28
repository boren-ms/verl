---
name: report-experiment
description: 'Build an evidence-backed native PPTX comparison report for verl ASR experiments through ppt-master, then update the research note and daily homepage through obsidian-research-notes by default. Covers material settings, validation p_err, and comparable reward-core trajectories. Use asr-detail-compare for utterance-level error review. Does not launch or modify experiments.'
---

# Report Experiment — Slide Deck

Compare two or more existing verl ASR experiments and produce a concise,
evidence-backed **slide deck** centered on three questions:

1. **What settings differ?**
2. **How does validation quality change by dataset and optimizer step?**
3. **How do comparable `reward-core/*` components change over optimizer steps?**

Keep experiment access read-only: do not edit recipes, launch jobs, or stop
runs. After the evidence package is verified, use
[ppt-master](../ppt-master/SKILL.md) to generate and verify the editable native
PPTX. Then use
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
in prose. Start with the union of relevant fields, then show material
differences plus a small set of essential shared controls.

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
runs; split the table by group when the slide would become too wide.

Use fully qualified effective config keys whenever available. Normalize simple
booleans, numbers, paths, and units for readability while preserving raw values
in `evidence.json`. Summarize long lists/dictionaries in the slide cell and link
them to the complete structured value in the evidence. Mark absent settings as
`N/A`, unresolved settings as `Unknown`, and identical essential controls as
`Same`; never infer values from experiment names.

For each difference, include its source and a brief expected effect only when
supported by code/config semantics. Visually emphasize changed cells, not the
control cells. If several settings changed together, add a table-level note that
the comparison is multivariable. Save the complete row set as
`data/config_differences.csv` or JSON and reuse those exact rows in the PPTX,
supporting Markdown, and evidence artifacts.

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
coverage matrix first, then plot shared comparable components. Retain run-only
components in the evidence and label them unavailable for cross-run comparison.

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

## Evidence package and native PPTX deliverables

Write a unique report directory, defaulting to
`tmp/experiment_reports/<timestamp>/`. Complete and verify the evidence package
before invoking the presentation workflow:

- `report.md`: the authoritative source brief for the deck, including the slide
  roster, exact takeaway titles, concise table rows, findings, and caveats;
- `evidence.json`: run identity, settings, exact metric keys, raw observations,
  comparison windows, transformations, slide/claim provenance, and missing-data
  reasons;
- `figures/`: reusable validation and `reward-core` figures;
- `data/`: plotted CSV/JSON tables so every figure and numeric table is
  reproducible.

The primary presentation deliverable is an editable native `report.pptx`
generated through [ppt-master](../ppt-master/SKILL.md). A rendered `report.pdf`
or self-contained HTML preview is optional when the environment supports it;
neither replaces the PPTX or becomes an independent source of claims. Keep the
ppt-master project/workspace path in `evidence.json` and copy or link the final
PPTX into the report directory without discarding its verification artifacts.

Use this slide sequence:

1. **Title and executive takeaway:** compared experiments and the main
   dataset-level tradeoff, without claiming an unsupported overall winner.
2. **Scope and comparability:** run identities, step coverage, and the most
   important limitations.
3. **Settings differences:** one or more readable slides showing material
   effective-setting differences and sources.
4. **Validation trajectories:** chart-dominant slides, split by dataset group;
   follow with a compact exact-shared-step summary when supported.
5. **Reward-core trajectories:** chart-dominant slides grouped by compatible
   components; follow with the shared-window summary when supported.
6. **Findings:** concise observed facts, supported interpretation, hypotheses,
   and material evidence gaps.

Every slide must have one main message expressed as a takeaway title. Prefer a
large figure or compact comparison table over prose; move exhaustive details to
`report.md` and `evidence.json`. Do not create document-like slides filled with
paragraphs, tiny tables, or repeated methodology.

The deck must contain actual embedded visuals, not prose placeholders or links:

- a run-identity and step-coverage table;
- at least one material configuration-difference table using the schema above;
- per-dataset validation trajectory figures for every comparable dataset;
- a side-by-side validation comparison table with exact shared-step values and
  deltas, plus clearly separated first/latest/best values;
- `reward-core` trajectory figures for every shared comparable component;
- a side-by-side reward shared-window table with values and deltas, or an
  explicit coverage table when no component is comparable.

Every figure must be generated from a saved file under `data/`, and every table
cell must trace to `evidence.json` or be marked `Unknown`/`N/A` with a reason.
Never use decorative charts, invented values, or screenshots as substitutes for
the underlying numeric evidence.

Read [slide-deck guidance](references/slide-deck.md) while preparing the source
brief. It owns the experiment-report content contract, while ppt-master owns
presentation construction, routing, SVG/PPTX generation, and visual QA.

### Invoke ppt-master

After `report.md`, `evidence.json`, `figures/`, and `data/` agree:

1. Read [ppt-master](../ppt-master/SKILL.md), run its attribution guard, and
   follow its router. This source-backed new-deck request selects **Generate
   PPTX**.
2. For the automatic report path, explicitly request ppt-master's **Quick
   Generate** profile: no Strategist/confirmation handoff, no template catalog
   selection, and no speculative research beyond the supplied verified report
   package. Honor an explicit user request for the Default profile instead.
3. Give ppt-master `report.md`, `evidence.json`, the saved plotted data, and all
   required figures as source/resource inputs. State that evidence values,
   experiment order/color mapping, metric semantics, slide roster, and caveats
   are binding; visual composition remains ppt-master's responsibility.
4. Require the PPTX to contain the visuals and tables listed above. Use the
   saved figures/data rather than screenshots, decorative replacements, or
   recomputed values. A ppt-master-native chart/table is allowed only when it
   preserves the exact saved values and provenance.
5. Complete ppt-master's lockless final quality check and native PPTX export.
   Render or inspect the exported deck when its workflow/environment provides a
   renderer, then cross-check every plotted value, numeric table cell, takeaway,
   and caveat against `evidence.json` before accepting `report.pptx`.

Do not hand-author a competing HTML slide deck or bypass ppt-master with a
separate presentation generator. If ppt-master is unavailable or fails, retain
the verified evidence package, report the exact incomplete handoff, and do not
claim that the PPTX exists.

If an input dimension is unavailable, preserve the gap as `null` with a reason
and show a compact coverage table instead of an empty or fabricated figure.
Return links to the native PPTX, optional rendered PDF/HTML preview when one was
actually produced, supporting Markdown, evidence, data, and figure directory,
plus the research note and daily homepage when updated. Lead with the
best-supported validation findings.

## 5. Update the research note and daily homepage

After the native PPTX and evidence pass verification, invoke
[obsidian-research-notes](../obsidian-research-notes/SKILL.md). Keep vault
organization and publication rules in that skill rather than duplicating them
here. Supply:

- exact compared run identities, control/reference choice, checkpoint/step
  coverage, and evidence collection time;
- verified setting differences, per-dataset validation findings and shared-step
  deltas, comparable `reward-core` windows/means with observation counts, and
  material provenance or coverage limits;
- `report.pptx`, any actual `report.pdf`/HTML preview, `report.md`,
  `evidence.json`, reusable figures, plotted data, the ppt-master project path,
  and any local dependencies needed to open the copied report;
- an existing comparison-note/daily-entry identity when known, so a refresh
  updates the same report rather than creating duplicate notes.

Use only the collected evidence. Utterance-level matches and bad cases are
optional inputs when already available; their absence must not trigger new
evaluation or invented examples. The note skill owns selecting figures,
copying linked assets, and updating the existing detailed note and daily entry.

Before delivery, verify the note links and copied report dependencies resolve,
the note metrics agree with the verified deck/evidence, and the affected daily
entry points to the refreshed report. Include both note links with the report
artifacts and state whether vault changes are local, committed, or pushed.
If the user opted out, skip the handoff; if the vault is unavailable, deliver
the verified report and explicitly record the incomplete note handoff rather
than claiming it succeeded.

# Research Note and Paper Architecture

Adapt these structures to the vault template and domain. Preserve stable paths/IDs. Replace drafting placeholders with project-specific content or explicitly labeled missing items. Do not require Dataview or install vault plugins.

## Owning research note

Follow existing frontmatter conventions. Where permitted, include idea ID, lifecycle status, created/updated dates, concise homepage summary, and paper-architecture link.

Suggested body:

1. **Current assessment:** value judgment, verified evidence, central uncertainty, next action, last checked time. State “尚无实验依据” where applicable.
2. **Question and scope:** importance, source-backed observations, target conditions, resource constraints.
3. **Literature synthesis:** search provenance, closest-work comparison, established knowledge, disagreements, opening, reusable advice with source locations. Link detailed reading notes where useful.
4. **Complete provisional story:** connected argument and candidate contributions; hypotheses distinct from findings.
5. **Claim ledger:** table below.
6. **Complete experiment plan:** overview plus detailed entries, ordered by dependencies and information value.
7. **Evidence and artifacts:** verified measurements/figures, sources, comparability limits, interim/full coverage, gaps.
8. **Decision history:** substantive assessment/criteria/scope changes with dates and triggering evidence.
9. **Paper links and remaining work:** architecture, related notes, missing citations/results/implementation/theory.

### Claim ledger

| Claim ID | Scoped hypothesis | Literature / existing evidence | Competing explanation | Refuting or narrowing observation | Experiments | Evidence status | Paper section |
|---|---|---|---|---|---|---|---|

Evidence status: untested, preliminary support, supported within stated scope, mixed, refuted, or inconclusive. Explain each assessment; a label is not proof.

### Experiment overview

| Experiment ID | Claims / question | Dependencies | Priority / estimated cost | Execution status | Predefined criterion | Current evidence | Next action |
|---|---|---|---|---|---|---|---|

Execution status: planned, ready, running, completed, failed, cancelled, or deferred. A deferred experiment is not a failed hypothesis.

### Detailed experiment entry

- **Purpose:** claim IDs, question, predictions, competing explanations.
- **Design:** intervention, controls, baseline identities, datasets/splits, training/evaluation settings, fairness/confounding.
- **Measurement:** primary/diagnostic metrics, units, coverage, appropriate uncertainty analysis.
- **Decision:** success/refutation/inconclusive criteria recorded before outcomes; next action for each outcome.
- **Execution plan:** dependencies, compute/time estimate and assumptions, config/code needs, identities/output locations only when assigned.
- **Observed evidence:** real source paths/settings/checkpoints, evidence time, last checked time; unmeasured results explicitly pending.
- **Paper use:** section, figure/table, missing-item IDs this experiment can fill.

## Linked paper-architecture note

Open with provisional title, one-sentence candidate contribution, draft status, owning-note link, and central evidence gaps. Draft an abstract using conditional/project-intent language where results are absent. Never insert hypothetical performance numbers.

| Section | Write before experiments | Missing material |
|---|---|---|
| Introduction | Importance, source-backed prior boundaries, opening, hypothesis, candidate contribution | Confirmed findings and final contribution wording |
| Related Work | Thematic synthesis, closest competitors, verified distinctions | Unresolved/new sources |
| Method / Formulation | Proposed algorithm/analysis, notation, assumptions, scope, rationale | Undecided implementation; unproved claims |
| Experiments | Claim-linked questions, baselines, protocols, metrics, planned figures/tables | Actual settings/results/variability/diagnostics |
| Discussion / Limitations | Alternatives, known limits, failure conditions | Evidence-based interpretation and revised boundaries |
| Conclusion | Research question and intended contribution | Supported final conclusions |
| Appendix | Planned reproducibility structure/supporting analyses | Configs, derivations, full results, cases |

Write each section's actual argument and key paragraphs/bullets with known sources. This table is a guide, not the deliverable itself.

### Figure/table plan

| Figure/table ID | Question / intended message | Claims | Required experiments / sources | Missing data | Status |
|---|---|---|---|---|---|

Keep conceptual illustrations distinct from empirical plots. Do not generate result-looking plots with invented data.

### Missing-item ledger

Assign stable IDs (`T01`, ...), typed as result, citation, implementation, theory, or framing. Record section/claim, exact gap, supplying experiment/action, dependencies, and status. Example placeholder: `[待补 T03：E03 的机制消融结果，用于检验 C02；若无区分力，修订机制解释]`.

Fill only items actually resolved by new evidence. Revisit abstract, introduction, discussion, and conclusion along with results. Preserve original hypotheses and explain story changes.

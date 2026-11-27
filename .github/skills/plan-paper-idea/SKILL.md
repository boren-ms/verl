---
name: plan-paper-idea
description: Develop ML paper ideas through literature review, research-value assessment, a complete provisional story, and claim-linked experiment planning. Deliver initial research notes and paper architecture in Obsidian before experiments, then update evidence and decisions as results arrive. Use for research planning and continuation, rather than paper search alone or reporting completed runs alone.
---

# Paper Idea and Experiment Planning

Turn a question, observation, paper, or partial result into a literature-grounded research plan. Before experiments, deliver a reviewable initial research note and paper architecture in Obsidian. “初版” means a draft, not publication or proof that the idea works.

## Scope and continuation

- Support new ideas and existing projects. Recover existing notes, reviewed literature, claims, experiment identities, results, and decisions before proposing new work.
- Infer available context from supplied artifacts and the repository. Ask only for missing information that materially affects the question, constraints, or interpretation; continue independent review/drafting meanwhile.
- Default to planning and vault updates. This skill alone does not authorize training, evaluation, scheduled monitoring, external publication, or Git commit/push. Honor explicit authorization already given, but complete the pre-experiment deliverables before execution.
- Adapt to the user's domain and budget. ASR/verl is a supported application, not a universal restriction.

## 1. Review literature before committing to a direction

Read [references/literature-review.md](references/literature-review.md) for search coverage, paper extraction, and completion criteria. Use the repository's [paper-search](../paper-search/SKILL.md) skill for discovery; its reading list is input, not sufficient review by itself. Read relevant full-text sections of decisive papers, especially methods, evaluation, limitations, and contrary findings.

Explain what is established, what remains uncertain, and which techniques or experimental practices can be reused under which conditions. Distinguish author conclusions from your interpretation. Verify citations and attributed claims against sources; mark inaccessible/unverified material as gaps.

## 2. Assess value and define falsifiable claims

Write an explicit assessment: worth testing, needs narrowing/reframing, already covered, or insufficient evidence to assess. Explain significance, closest prior work, proposed difference, feasibility, and the evidence needed to make that difference meaningful. A missing search hit does not establish novelty; a method combination does not automatically establish value.

Assign stable claim IDs (`C01`, ...). Record scope, literature/existing evidence, predicted observations, competing explanations, and refuting or narrowing observations. Separate performance claims from causal/mechanistic explanations.

If the direction is covered or unsupported, still deliver the review, assessment, provisional architecture, and missing-evidence list. Mark the story as requiring revision and experiments as deferred. Recommend a reasoned alternative without inventing novelty or launching work merely to fill the plan.

## 3. Draft the complete provisional story and paper architecture

Write a connected story: important problem → established approaches and boundaries → evidence-grounded opening → hypothesis and proposed method → candidate contribution → required tests and possible counterevidence.

Complete the argument structure before execution. Label planned claims as hypotheses and use conditional language for expected results. Never present an unmeasured improvement, unproved theorem, or predicted mechanism as a finding. Preserve competing explanations.

Read [references/note-and-paper-template.md](references/note-and-paper-template.md) when drafting. Populate each section with its actual argument, sources, intended figures/tables, and claim-linked missing material; section headings alone are insufficient. Use `ml-paper-writing` if available for detailed drafting/citation mechanics, without expanding the task to submission work.

## 4. Plan experiments to resolve evidence gaps

Assign stable experiment IDs (`E01`, ...), linked to claims and paper sections. Record question, intervention, controls, baselines, data/splits, metrics, resource estimate and uncertainty, dependencies, decision criteria, and known artifact locations. Mark unknown run/config paths as unassigned, not as existing links.

Prioritize low-cost experiments that can change the go/no-go decision, then main-effect, mechanism, generalization, and boundary tests as relevant. Do not impose a fixed experiment count or obligatory benchmark sweep.

- Make training/data/decoding comparisons fair or explain the tradeoff. Address leakage, selection bias, confounding, and baseline reproduction where material to the claim.
- Choose repeated runs, uncertainty estimates, or paired analyses proportionally to expected variance and the claim. Record practical decision criteria before outcomes; disclose thresholds informed by pilots.
- Explain how positive, negative, and inconclusive outcomes change the claim/next experiment. Distinguish implementation failure, insufficient diagnostic power, and evidence against the hypothesis.
- Link each planned figure/table to required experiments/data. Map missing paper items to experiments, citation checks, implementation decisions, or theoretical tasks.

## 5. Deliver the pre-experiment notes in Obsidian

Use the repository's [obsidian-research-notes](../obsidian-research-notes/SKILL.md) skill for vault instructions, existing-note search, assets, links, and affected homepage navigation. Its default vault is `/mnt/c/Users/boren/code/ASR-Research`; honor a user-specified vault. Use matching research-topic folders/templates, not necessarily the comparison folder.

- Maintain one owning research note per idea and a linked paper-architecture note. Update existing owners; preserve backlinks and substantive historical evidence. Keep small experiments in the research note; split complex ones only when useful.
- Write in the vault's language, currently Chinese. Open with current assessment, strongest available evidence, unresolved issues, experiment status, next action, and last checked time. Explicitly state when no experimental evidence exists.
- Include the full review synthesis, value assessment, story, claims, complete experiment plan, provenance, and decision history. Add concise navigation on the affected homepage using vault conventions; do not duplicate the full plan there.
- Verify written files and links; link only existing artifacts. Return both deliverable links and outstanding gaps. If the vault is inaccessible, prepare drafts in an accessible task location, report their real paths, and keep Obsidian delivery incomplete.

### Pre-experiment completion gate

Before any experiment launch, verify:

- The review meets documented coverage/depth criteria; residual gaps do not undermine central positioning.
- Research value is assessed, closest prior work compared, and principal claims have meaningful validation/falsification paths.
- The provisional story, complete experiment plan, and populated paper architecture are saved in the vault, cross-linked, and accessible for review.
- Unknown results, citations, method details, and figures are labeled as missing with corresponding next actions.

If a condition is unmet, continue review/drafting or report the concrete blocker. Do not bypass this gate with a quick pilot. Deliverables do not require a new permission request unless the user explicitly requested approval before execution; apply existing execution authorization after the gate.

## 6. Continue from verified results

When asked to check progress or supplied new results, update the same notes. Use `run-asr-exp` for authorized ASR execution/monitoring, `report-experiment` for run evidence, and ASR detail-analysis skills for utterance evidence where appropriate. These are conditional integrations, not general ML dependencies.

- Track experiment status separately from claim evidence status. A completed run is not a supported claim. Record actual run/config/checkpoint identities, coverage, metric definitions, units, sample counts, and source paths when available.
- Distinguish interim, subset, and full-evaluation evidence. Record evidence production time separately from last checked time. Progress refresh occurs on invocation or through an explicitly requested monitor, not automatically.
- Update result → evidence assessment → claim → story → paper missing items. Preserve inconclusive and negative results. Do not cherry-pick checkpoints or revise success thresholds without dated explanation.
- Preserve original predictions and criteria. Log substantive changes with date, triggering evidence, old/new judgment, and next decision. Supersede stale summaries while retaining labeled historical evidence.
- Return updated note links, the substantive evidence change, and the next informative experiment or reason to stop/reframe.

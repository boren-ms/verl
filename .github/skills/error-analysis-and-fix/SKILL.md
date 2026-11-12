---
name: error-analysis-and-fix
description: Turn ASR experiment error patterns into traced training data, reproducible subsets, a refinement recipe, and a measured fix against the original experiment. Use for the error analysis, data retrieval, ReMax training, and comparison loop; use asr-detail-compare alone for an utterance report without data or refinement work.
---

# Error Analysis and Fix

Close the loop from observed ASR failures to a measured refinement result. A ready config, completed mining run, or reduced training loss does not establish a fix.

## Scope and evidence

Determine the requested stages from the conversation: analysis, retrieval, subset preparation, configuration, training, evaluation, reporting. Complete already-authorized stages without repeatedly asking. Skill creation or a request to prepare a recipe does not authorize running GPU jobs. Preserve requested checkpoints, dataset order, ZIP layout, topology, priority, and resume semantics. Use an isolated worktree when requested; preserve unrelated changes.

Copy [assets/case-plan.yaml](assets/case-plan.yaml) into the task's report directory and fill it as work progresses. Define success criteria and the authorized iteration/budget limit. Record exact code, model, source, scorer and output identities; stage status and evidence; missing inputs and confounds. Keep selection model, refinement initialization, and original comparison experiment separate. These may be three different checkpoints.

## 1. Establish the error pattern

- Use the installed `asr-detail-compare` skill for paired or single-output analysis. Join by stable sample identity, reproduce the canonical metric, inspect raw outputs, and rank concrete regressions. Keep CER, WER, DTER and parser effects distinct.
- Write a versioned, machine-readable pattern definition: fields searched, normalization, matcher, positive and negative examples, evidence type, known false positives. Patterns may concern reference lexemes, substitutions, omissions, hallucinations, formatting, language or acoustic conditions; do not assume a regex can identify every error.
- Distinguish **reference-feature candidates**, **model-confirmed errors**, and **hybrid selection**. A word in a reference is evidence for retrieval, not proof of recognition failure. State the searched corpus and completed coverage.

## 2. Find source data and retrieve candidates

Read [references/data-retrieval.md](references/data-retrieval.md) for lineage discovery, remote manifests, supported adapter fields, and the local selection helper.

- Search the current codebase, effective training configs and recorded manifests, then related codebases such as MOE if needed. Resolve nested manifests to actual audio and human reference records. Stored data existence and inclusion in a particular training run are different claims.
- Record source URI/revision, split, label provenance, record IDs, loader selectors, counts and hashes. Verify remote account and source listing before copying. Use the installed transfer skill for an authorized transfer.
- Search train data for training candidates. Use heldout data only for evaluation. Preserve parent/recording boundaries where chunks or segments share audio. Keep source labels; generated hypotheses belong in an audit or score file.
- If confirmed failures are needed, decode the declared candidate universe using the pinned selection checkpoint, prompt, parser and canonical scorer. Verify expected unique IDs and complete coverage before global ranking. Partial outputs cannot establish global Top-K.
- For an error-count request rank absolute `n_err`, with deterministic ties; WER/CER ratio is a different selection rule. State thresholds, quotas, deduplication, duration, seed and exclusions. Never silently pad a subset or change its size.

## 3. Freeze data products

Produce the products needed for this case; do not invent extra sets merely to fill the template:

| Product | Purpose |
| --- | --- |
| Source/candidate manifest and audit | Reproduce retrieval and quantify searched coverage |
| Pattern training subset | Train on retrieved examples with trusted references |
| Optional subtraining set or replay/control mixture | Pilot cheaply, vary dose, or protect broad coverage; record membership and weights |
| Fixed pattern diagnostic | Measure the targeted behavior on heldout examples |
| Representative/full validation and guard sets | Measure generalization and regressions beyond the mined pattern |
| Summary validation bundle | Compact monitoring suite with explicit component metrics and denominators |

A summary validation bundle is a set of named evaluations, not a new average with implicit weights. Preserve established headline coverage and weighting; report new rows separately when appropriate.

Audit ID, audio/segment, parent/recording and, where useful, normalized-text overlap. Document unavailable checks. Fix evaluation membership before refinement; label diagnostics selected from baseline test errors as biased hard cases. Prefer dev for iterative tuning and reserve a final test. Report full validation alongside hard-case results; never present a selected Top-K diagnostic as representative.

## 4. Prepare and validate the refinement recipe

Read [references/experiment-loop.md](references/experiment-loop.md) for recipe checks, lifecycle evidence and comparison design.

- Start from the requested parent experiment. Save its resolved config and an effective diff. Change the data and explicitly scoped refinement settings; explain any reward, model, optimizer, prompt, topology or budget change that affects comparability.
- Preserve the repository's DataSpec and ZIP/chunk conventions. Exercise the real dataset loader, audio decoding, reference field and token/truncation behavior on a small sample. Verify counts, durations, language hints and sampling weights.
- Check scorer and reward dependencies on every worker. Test actual reward behavior on representative correct and incorrect transcripts; metric CER does not automatically imply a character-level training reward. Record intentional reward changes.
- Compose Hydra configs through the actual entrypoint. Check both the absolute optimizer target and epoch bound, checkpoint/resume behavior, and validation cadence. A YAML that parses is insufficient evidence of an executable experiment.

## 5. Run authorized stages and compare

For remote submission, use the installed `run-asr-exp` skill and its current capacity, ownership, monitoring, recovery and completion requirements. Do not copy historical node or submission commands from an example. Wait for requested topology when capacity is unavailable; do not alter unrelated jobs. Record exact submission IDs and keep recovery within existing authorization.

Evaluate the original experiment and refined checkpoint with the same frozen datasets, parser, normalization, prompt and decoding settings. If initialization differs from the original baseline, include its own pre-refinement evaluation so those effects are visible. Compare matching optimizer steps or disclose unequal budgets; use N/A for unsupported deltas. Include actual validation trajectories when available. Name requested suites that did not run.

Report target-pattern severity and prevalence, resolved/persistent/introduced errors with concrete IDs and raw outputs, and guard-set regressions. Distinguish observed improvement from causal explanations. Show training selection and evaluation denominators, corpus coverage, truncation policy, leakage audit and confounds. Claim a fix only to the extent demonstrated by completed evaluation.

If another iteration is authorized, use remaining and newly introduced errors to update the hypothesis, version the pattern and training membership, and retain the frozen comparison sets. Record each iteration's parent and effective diff. Stop at the agreed success criteria, budget/iteration limit, or an external action requiring new authorization; do not silently grow the experiment campaign.

## Deliverables and finish

Keep the ledger, pattern definition, source/selection manifests, subset hashes, effective recipe diff, job identities and comparison artifacts together. Report preparation, training, evaluation, checkpoint integrity and resource pause independently; a pending stage stays pending.

Use the installed `report-experiment` skill for completed experiment comparisons, `asr-detail-compare` for utterance evidence, `openasr-report` when Excel is requested, and `obsidian-research-notes` when a note is requested. Respect existing authorization for publication; do not infer commit, push or external sharing from a reporting request.

For a concrete historical application, read [references/aishell-example.md](references/aishell-example.md). Its paths, sizes, model choices and reward settings are examples, not defaults.

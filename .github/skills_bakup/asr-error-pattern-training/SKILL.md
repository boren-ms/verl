---
name: asr-error-pattern-training
description: "Build an ASR experiment from observed dataset error patterns: compare baseline and target utterances, collect bad cases and regressions, select matching train-only examples, and create runnable data/base/ReMax configs. Use for requests such as FLEURS or MLS error-driven training subsets; launch and monitor only when requested."
---

# ASR Error Pattern Training

Turn verified ASR errors into an auditable training-data selection and a runnable
experiment. Keep three sets distinct: observed evaluation errors, observed
regressions relative to baseline, and training examples selected by error-related
features. A selected training reference is not a confirmed model error unless
that training example was actually decoded and scored.

## Resolve scope and evidence

Resolve the dataset family/locales, exact target checkpoint, baseline identity,
training-source config, donor recipe, output/version, and whether the user wants
analysis, config preparation, or launch. Reuse explicit choices from the
conversation. Resolve aliases such as “Flirs” to FLEURS from repository context;
ask only if the identity remains ambiguous.

Creating this skill or asking for a recipe does not authorize a new live job.
For an authorized launch, finish the analysis, data and effective config first,
then hand off to [verl-asr-run](../verl-asr-run/SKILL.md). Do not add exports,
external benchmarks, commits, pushes or remote-note publication without scope
support. Preserve active experiments when adding a new candidate.

Record exact baseline/target model versions and checkpoint steps. If a previously
accepted baseline differs from target initialization, retain and label that
comparison; do not imply a controlled single-factor regression experiment.
Do not substitute a different checkpoint because its artifacts are easier to use.

## Compare and collect observed errors

Use [asr-detail-compare](../asr-detail-compare/SKILL.md) for pairing, normalization,
WER accounting and HTML review. Reuse verified full CSV/summary artifacts when
identities, inputs and scorer match. For single-result analysis, follow that
skill's reference-baseline mode; model-relative regression is then unavailable.

- Prefer stable unique IDs or audio keys. Prove coverage, uniqueness and reference
  consistency before exporting subsets; do not silently use row order.
- Report full-dataset WER/error totals separately for each locale, plus improved,
  degraded and unchanged counts. Overall improvement can coexist with individual
  regressed utterances.
- Define **bad** as target error count > 0 and **regressed** as target errors minus
  baseline errors > 0 under the same scorer. Save both sets, their intersection
  semantics and selection thresholds. A requested alternate threshold takes
  precedence. Preserve zero-row locales explicitly rather than inventing examples.
- Join scores back to original dataset manifests for runnable audio/reference
  rows. Keep model hypotheses and selection annotations in separate analysis
  artifacts; never use hypotheses as training labels.
- Review representative cases per locale and distinguish numeric/time/unit
  rendering, names/acronyms/loanwords, spelling/accents/compound boundaries,
  inflection/function words, paraphrase, deletion and repetition where supported.
  Separate equivalent rendering/normalizer effects from clear content errors;
  mark hypotheses that need listening. SequenceMatcher spans are descriptive,
  not replacement WER counts or proof of error causation.

Save per-locale `data.jsonl` (bad), `regressed.jsonl`, `error_cases.jsonl`, and
`comparison_summary.json`/`error_patterns.json` under a versioned ignored workdir.
Use `tmp/<dataset>_pattern_<version>/` unless the caller provides another root.

## Find and select real training data

Trace the donor's effective training config to its actual JSONL/DataSpecs and
manifests. Verify source split and record hashes. If no eligible train split is
available, report the gap; never relabel evaluated examples as training data.
For repository chunk formats, read [references/repository-data.md](references/repository-data.md).

Translate reviewed patterns into explicit, dataset-specific selection rules:
matched error-reference lexemes, number expressions, entity/compound contexts or
other supported features. Filter frequent function words so lexeme matching does
not simply select nearly every sentence. Do not copy MLS lexemes into FLEURS or
assume accent density proves acoustic difficulty. Log rule definitions, weights,
per-locale counts and eligibility/exclusion reasons.

- Preserve the original audio, language and reference bytes. Exclude translation
  or duplicated prompt variants when the intended task is same-language ASR;
  inspect source metadata rather than guessing from names alone.
- Check overlap against the **entire relevant held-out inventory**, including
  correct utterances, not just selected bad cases. Exclude exact audio identity
  and normalized reference overlaps; record normalization and scope. Check
  speaker/recording/parallel-sentence leakage where source IDs allow it and state
  remaining uncertainty. Exact-text exclusion alone does not establish disjoint
  speakers or semantic content.
- Deduplicate deterministically; use source-group caps and locale budgets when
  justified by available data. Record candidates, selected counts and shortfalls.
  Do not pad quotas with duplicates or unrelated examples. Derive expected batch
  counts from selected data and donor settings; a tiny dataset can end before the
  configured test/save interval. Report that consequence rather than silently
  changing epochs or batch size.
- Include each selected row's stable ID, language, selection tags/score and source
  provenance. Export **two examples per language as pretty UTF-8 JSON** with
  readable reasons (or all available examples if fewer than two).

Keep a rerunnable builder with a concrete dataset/version. Existing builders are
examples, not generic CLIs: inspect hardcoded paths, schema, locale lists,
thresholds, quotas, upload flags and config destinations before adapting them.
Do not rerun an old builder against active artifacts or overwrite a published
version with different content. Publish verified data to a new versioned path
when publication is part of the task; compare uploaded bytes/hashes.

## Create a compact runnable recipe

Create only the files needed to run it:

1. One training-data YAML.
2. One combined validation-data YAML containing separate locale entries.
3. One named base derived from the exact selected donor.
4. One training entry YAML, e.g. `remax_dev_base2_fleurs_v1.yaml`.

Reuse existing files if they already satisfy the requested scope. Default for
this workflow is **bad-case diagnostic validation**; retain regression-only
artifacts for analysis without loading them unless requested. If the user asks
for an independent train/validation split, implement that instead. Always label
error-selected validation as biased diagnostic evidence, not full-dataset quality.

Preserve donor initialization, algorithm, reward formula, LoRA rank/alpha,
LR/scheduler, rollout n, batch size, epochs, save/test cadence and topology except
for requested changes. Keep training rewards in the ReMax entry YAML following
the local recipe convention; avoid copying unrelated donor data/scorer mappings.
Explicitly map each new validation source to the correct scoring function and
route training source labels to the intended reward dispatcher. Set fresh vs
resume intentionally; a new experiment must not accidentally reuse another
experiment's output/checkpoint. Resolve Hydra search paths and interpolations.

## Verify, report, and optionally launch

Before declaring data/config ready, verify:

- Selected references match source records; IDs are unique; split and overlap
  audits pass; output counts and hashes match published data when applicable.
- All referenced audio containers/files are nonempty; sampled native audio
  decoding succeeds per language. Use the actual JSONL/chunk loader and task
  formatter, including verbatim/language hints when inherited from the donor.
- Effective Hydra composition uses exactly the intended train/validation sets,
  source sampling counts, rewards, model and output. Instantiate relevant loader
  and reward paths or compare already-tested unchanged blocks. Composition alone
  is not runtime proof.

Deliver an evidence-backed report, per-locale counts/metrics, HTML comparison
links, pretty example JSON, provenance/limitations, and the runnable YAML paths.
Apply [obsidian-research-notes](../obsidian-research-notes/SKILL.md) for note/homepage
deliverables when requested or required by the comparison handoff; distinguish
local edits from remote publication. Prefer an existing matching note and retain
exact baseline/target identities and diagnostic-set caveats.

When launch is requested, delegate the run lifecycle to
[verl-asr-run](../verl-asr-run/SKILL.md): discover an actually free pool, verify both
pods and GPU-bearing Ray members, sync the exact source, run CUDA/kernel checks,
submit once, and capture actual job ID/package URI/output/W&B. Persist state under
`tmp/verl_asr_run/<entry_config>/`. Training starts only after complete finite
optimizer rows; initial validation and checkpoint restore are not new updates.
Honor existing reservations and deliberate stops.

Install a persistent monitor with tools actually available. A detached read-only
poller may capture state/events but does not provide agent autofix or chat
notifications; disclose that limitation rather than claiming scheduled agent
wakeups. Refresh validation figures via the runner's comparison handoff when new
observations arrive. Report completion only after terminal Ray success and
verified final metrics/checkpoints; a watcher observing success is not artifact
verification.

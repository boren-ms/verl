# Recipe, execution, and measured repair

## Recipe and comparison design

Record three identities independently: the model used to mine examples, the model used to initialize refinement, and the original experiment used as a comparator. Preserve full checkpoint/export revision, parser, prompt, scorer and decode settings. If the initialization differs, compare its pre-refinement outputs as well as the original baseline. Otherwise a pre-existing initialization difference can look like a training gain.

Freeze evaluation membership before refinement. Use development data for selecting patterns, thresholds and hyperparameters when available. If the user requests analysis of an existing test set, label subsequent test-derived diagnostics as adaptive and biased; retain a separate representative evaluation and disclose repeated test use.

Prepare the native training DataSpec and optional small pilot/replay/control mixture. Record mixture weights, replacement behavior and effective data exposure. The pilot need not be a new heldout split. Training and validation references must remain independently sourced and disjoint under the declared overlap checks. Record exclusions and boundary coverage; never claim audio leakage was checked merely because IDs differ.

Resolve the original parent recipe and proposed recipe using the real Hydra entrypoint; save the effective diff. Dataset changes should not silently change optimizer target, learning rate, replay, reward, initialization, prompt, precision or resume. Nested parents with their own `hydra.searchpath` may require a shared independent base; verify composition, rather than flattening guessed defaults.

Check runtime invariants on real sample records:

- Audio selectors, formats, tokenization, reference fields and language hints work in the actual reader.
- Sampling probabilities and counts reflect the intended subset or mixture.
- Scorer imports and normalizer versions work on all workers; run an actual scoring call to exercise lazy imports.
- Reward returns sensible values for exact reference, spacing/format variants, a known substitution and an invalid output. A word reward may penalize Chinese spacing despite identical canonical CER. Changing it is an experiment decision, not an automatic consequence of language.
- Maximum decode/sequence length and truncation policy are declared for mining, training and evaluation. Keep truncated rows or exclude them consistently and show counts.
- Absolute optimizer target is achievable within the epoch bound; resume and checkpoint saving refer to the intended state.

## Submission and evidence

Use `run-asr-exp` for current operational mechanics, including requested topology, capacity/ownership checks, exact submission monitoring and recovery limits. Do not treat idle GPUs or unrelated pools as available capacity. Keep a single durable monitoring handoff for a long-running case and update the ledger when ownership changes.

| Stage | Evidence required to mark complete |
| --- | --- |
| Source discovery | Actual manifests, verified record lineage and split, declared counts |
| Generation/scoring | Exact expected unique IDs, all shards, terminal job state, canonical score provenance |
| Selection | Frozen membership, exclusions, hashes, unchanged references, leakage scope |
| Recipe preparation | Effective diff, real loader/scorer/reward checks, budget/resume settings |
| Training | Exact submission reached intended optimizer target; required validation/checkpoint evidence |
| Evaluation | Completed requested sets and sample coverage, parser/scorer identity, raw details |
| Fix claim | Comparable baseline/refinement results on frozen diagnostics and broader guards |
| Resource pause | Fresh ownership checks and verified pause state for the specifically authorized pool |

A success status below the optimizer target is an early exit. A tracker entry or existing directory does not prove checkpoint completeness. Missing evaluations and resource pause are reported separately from training success. Do not launch retries, cleanup or extra ablations outside the authorized scope.

## Report the repair

Use `report-experiment` for run-level evidence and `asr-detail-compare` for utterance-level reports. Include:

1. The error hypothesis, concrete baseline examples and retrieval rule/version.
2. Source lineage, searched/decoded universe, candidate and selected counts/duration, human-label provenance, leakage scope and truncation policy.
3. Original experiment, selection checkpoint, initialization and refined checkpoint; effective config diff and training budget/exposure.
4. Target-pattern diagnostics, full/representative metrics and guard sets with sample and edit/reference denominators. Show individual components of the summary validation bundle; use explicit weighting if an aggregate is justified.
5. Baseline-to-refinement resolved, persistent and newly introduced errors, with raw hypotheses and stable IDs. Distinguish pattern prevalence from conditional severity.
6. Comparable validation trajectories, missing suites and confounds. Different optimizer budgets require clearly labeled endpoints and N/A for unsupported matched-budget claims.
7. Whether evidence supports a fix, only a preparation milestone, or an inconclusive/negative result; link configs, manifests, details and exact submissions.

Avoid attributing improvement specifically to retrieval when reward, initialization or other settings changed without a controlled comparison. Recommend an ablation when useful; run it only within the requested scope. Use Excel or Obsidian skills only for the requested artifact destinations and preserve existing manual content.

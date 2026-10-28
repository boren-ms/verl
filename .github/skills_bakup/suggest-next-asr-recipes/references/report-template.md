# Next Recipe Candidates: <project>

## Recommendation

<Plain-language conclusion: what validation p_err indicates, what to test next,
and why. Distinguish observed results from hypotheses. If ongoing runs already
cover the question, recommend waiting instead of a duplicate. If blocked, name
the missing evidence and do not claim any candidates are validated.>

- Snapshot: <start/end UTC; live API or dated offline source; completeness>
- W&B project: <link>
- Recipes: <directory link>; source revision: <revision and relevant dirty state>
- Evidence: <evidence.json link>
- Created: <number> YAMLs; launched: **none**

## Objective and Comparability

- Exact metric keys: <val-core/val-aux dataset p_err/mean@1 keys>
- Target datasets and fixed weights: <list; identify lexical/verbatim separately>
- Aggregate: <arithmetic mean or explicit user weights>; **lower is better**
- Display: `p_err * 100%`; deltas in percentage points, negative is better
- Validation manifests/subsets/reference style/scorer/aggregation/decoding: <identities>
- Cohorts and exclusions: <incompatibilities, missing provenance, missing p_err>
- Matched comparison budget and success guardrails: <steps; per-dataset tolerance>

## Experiment Coverage

Include all project runs, not just successful examples. Keep each run ID separate.

| Run / W&B link | State / last activity | Recipe / mapping confidence | Model and key settings | Latest training step / budget | Latest complete validation step | Cohort / eligibility / reason |
|---|---|---|---|---|---|---|
| <run> | <finished/running/etc.> | <link or unmapped> | <data, LR, n, rank, seed> | <step / budget> | <step or missing> | <comparable / provisional / excluded> |

## Validation `p_err` Evidence

Use one row per run, observation type, and dataset. Every aggregate row must be
calculated from the complete dataset vector at that same step. Include initial,
best trained, latest complete, and matched-budget observations. Mark unavailable
values as `missing`, never zero.

| Run | Observation | Step | Dataset | p_err (%) | Delta vs matched control (pp) | Evidence / caveat |
|---|---|---|---|---|---|---|
| <run> | <initial/best/latest/matched> | <step> | <dataset or fixed-set mean> | <value> | <delta or not comparable> | <source key; running provisional> |

### What the trajectories say

<Last three complete points, stability/regression, per-dataset trade-offs,
step-0 comparison, matched-step findings, repetitions/uncertainty, and confounds.
Show both datasets when they move in opposite directions. A lower aggregate
must not hide a regression. Explain partial or missing later observations.>

## Proposed Candidates in Priority Order

| Priority | YAML | Control / parent | Exact factor changed | Validation p_err evidence | Hypothesis / why now | Resources | Status |
|---|---|---|---|---|---|---|---|
| <1> | <link> | <recipe and run links> | <key: old -> new> | <run/step/dataset/value> | <testable explanation> | <GPUs and expected cost direction> | <created and config-validated / blocked> |

### <Candidate filename>

Repeat this explanation for **every** candidate:

1. **Observation:** <Cited runs, exact validation steps and per-dataset p_err;
   state whether evidence is complete or provisional.>
2. **Hypothesis:** <Explain in plain language why this change might address that
   observation. Separate causal speculation from measured improvement.>
3. **Config changes:** <Dotted-key before/after table including necessary coupled
   changes; explain unchanged initialization, validation protocol, and budget.>
4. **Why this candidate next:** <Why better information/value than alternatives;
   confirm no running/completed/local recipe already performs this test.>
5. **Success/failure rule:** <Exact target set, aggregate, matched control steps,
   per-dataset regression guardrail, confirmation cadence, and training budget.
   Do not promise an unmeasured numerical gain.>
6. **Risks and limits:** <Instability, data/reward confounds, seed uncertainty,
   cost, running-run maturity; specify what outcome would refute the hypothesis.>
7. **Validation:** <Hydra composition/resolution result with the launch-time
   experiment-name override, intended resolved diff, filename/launcher
   consistency, unique output roots; GPU execution not tested.>

## Deferred or Rejected Ideas

| Idea | Reason | Evidence needed to reconsider |
|---|---|---|
| <idea> | <already running / weak evidence / p_err regression / incompatible cohort> | <specific validation or provenance gap> |

## Artifacts and Limitations

- Report: <link>
- Evidence snapshot and resolved diffs: <link>
- Candidate YAMLs: <one link per actual file>
- Validation commands and results: <actual checks performed, not intended checks>
- Remaining limitations: <auth/data coverage/provenance/resource validation gaps>
- No training jobs, remote syncs, workload-cache edits, or monitors were started.

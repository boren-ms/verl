# BCR16 T1 versus its no-replay T1 baseline
Training STOPPED; monitor deleted; exact driver10000000 has0 GPU allocation.
## Shared validation step600
| Dataset | Baseline p_err | BCR16 T1 p_err | Delta pp |
|---|---:|---:|---:|
| OpenML bad | 7.9695% | 7.9354% | -0.0341 |
| Earnings (6 parents) | 4.4560% | 4.4755% | +0.0195 |
| LibriSpeech clean | 1.2056% | 1.1603% | -0.0453 |
| LibriSpeech other | 2.6476% | 2.5681% | -0.0795 |

## Run identities
- Baseline: L2 T1: [remax_dev_base0_v1h2a_fix_L2_t1](https://msaip.wandb.io/genai/v2611/runs/d807d91db6c775c7d097c5ef7ae2f734), ID `d807d91db6c775c7d097c5ef7ae2f734`. W&B tracking state `crashed`; exact Ray state takes precedence. Complete unfiltered scan: 619 rows.
- BCR16 T1: [remax_dev_base0_v1h2a_fix_L2_RCB16_T1](https://msaip.wandb.io/genai/v2611/runs/73acea8095b222e76c04affb7133c92e), ID `73acea8095b222e76c04affb7133c92e`. W&B tracking state `running`; exact Ray state takes precedence. Complete unfiltered scan: 755 rows.

## Initial, latest and independently best validation
|Dataset|Run|Initial|Latest|Best|
|---|---|---:|---:|---:|
|OpenML bad|Baseline: L2 T1|15.0080% @0|7.9695% @600|7.6486% @260|
|OpenML bad|BCR16 T1|15.0080% @0|7.6882% @740|7.6868% @480|
|Earnings (6 parents)|Baseline: L2 T1|4.8392% @0|4.4560% @600|4.4105% @540|
|Earnings (6 parents)|BCR16 T1|4.8392% @0|4.4365% @740|4.3326% @380|
|LibriSpeech clean|Baseline: L2 T1|1.2622% @0|1.2056% @600|1.1961% @380|
|LibriSpeech clean|BCR16 T1|1.2622% @0|1.1282% @740|1.1282% @740|
|LibriSpeech other|Baseline: L2 T1|2.7270% @0|2.6476% @600|2.6476% @360|
|LibriSpeech other|BCR16 T1|2.7270% @0|2.5681% @740|2.5473% @420|

## Reward means: exact shared steps591–600
Raw measures and deltas have the same definitions; core/score includes divisor4 vs3 and is not directly comparable. Different replay batches limit interpretation.
|Exact key|Baseline mean|BCR16 mean|n per run|
|---|---:|---:|---:|
|`reward-core/char/mean`|0.91512812|0.85482949|10|
|`reward-core/fmt/mean`|0.99941406|1.00000000|10|
|`reward-core/lang/mean`|0.96865234|0.99921875|10|
|`reward-core/score/mean`|0.00025047|0.00389774|10|
|`reward-core/word/mean`|0.87786692|0.80055793|10|
|`reward-core/word_del/mean`|0.94881243|0.91423600|10|
|`reward-core/word_ins/mean`|0.98533104|0.98174456|10|
|`reward-core/word_kw/mean`|0.99264528|0.99372683|10|
|`reward-delta/char/mean`|-0.00553793|-0.00216827|10|
|`reward-delta/fmt/mean`|-0.00058594|0.00000000|10|
|`reward-delta/lang/mean`|-0.00009766|-0.00078125|10|
|`reward-delta/word/mean`|-0.01156129|-0.00507596|10|
|`reward-delta/word_del/mean`|-0.00209833|-0.00230257|10|
|`reward-delta/word_ins/mean`|-0.00345749|-0.00051080|10|
|`reward-delta/word_kw/mean`|-0.00052769|-0.00070280|10|
|`reward-vote/char/mean`|-0.00553793|-0.00216827|10|
|`reward-vote/fmt/mean`|0.99941406|1.00000000|10|
|`reward-vote/lang/mean`|0.96865234|0.99921875|10|
|`reward-vote/word/mean`|-0.01151247|-0.00507596|10|
|`reward-vote/word_del/mean`|-0.00209833|-0.00230257|10|
|`reward-vote/word_ins/mean`|-0.00335983|-0.00051080|10|
|`reward-vote/word_kw/mean`|-0.00052769|-0.00070280|10|

## Full matched example scope
|Dataset|Matched units|Improved|More errors|Net added errors|Changed n_ref|
|---|---:|---:|---:|---:|---:|
|OpenML bad|393|27|20|-2|2|
|Earnings (6 parents)|6|2|4|+3|0|
|LibriSpeech clean|2620|50|28|-24|0|
|LibriSpeech other|2939|99|62|-42|0|

## Verified scoring diagnostic
Pinned scorer replay reproduced12/12 selected case error counts. OpenML ID11746087412974645092 loses the parenthesized year2006 during normalization,adding4 errors(2→6). This is one verified scoring-sensitive case,not a population prevalence estimate. See sources/scoring_diagnostics.json.

## Limits
- Baseline is L2_t1 without replay, temperature1; target is BCR16_T1 temperature1. Reward divisor changes4 to3 as well as replay; causal isolation unavailable.
- Both runs were explicitly user-stopped. Target current recovery750 restored but stopped during initial validation before any new optimizer update; historical last finite optimizer755.
- Target multiple infrastructure recoveries reset replay; W&B reuses one identity and may omit repeated steps. Monotonic W&B history is not a single uninterrupted lineage. Immutable per-segment startup proofs retained in source state.
- Equal optimizer steps do not imply equal fresh-data exposure: baseline128 fresh, target112 fresh plus up to16 replay. Training batches may differ.
- OpenML hypothesis-dependent normalization can change n_ref; do not assume fixed denominator or replace official p_err.
- Initial/latest/best are descriptive; best steps differ across datasets and latest budgets differ. Selected table examples are not prevalence estimates.
- All validation examples matched by stable identity plus identical reference; raw output differences are descriptive, not official edit alignment or acoustic diagnoses.

## Figures and slide narrative
## Replay changes sampling and reward scale

Pinned source audit and collected W&B configs; live stop verified by exact Ray submission.

![figure](figures/01_config.png)

## Validation trajectories — preserve dataset tradeoffs

Observed points only; lines are guides. Target latest740 vs baseline600 is an unequal-budget comparison. Initial/latest/best shown separately.

![figure](figures/02_validation_0.png)

## Validation trajectories — preserve dataset tradeoffs

Observed points only; lines are guides. Target latest740 vs baseline600 is an unequal-budget comparison. Initial/latest/best shown separately.

![figure](figures/02_validation_1.png)

## At step600: LS gains; Earnings slightly regresses

Exact shared step600. No cross-dataset averaging; single-run descriptive comparison with reward-scale and exposure confounds.

![figure](figures/03_matched.png)

## reward-core: raw batch means through step600

No smoothing/interpolation. Logged history includes abandoned pre-recovery updates; all22 keys retained. Batches differ by replay.

![figure](figures/reward-core_traces.png)

## reward-core: exact shared window591–600

10 observations per run per component; arithmetic means. Core score has a different divisor and is not a like-for-like quality metric.

![figure](figures/reward-core_means.png)

## reward-delta: raw batch means through step600

No smoothing/interpolation. Logged history includes abandoned pre-recovery updates; all22 keys retained. Batches differ by replay.

![figure](figures/reward-delta_traces.png)

## reward-delta: exact shared window591–600

10 observations per run per component; arithmetic means. Core score has a different divisor and is not a like-for-like quality metric.

![figure](figures/reward-delta_means.png)

## reward-vote: raw batch means through step600

No smoothing/interpolation. Logged history includes abandoned pre-recovery updates; all22 keys retained. Batches differ by replay.

![figure](figures/reward-vote_traces.png)

## reward-vote: exact shared window591–600

10 observations per run per component; arithmetic means. Core score has a different divisor and is not a like-for-like quality metric.

![figure](figures/reward-vote_means.png)

## Matched examples quantify improvements and regressions

Step600 full dumps; unchanged stable IDs and references. Earnings grouped341 chunks into6 parents; OpenML393,LS clean2620,LS other2939.

![figure](figures/07_patterns.png)

## Stable-ID cases: content and scoring-sensitive forms

Excerpts are illustrative; complete outputs, references and logged error counts retained in evidence.json. No acoustic cause inferred.

![figure](figures/08_examples_0.png)

## Stable-ID cases: content and scoring-sensitive forms

Excerpts are illustrative; complete outputs, references and logged error counts retained in evidence.json. No acoustic cause inferred.

![figure](figures/08_examples_1.png)

## Stable-ID cases: content and scoring-sensitive forms

Excerpts are illustrative; complete outputs, references and logged error counts retained in evidence.json. No acoustic cause inferred.

![figure](figures/08_examples_2.png)

## Earnings: six parents, small net change hides mixed effects

Pinned source audit and collected W&B configs; live stop verified by exact Ray submission.

![figure](figures/08_earnings.png)

## Check scoring-sensitive changes before tuning rewards

Pinned source audit and collected W&B configs; live stop verified by exact Ray submission.

![figure](figures/09_next.png)

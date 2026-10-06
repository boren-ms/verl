# ReMax `first_diff`: advantage-to-loss flow

## Overview

For [this recipe](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff.yaml), the flow is:

```text
128 prompts
  |-- 8 sampled responses/prompt -> sampled rewards
  `-- 1 greedy response/prompt   -> baseline reward
                 |
       ReMax: sampled reward - greedy reward
                 |
       Zero advantage before first token difference
                 |
       Token-level PPO policy loss + TIS weighting
                 |
       Add reference KL loss over ALL valid response tokens
                 |
       Microbatch scaling -> backward -> gradient clipping -> optimizer step
```

The settings below were verified through Hydra composition and implementation tracing.
Two important details: **`first_diff` masks advantages, not the loss's denominator or KL**,
and **this configuration normally takes the actor's on-policy branch, making the PPO
ratio numerically 1**.

### Available aggregation variants

All three recipes keep `algorithm.remax_advantage_mask: first_diff`. The two
variants differ from the original only in `actor_rollout_ref.actor.loss_agg_mode`:

| Recipe | Actor loss aggregation |
| --- | --- |
| [Original](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff.yaml) | `token-mean` |
| [Sequence token-sum variant (`smts`)](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff_smts.yaml) | `seq-mean-token-sum` |
| [Sequence token-mean variant (`smtm`)](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff_smtm.yaml) | `seq-mean-token-mean` |

The main walkthrough describes the original `token-mean` recipe. Section 6
compares the alternative reductions; each variant applies its selected mode to
both policy loss and actor-side KL loss.

The related [edit-boundary token-sum recipe (`edit_smts`)](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_edit_smts.yaml)
keeps `seq-mean-token-sum` but changes the advantage mask to `edit_boundary`.
It selects unmatched sampled tokens and deletion-boundary tokens rather than
the entire suffix from the first difference. The `first_diff` mask behavior
described below does not apply to that variant.

## 1. Sampled reward and greedy baseline

The [DAPO trainer](../../../dapo/dapo_ray_trainer.py#L178) generates eight samples
per prompt at temperature **1.2**, plus one greedy baseline using validation
generation settings: temperature **0**, maximum **256 tokens**. Sampled responses
can use up to **768 tokens**.

Both are scored with the reward function associated with their data source.
The component scores `char`, `word`, `lang`, `fmt`, and `bracket` are clipped
to `[-1, 1]` by default.

All equations below use plain-text notation: `*` means multiplication, `/`
means division, `[t]` selects token position `t`, and `sum(...)` adds the
specified values. `clamp(value, low, high)` limits a value to that interval.

For earnings:

```text
earnings_reward = (char + word + 0.5 * lang + 0.5 * fmt) / 4
```

For OpenML:

```text
openml_reward = (char + word + 0.5 * lang + 0.5 * fmt + bracket) / 4
```

**Both divide by 4**, even though OpenML has five components. This is a fixed
divisor, not normalization by the component count or sum of weights.
See [reward reduction](../../reward/asr_measure.py#L396) and
[component scaling](../../reward/asr_measure.py#L431).

The [reward manager](../../../../verl/workers/reward_manager/dapo.py#L140)
places the scalar reward on the final valid response token; earlier token
rewards are zero. Overlong penalties are disabled.

### How `reward_baselines` is computed

`reward_baselines` is **the reward of each prompt's greedy response under the
current rollout policy**, computed before the actor update. It is not a critic
prediction, a reference-model reward, or an average of the eight sampled rewards.

The [baseline-generation and scoring path](../../../dapo/dapo_ray_trainer.py#L178)
does the following:

1. Copy the prompt batch and set `meta_info["validate"] = True`.
2. Generate one greedy response per prompt using the current actor/rollout
   weights. The [validation generation branch](../../../../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L362)
   sets `n=1` and uses this recipe's validation settings: temperature **0** and
   maximum **256 tokens**. This flag selects decoding settings; it does not
   substitute validation-dataset prompts.
3. Combine those responses with the original prompt metadata and call
   `self.reward_fn(new_batch, return_dict=True)`. The reward manager decodes
   each greedy response and scores it against that prompt's ground truth using
   the corresponding earnings or OpenML reward function.
4. Take the returned token-level `reward_tensor` and sum along the response
   dimension. Because this recipe places only one scalar reward on the final
   valid token, the sum simply recovers that response's scalar reward.
5. Store the result as `new_batch.batch["reward_baselines"]`, then repeat each
   prompt row **eight times, interleaved**, to align it with the sampled rows.

The key operations in the trainer are:

```python
baseline_result = self.reward_fn(new_batch, return_dict=True)
reward_baseline_tensor = baseline_result["reward_tensor"]
reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
new_batch.batch["reward_baselines"] = reward_baseline_tensor

new_batch = new_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,
    interleave=True,
)
new_batch = new_batch.union(gen_batch_output)
```

For a concrete earnings example, suppose a greedy response has component scores:

```text
char = 0.80, word = 0.80, lang = 1.00, fmt = 1.00

Greedy reward = (0.80 + 0.80 + 0.5 * 1.00 + 0.5 * 1.00) / 4
              = 0.65
```

With five valid response tokens and two padding positions, the reward manager
returns `[0, 0, 0, 0, 0.65, 0, 0]`; summing that row gives baseline **0.65**.
This is a sum over tokens, **not** division by the greedy response length.

For two prompts whose greedy rewards are **0.65** and **0.50**, respectively:

```text
Before repeat:
  reward_baselines shape: [2]
  values: [0.65, 0.50]

After repeat with n=8:
  reward_baselines shape: [16]
  values: [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
           0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
  rows:   [prompt 0 samples 0..7, prompt 1 samples 0..7]
```

The real nominal batch has **128** greedy baselines, expanded to **1024**
sample-aligned entries. Subsequent batch reordering/filtering carries each
baseline along with its sampled row.
The [advantage dispatcher](../../../../verl/trainer/ppo/ray_trainer.py#L266)
passes this tensor to `compute_remax_outcome_advantage`, where a prompt-0 sample
with reward **0.80** gets raw advantage **0.80 - 0.65 = 0.15**.

The baseline is regenerated for each training batch with the current policy;
it is neither a running average nor a baseline fixed at the initial checkpoint.
The `first_diff` mask does not change its computation.

## 2. Compute the ReMax advantage

For each sampled response, use the greedy reward from the same prompt:

```text
reward_difference = sampled_reward - greedy_reward
```

The [ReMax implementation](../../../../verl/trainer/ppo/core_algos.py#L950)
first computes reverse cumulative token rewards, then subtracts the greedy
baseline. With this terminal-only reward, every valid token initially receives
the same advantage:

```text
raw_advantage[t] = reward_difference    # At each valid response token
```

For this YAML:

- No ReMax L2/RMS normalization is configured.
- No binary/sign-only advantage transformation is configured.
- Advantage scale is **1.0** for both data sources.
- No KL is subtracted from rewards.
- No critic or GAE is needed.
- The eight samples **do not** supply a GRPO group-mean baseline; each uses the
  same prompt's greedy reward.

The inherited `norm_adv_by_std_in_grpo: true` does not normalize this ReMax path.

### Worked example: reverse cumulative rewards

The exact expression is:

```python
returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
```

`dim=-1` is the response-token dimension. Each batch row is processed
independently. Multiplication removes rewards at invalid positions; reversing,
taking a cumulative sum, and reversing again computes the reward remaining
from each token through the end:

```text
return[t] = sum(reward[u] * response_mask[u] for u from t to the last token)
```

Consider one response with five valid tokens, two trailing padding positions,
and a scalar reward of **0.80** on its final valid token:

```text
Response tokens:                 A     B     X     D    EOS   PAD   PAD
token_level_rewards:           [0.00, 0.00, 0.00, 0.00, 0.80, 0.00, 0.00]
response_mask:                 [1,    1,    1,    1,    1,    0,    0   ]

1. rewards * mask:             [0.00, 0.00, 0.00, 0.00, 0.80, 0.00, 0.00]
2. flip token dimension:       [0.00, 0.00, 0.80, 0.00, 0.00, 0.00, 0.00]
3. cumulative sum:             [0.00, 0.00, 0.80, 0.80, 0.80, 0.80, 0.80]
4. flip back -> returns:       [0.80, 0.80, 0.80, 0.80, 0.80, 0.00, 0.00]
```

Every valid token has return **0.80**, because its remaining trajectory includes
the terminal reward. The trailing padding positions have return zero.
This is an undiscounted sum, not an average or an immediate-token reward.

Now suppose the greedy response's reward is **0.65**.
`reward_baselines` has shape `[batch_size]`; `unsqueeze(-1)` changes it to
`[batch_size, 1]`, so each row's scalar baseline broadcasts across its tokens.
Multiplication by `response_mask` keeps that subtraction off padding:

```text
returns:                       [0.80, 0.80, 0.80, 0.80, 0.80, 0.00, 0.00]
baseline * response_mask:      [0.65, 0.65, 0.65, 0.65, 0.65, 0.00, 0.00]
raw advantages:                [0.15, 0.15, 0.15, 0.15, 0.15, 0.00, 0.00]
```

If the greedy tokens are `A B C D EOS`, the first difference is `X` versus `C`.
The next stage applies:

```text
first_diff mask:               [0,    0,    1,    1,    1,    0,    0   ]
final advantages:              [0.00, 0.00, 0.15, 0.15, 0.15, 0.00, 0.00]
```

The `first_diff` stage changes **advantages only**; the stored `returns` remain
`[0.80, 0.80, 0.80, 0.80, 0.80, 0.00, 0.00]`.

## 3. Apply `first_diff`

### Where `compute_remax_disagreement_mask` is called

The production call chain is:

```text
RayDAPOTrainer.fit()
  -> _attach_remax_disagreement_mask(...)
     -> compute_remax_disagreement_mask(...)
        -> sampled.batch["remax_advantage_mask"]

Later: compute_advantage(...)
  -> compute_remax_outcome_advantage(...)
  -> multiply advantages by the stored remax_advantage_mask
```

The outer call is in
[RayDAPOTrainer.fit](../../../dapo/dapo_ray_trainer.py#L234), inside the ReMax
greedy-baseline branch:

```python
mask_mode = self.config.algorithm.get("remax_advantage_mask")
if mask_mode is not None:
    _attach_remax_disagreement_mask(
        gen_batch_output,
        gen_baseline_output,
        self.config.actor_rollout_ref.rollout.n,
        mode=mask_mode,
    )
```

For this recipe, the arguments are the **eight sampled responses per prompt**,
the **one greedy response per prompt**, `rollout_n=8`, and `mode="first_diff"`.
The call runs after greedy generation and baseline scoring, before merging the
sampled outputs into the repeated prompt batch and scoring the sampled responses.

The direct call to the function is in
[_attach_remax_disagreement_mask](../../../dapo/dapo_ray_trainer.py#L71):

```python
sampled.batch["remax_advantage_mask"] = compute_remax_disagreement_mask(
    sampled.batch["responses"],
    compute_response_mask(sampled),
    baseline.batch["responses"],
    compute_response_mask(baseline),
    baseline_index=np.arange(len(sampled)) // rollout_n,
    mode=mode,
)
```

`baseline_index` maps each sampled row to its prompt's greedy row. For two
prompts with eight samples each:

```text
Sampled rows:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Greedy row:    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1,  1,  1,  1,  1,  1,  1]
```

The returned mask has the same shape as sampled response IDs and their validity
mask: `[number_of_sampled_responses, response_length]`. It is carried with the
sampled batch through subsequent merging, reordering, and filtering.

The token comparison is **not performed inside**
`compute_remax_outcome_advantage` or the actor loss. Later,
[compute_advantage](../../../../verl/trainer/ppo/ray_trainer.py#L298) validates
the stored mask, computes raw ReMax advantages, and
[multiplies those advantages by the mask](../../../../verl/trainer/ppo/ray_trainer.py#L371).
The actor receives the already-masked advantages.

Other direct calls are in the
[core algorithm tests](../../../../tests/trainer/ppo/test_core_algos_on_cpu.py)
and helper calls are covered by the
[DAPO mask tests](../../../../tests/recipe/dapo/test_remax_advantage_mask.py).

### Mask behavior

The [mask implementation](../../../../verl/trainer/ppo/core_algos.py#L771)
compares **valid token IDs positionally**, not words or decoded strings.

Let `first_difference` be the first differing position. The mask is:

```text
first_diff_mask[t] = 0    if t is before first_difference
first_diff_mask[t] = 1    if t is at/after first_difference and the token is valid
first_diff_mask[t] = 0    if the token is padding/invalid
```

The [advantage dispatcher](../../../../verl/trainer/ppo/ray_trainer.py#L371)
then applies:

```text
A[t] = raw_advantage[t] * first_diff_mask[t]
     = reward_difference * first_diff_mask[t]
```

`A[t]` is the final advantage used by the actor loss.

Example:

```text
Greedy:       A    B    C    D    EOS
Sample:       A    B    X    D    EOS
first_diff:   0    0    1    1     1

Sample reward = 0.80; greedy reward = 0.65
Advantage:    0    0   0.15 0.15  0.15
```

Notice that `D` and `EOS` remain selected even though they match again:
**the entire suffix is selected after the first difference**.

Edge cases:

- Identical responses: all-zero mask.
- Sample is a shorter matching prefix: all-zero mask.
- Sample extends a matching greedy response: select the extra sampled tokens.
- Padding: never selected.

EOS participates if it is included in the valid response mask. Consequently,
ending early with an EOS that differs from the greedy continuation can create
a first difference.

## 4. Compute token log-probabilities

The trainer initially uses rollout log-probabilities as `old_log_probs`, because
`use_rollout_log_probs_as_old: true`. It also computes reference-policy
log-probabilities. See
[trainer log-probability preparation](../../../dapo/dapo_ray_trainer.py#L386).

The actor forward pass computes current token log-probabilities with logits
divided by the training temperature, **1.2**.
See [actor forward](../../../../verl/workers/actor/dp_actor.py#L198).

There are three distinct quantities:

- `log_prob[t]`: current actor log-probability.
- `rollout_log_prob[t]`: stored rollout-engine log-probability.
- `ref_log_prob[t]`: reference log-probability.

`old_log_prob[t]` is the detached comparison log-probability used in PPO.
`exp(value)` is the exponential function.

Because LoRA is enabled, the reference is the actor's base model with adapters
disabled, not a moving greedy-policy snapshot.
See [reference selection](../../../../verl/trainer/ppo/ray_trainer.py#L466).

## 5. PPO policy loss and importance weighting

The configured loss is
[vanilla token-level PPO](../../../../verl/trainer/ppo/core_algos.py#L1255).

In its general form:

```text
log_ratio[t] = clamp(log_prob[t] - old_log_prob[t], -20, 20)
ratio[t] = exp(log_ratio[t])
```

The asymmetric clipping interval is **[0.8, 1.28]**:

```text
unclipped_loss[t] = -A[t] * ratio[t]
clipped_loss[t]   = -A[t] * clamp(ratio[t], 0.8, 1.28)
ppo_loss[t]       = max(unclipped_loss[t], clipped_loss[t])
```

For negative advantages, dual clipping with coefficient **3.0** adds:

```text
dual_clip_loss[t] = min(ppo_loss[t], -3 * A[t])    if A[t] < 0
dual_clip_loss[t] = ppo_loss[t]                   if A[t] >= 0
```

TIS then weights this loss:

```text
tis_weight[t]    = min(exp(old_log_prob[t] - rollout_log_prob[t]), 5)
PG_token_loss[t] = tis_weight[t] * dual_clip_loss[t]
```

### Important: the actual on-policy branch here

The [FSDP worker](../../../../verl/workers/fsdp_workers.py#L277) converts
the configured minibatch into trajectory units and divides across
data-parallel workers:

```text
Global: 128 prompts x 8 samples = 1024 trajectories
Workers: 4 nodes x 8 GPUs = 32
Nominal local minibatch: 1024 / 32 = 32 trajectories
```

With one minibatch and `ppo_epochs: 1`, the
[actor](../../../../verl/workers/actor/dp_actor.py#L418) sets:

```python
old_log_prob = log_prob.detach()
```

Therefore, on this normal path:

```text
old_log_prob[t]  = detach(log_prob[t])
ratio[t]        = 1
dual_clip_loss[t] = -A[t]
tis_weight[t]   = min(exp(detach(log_prob[t]) - rollout_log_prob[t]), 5)
PG_token_loss[t] = -tis_weight[t] * A[t]    # Numerical loss value
```

**PPO clipping is inactive, but the policy gradient is not zero.**
The old log-probability is detached, so the ratio still has a gradient through
the current log-probability:

```text
gradient(PG_token_loss[t]) = -tis_weight[t] * A[t] * gradient(log_prob[t])
```

Here `gradient(...)` means the derivative with respect to the trainable model
parameters. `detach(...)` keeps the value but stops gradients through that
branch. The gradient above comes from the original PPO ratio expression;
it would be lost if the implementation replaced that ratio with a literal `1`.

Positive advantage encourages the sampled suffix; negative advantage discourages
it. TIS can still differ from 1 because actor and rollout-engine log-probabilities
can differ.

## 6. Aggregate with `token-mean`

For each microbatch,
[loss aggregation](../../../../verl/trainer/ppo/core_algos.py#L1151) computes:

```text
PG_loss = sum(response_mask * PG_token_loss) / sum(response_mask)
```

Here `response_mask` is the **full valid response mask**, not `first_diff`. The sums
cover all response positions and trajectories in the microbatch.

Thus, prefix tokens have zero policy-loss contribution, but **still count in
the denominator**. In the earlier five-token example, with ratio and TIS weight
both equal to **1**:

```text
PG_loss = (0 + 0 - 0.15 - 0.15 - 0.15) / 5 = -0.09
```

It is **not** **-0.15**, which would result from averaging only the selected
suffix.

### Comparing all `loss_agg_mode` choices

The [aggregation implementation](../../../../verl/trainer/ppo/core_algos.py#L1151)
supports four modes. They change how the same token-loss matrix is reduced;
they do not change rewards, advantages, `first_diff`, PPO clipping, or TIS.

Let:

- `B`: number of trajectories in the current microbatch.
- `T`: response-mask tensor width, including padding columns.
- `n[i] = sum(response_mask[i])`: valid response-token count for trajectory `i`.
- `S[i] = sum(response_mask[i] * token_loss[i])`: its sum of valid token losses.

In the table below, `sum(S)` and `sum(n)` add across trajectories, and
`mean(...)` is the arithmetic average across trajectories.

Then:

| Mode | Exact microbatch reduction | Interpretation |
| --- | --- | --- |
| `token-mean` | `sum(S) / sum(n)` | Average over all valid tokens; equivalent to weighting each sequence's token mean by its valid length. |
| `seq-mean-token-sum` | `sum(S) / B` | Average each sequence's total loss, without dividing by its length. |
| `seq-mean-token-mean` | `mean(S[i] / n[i])` | Average each sequence's token mean; equal weight per sequence regardless of length. |
| `seq-mean-token-sum-norm` | `sum(S) / T` | Divide the whole microbatch's token-loss sum by response tensor width, not by valid-token count or sequence count. |

Despite its name, **`seq-mean-token-sum-norm` does not divide by the sequence
count `B` in this implementation**. Its denominator is `loss_mask.shape[-1]`, not necessarily the
configured maximum response length; that width should be held constant for
consistent normalization. `seq-mean-token-mean` requires positive valid lengths:
the implementation does not clamp its per-sequence denominator.

### Shared numerical sample

Assume the normal on-policy branch with PPO ratio and TIS weight both **1**,
so a selected token's policy loss is `-A[t]`. Use two responses padded to
width `T = 5`:

```text
Sequence 1: 4 valid tokens, advantage +0.20, first difference at token 3
  response_mask: [1,     1,     1,     1,     0   ]
  first_diff:    [0,     0,     1,     1,     0   ]
  PG token loss: [0.00,  0.00, -0.20, -0.20,  0.00]
  n[0] = 4; S[0] = -0.40

Sequence 2: 2 valid tokens, advantage -0.10, first difference at token 2
  response_mask: [1,     1,     0,     0,     0   ]
  first_diff:    [0,     1,     0,     0,     0   ]
  PG token loss: [0.00, +0.10,  0.00,  0.00,  0.00]
  n[1] = 2; S[1] = +0.10

B = 2; T = 5; total valid tokens = 6; total PG loss sum = -0.30
```

| Mode | Calculation | PG loss |
| --- | --- | --- |
| `token-mean` | `(-0.40 + 0.10) / (4 + 2)` | **-0.050** |
| `seq-mean-token-sum` | `(-0.40 + 0.10) / 2` | **-0.150** |
| `seq-mean-token-mean` | `((-0.40 / 4) + (0.10 / 2)) / 2` | **-0.025** |
| `seq-mean-token-sum-norm` | `(-0.40 + 0.10) / 5` | **-0.060** |

Under `seq-mean-token-mean`, sequence 1 contributes token mean **-0.10**,
sequence 2 contributes **+0.05**, and the two sequence means are averaged.
Under `token-mean`, the same means are weighted by lengths **4** and **2**:
the longer sequence gets twice the weight.

For `seq-mean-token-sum`, each selected token has weight `1 / B`, so more
selected suffix tokens produce a proportionally larger sequence contribution
when advantage and importance weights are otherwise equal. It avoids dilution
by that sequence's matching prefix, but does not normalize suffix length.

For both mean-over-token modes, matching prefix tokens still count in valid
lengths despite their zero PG loss. None of the four modes averages only over
`first_diff`-selected tokens.

These values illustrate different weighting and scale, not model quality:
a more negative number here does not establish that a mode is better.

### KL and microbatch effects when changing the mode

The [actor](../../../../verl/workers/actor/dp_actor.py#L475) uses the same mode
for policy loss and reference KL loss, and for entropy if enabled. The objective
remains `PG + 0.001 * KL`; changing the mode changes both reductions.
For this recipe, KL still includes all valid response tokens.

The actor additionally multiplies each dynamic microbatch loss by its
trajectory count divided by configured local minibatch size. Consequently:

- `seq-mean-token-sum` and `seq-mean-token-mean` preserve their corresponding
  local minibatch sequence averages across microbatch partitions, assuming
  the full configured minibatch is present.
- `token-mean` becomes a trajectory-count-weighted average of microbatch token
  means, not necessarily the full minibatch token mean.
- `seq-mean-token-sum-norm` already sums across sequences before actor scaling;
  because it has no division by microbatch sequence count, its effective scale
  can depend on the partition even if `T` is unchanged.

For example, keep `T = 5` and configured local minibatch size **2**. Process the
shared sample above either together or as two single-sequence microbatches.
Each single-sequence microbatch receives actor scale **1/2**:

| Mode | Together: scale 1 | Separate: sum of two scaled losses |
| --- | --- | --- |
| `token-mean` | **-0.050** | `0.5 * (-0.40 / 4) + 0.5 * (0.10 / 2)` = **-0.025** |
| `seq-mean-token-sum` | **-0.150** | `0.5 * (-0.40) + 0.5 * (0.10)` = **-0.150** |
| `seq-mean-token-mean` | **-0.025** | `0.5 * (-0.40 / 4) + 0.5 * (0.10 / 2)` = **-0.025** |
| `seq-mean-token-sum-norm` | **-0.060** | `0.5 * (-0.40 / 5) + 0.5 * (0.10 / 5)` = **-0.030** |

This compares reduction/scaling with a fixed token-loss matrix; it does not
assume an optimizer update between microbatches. It matters here because dynamic
microbatching is enabled. The actual recipe remains **`token-mean`**; no YAML
setting is changed by these examples.

## 7. Add reference KL loss

The [actor loss assembly](../../../../verl/workers/actor/dp_actor.py#L493)
adds actor-side KL with coefficient **0.001**.

For `low_var_kl`, the
[implementation](../../../../verl/trainer/ppo/core_algos.py#L1774) computes:

```text
z[t]             = clamp(ref_log_prob[t] - log_prob[t], -20, 20)
KL_token_loss[t] = clamp(exp(z[t]) - z[t] - 1, -10, 10)
KL_loss          = sum(response_mask * KL_token_loss) / sum(response_mask)
```

The microbatch objective is:

```text
actor_loss = PG_loss + 0.001 * KL_loss
```

Entropy coefficient is **0**, so there is no entropy-loss term.

**KL applies to all valid response tokens, including the matching prefix.**
An identical sampled/greedy response has zero ReMax policy loss, but can still
produce KL loss if the actor differs from the reference.

## 8. Backward and optimizer step

Dynamic microbatching is enabled with a **2048-token budget per GPU**. The
[actor update](../../../../verl/workers/actor/dp_actor.py#L452) scales each
microbatch objective by:

```text
microbatch_scale = microbatch_trajectory_count / configured_local_minibatch_size
backward_loss    = microbatch_scale * actor_loss
```

It calls `backward_loss.backward()`, accumulates gradients,
[clips gradient norm to 1.0 and steps the optimizer](../../../../verl/workers/actor/dp_actor.py#L308).

One precision detail: this is **a trajectory-count-weighted sum of microbatch
token means**, not necessarily one exact global token mean when microbatches
contain different token counts.

## Bottom line

This recipe assigns one sampled-minus-greedy reward difference to the response,
then gives **direct ReMax policy-gradient credit only to the suffix starting at
the first token divergence**. The shared prefix remains in the averaging
denominator and still receives KL regularization. The suffix loss can also
affect shared model parameters and earlier context representations; the mask
does not freeze prefix computation.

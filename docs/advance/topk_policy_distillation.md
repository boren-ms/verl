# Top-k Online Policy Distillation

## Overview

Top-k online policy distillation gives the student a richer training target than sampled-token distillation without transporting full-vocabulary logits through Ray.

For every response position, the frozen teacher exports:

- the token IDs of its top-$k$ vocabulary entries;
- the full-vocabulary-normalized log-probability of each retained entry;
- the combined probability mass of every token outside the retained set.

The student evaluates its own distribution on the same token IDs. The remaining student probability is also grouped into one tail bucket. Training minimizes forward KL from the teacher's grouped distribution to the student's grouped distribution.

The PHIMM ASR base enables this mode with $k=64$.

## Data flow

```text
student prompt ──> student rollout ──> sampled response tokens
                                            │
teacher prompt + sampled response tokens ───┘
                 │
                 └─> frozen teacher forward
                       ├─ top-k token IDs            (B, T, K)
                       ├─ top-k log-probabilities    (B, T, K)
                       └─ aggregate tail log-prob    (B, T)
                                      │
                                      ▼
                              Ray/DataProto transfer
                                      │
                                      ▼
                               student forward
                       ├─ gather student top-k probs
                       ├─ compute student tail mass
                       └─ masked grouped forward KL
```

Here $B$ is the micro-batch size, $T$ is the response length, and $K$ is `distill_topk`.

The response tokens and response mask are identical between the student and teacher paths. Their prompt contexts may differ, allowing the teacher to receive privileged task or keyword-biasing context.

## Objective

Let $S_T$ be the teacher's top-$k$ token set at one response position. The retained probabilities are normalized against the complete vocabulary, not renormalized within the top-$k$ set:

$$
p_T(i)=\frac{\exp(z_T(i)/\tau)}{\sum_{v\in V}\exp(z_T(v)/\tau)},
\qquad i\in S_T.
$$

The tail probability preserves omitted mass:

$$
p_T(\mathrm{tail})=1-\sum_{i\in S_T}p_T(i).
$$

The student uses the same teacher-selected token set:

$$
p_S(\mathrm{tail})=1-\sum_{i\in S_T}p_S(i).
$$

The per-token grouped forward KL is:

$$
D_{\mathrm{KL}}^{\mathrm{top}k+\mathrm{tail}}(T\Vert S)
=
\sum_{i\in S_T}p_T(i)\log\frac{p_T(i)}{p_S(i)}
+
p_T(\mathrm{tail})\log\frac{p_T(\mathrm{tail})}{p_S(\mathrm{tail})}.
$$

The final loss applies temperature scaling, the response mask, configured aggregation, and the KL coefficient:

$$
\mathcal{L}
=
\lambda\tau^2\operatorname{Agg}_{m_t}
\left[D_{\mathrm{KL}}^{\mathrm{top}k+\mathrm{tail}}(T\Vert S)\right].
$$

Teacher tensors are detached. Gradients flow only through the student top-$k$ probabilities and student tail mass.

## Configuration

The actor configuration exposes two settings:

```yaml
actor_rollout_ref:
  actor:
    distill_only: true
    use_kl_loss: true
    distill_topk: 64
    distill_temperature: 1.0
    kl_loss_coef: 1.0
```

- `distill_topk: 64` retains 64 teacher entries per response position.
- `distill_temperature` must be positive and is applied to both distributions.
- `distill_topk: 0` disables top-k transport and restores sampled-token KL through `kl_loss_type`.
- Top-k mode requires `use_kl_loss: true`.
- Top-k mode requires fused actor kernels to be disabled because raw logits are needed.

Top-k and sampled-token KL are alternative objectives selected per run. They are not summed: when `distill_topk` is positive, the grouped top-k-plus-tail loss replaces `kl_loss_type`; when it is zero, the original sampled-token estimator is used.

The reusable configuration is `recipe/phimm/config/base/distill_asr.yaml`. The concrete 2607 experiment is `recipe/phimm/config/distill_2607v1/distill_2607v1_bad_mix13k_s200_bs64_lid05_sfl_tm1.yaml`.

## Implementation

### Probability extraction

`compute_topk_log_probs()` in `verl/trainer/ppo/core_algos.py`:

1. selects the teacher's top-$k$ logits, or gathers student logits at supplied teacher indices;
2. computes the exact full-vocabulary log-normalizer in vocabulary chunks;
3. returns full-vocabulary-normalized top-$k$ log-probabilities;
4. computes aggregate tail mass with `log1p(-topk_mass)`.

Chunked normalization avoids creating a second full-vocabulary FP32 logits tensor while retaining exact normalization.

### Teacher export

`DataParallelPPOActor.compute_log_prob()` in `verl/workers/actor/dp_actor.py` optionally returns top-$k$ values, tail mass, and indices. Both padded and remove-padding paths are supported, including dynamic micro-batching and Ulysses sequence-parallel gather/unpadding.

`compute_ref_log_prob()` in `verl/workers/fsdp_workers.py` transports:

- `ref_topk_log_probs`;
- `ref_tail_log_prob`;
- `ref_topk_indices` as `int32`.

This works for a standalone frozen teacher and an adapter-disabled LoRA base teacher.

### Student loss

During `update_policy()`, the actor gathers student logits at `ref_topk_indices`, computes its exact top-$k$ probabilities and tail mass, and calls `topk_distill_kl()`.

In pure distillation mode, this remains the only optimization objective. Reward computation, old-policy scoring, advantages, returns, critic updates, PPO/GRPO/ReMax losses, entropy regularization, and truncated importance sampling remain disabled.

## Transport and memory

Full logits require $O(BTV)$ transport, where $V$ is vocabulary size. Top-k transport requires $O(BTK)$ values and indices plus $O(BT)$ tail values.

For $B=64$, $T=512$, and $K=64$:

- FP32 top-k log-probabilities: about 8 MiB;
- INT32 token IDs: about 8 MiB;
- FP32 tail values: about 0.125 MiB.

This is substantially smaller than transferring complete logits for a vocabulary containing hundreds of thousands of tokens.

## Metrics

Top-k mode reuses the existing actor metrics:

- `actor/loss`;
- `actor/kl_loss`;
- `actor/kl_coef`;
- `actor/grad_norm`.

The KL metric represents the grouped top-$k$-plus-tail forward KL after response masking and loss aggregation.

## Validation coverage

The regression coverage verifies:

- top-$k$ values use exact full-vocabulary normalization;
- chunked normalization matches direct `log_softmax` results;
- supplied teacher indices select the same student support;
- tail probability equals the omitted vocabulary mass;
- grouped forward-KL values match a direct grouped-distribution calculation;
- temperature-squared scaling is applied;
- gradients flow to student logits but not teacher logits;
- invalid top-k, fused-kernel, and temperature configurations are rejected;
- `distill_topk: 0` keeps the previous two-value scoring interface.

Syntax checks, randomized numerical comparisons, gradient checks, and Hydra composition of the top-64 experiment pass. Full local pytest collection is currently blocked by the environment's incompatible torchvision operator registration, unrelated to this implementation.

## Current limitations

- Teacher and student must share compatible vocabulary IDs.
- The implementation currently targets the FSDP actor path.
- Fused actor kernels are unsupported in top-k mode because they do not expose raw logits.
- Tokens outside the teacher top-$k$ set are represented by one tail bucket, so distinctions among individual tail tokens are intentionally discarded.
- This is exact KL for the grouped top-$k$-plus-tail distribution, not exact token-level KL over the complete vocabulary.

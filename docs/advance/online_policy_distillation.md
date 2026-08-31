# Online Policy Distillation for ASR

## Summary

This change adds a pure online policy-distillation path for PHIMM ASR training. The student samples responses from its current policy, while a frozen teacher supplies a top-k distribution under an independently configured prompt context. Training minimizes only the masked teacher KL objective—there is no PPO, GRPO, ReMax, reward-derived advantage, or critic loss.

## Training flow

```text
ASR sample
  ├─ student prompt ──> student rollout ──> sampled response tokens
  └─ teacher prompt + same response tokens ──> frozen teacher log-probabilities

student probabilities + teacher top-k probabilities and tail mass
  └─ masked KL distillation loss ──> student update
```

The response tokens and response mask are shared between the student and teacher paths. Only the prompt context is replaced for teacher scoring.

## Pure-distillation behavior

Set `actor_rollout_ref.actor.distill_only: true` to activate the dedicated path. In this mode, training does not:

- execute a PPO/GRPO/ReMax policy-loss function;
- call training reward functions or reward models;
- compute old-policy log-probabilities;
- compute critic values;
- compute or materialize advantages, returns, token-level scores, or token-level rewards;
- apply entropy regularization or truncated importance sampling.

Shared rollout metrics remain available without requiring RL-only tensors. Validation continues to use the configured validation reward manager.

With top-k distillation enabled, the objective groups the vocabulary outside the teacher's top-k set into one tail bucket:

$$
\mathcal{L}_{\mathrm{distill}}
= \lambda\,\operatorname{Agg}_{m_t}\left[
\sum_{i\in\operatorname{TopK}_T}p_T(i)\log\frac{p_T(i)}{p_S(i)}
+p_T(\mathrm{tail})\log\frac{p_T(\mathrm{tail})}{p_S(\mathrm{tail})}
\right],
$$

where $c_S$ and $c_T$ may differ, $m_t$ is the response mask, and the teacher is frozen. The ASR base retains 64 teacher entries per response token. Set `actor_rollout_ref.actor.distill_topk: 0` to use the sampled-token `k3+` objective instead.

The two supported modes are selected per run:

```yaml
# Top-k-plus-tail forward KL (the ASR default)
actor_rollout_ref:
  actor:
    distill_topk: 64
    distill_temperature: 1.0
```

```yaml
# Previous sampled-token KL
actor_rollout_ref:
  actor:
    distill_topk: 0
    kl_loss_type: k3+
```

Top-k mode replaces sampled-token KL rather than adding both losses together. See [Top-k Online Policy Distillation](topk_policy_distillation.md) for the probability formulation, tensor flow, transport cost, and implementation details.

## Teacher model selection

The reference-policy worker acts as the teacher:

```yaml
actor_rollout_ref:
  ref:
    model:
      path: null
```

- `path: null` with a LoRA student uses the adapter-disabled base model.
- Setting `path` to another checkpoint launches a separate frozen teacher, even when the student uses LoRA.

Student and teacher checkpoints must use compatible token IDs because the teacher scores the exact tokens sampled by the student.

## Per-sample teacher context

The `add_teacher` dataset transform creates `teacher_prompt` for every sample:

```yaml
add_teacher:
  task: asr
  biasing: true
```

Supported options are:

- `task`: any task accepted by `get_task_prompt()`;
- `model_version`: supplied automatically by the dataset pipeline;
- `biasing`: when true and `keywords` is non-empty, append a starred keyword hint after the formatted task prompt.

Example for Qwen 2607:

```text
Transcribe the audio clip into text.<audio>
Pay extra attention to the following phrases/words: *Abate*, *Alberto Sordi*.
```

With no keywords—or with `biasing: false`—the teacher receives only the formatted task prompt. `verl_format_ds()` preserves `teacher_prompt`, and `RLHFDataset` tokenizes it with the same audio as the student prompt.

## Configuration

The reusable base is `recipe/phimm/config/base/distill_asr.yaml`. It derives directly from `ppo_trainer` and defines:

- the PHIMM audio dataset class;
- FSDP2 actor and frozen-reference settings;
- vLLM online rollout settings;
- pure-distillation actor settings;
- top-64 teacher distributions with aggregated vocabulary-tail mass;
- disabled critic, reward KL, filtering, rollout log-probabilities, and TIS;
- validation and checkpoint defaults.

A concrete 2607 experiment is provided at:

```text
recipe/phimm/config/distill_2607v1/distill_2607v1_bad_mix13k_s200_bs64_lid05_sfl_tm1.yaml
```

Its training dataset uses language-detection prompts for both policies, while the teacher additionally receives per-sample keyword biasing context.

## Metrics

The actor emits:

- `actor/loss`;
- `actor/kl_loss`;
- `actor/kl_coef`;
- `actor/grad_norm`.

Policy-gradient metrics remain zero-valued compatibility fields inside the actor update. Reward, advantage, return, and critic metrics are omitted because their tensors are not computed.

## Validation coverage

The change includes tests for:

- external teacher selection for LoRA students;
- teacher-context batch reconstruction without mutating student inputs;
- data metrics without RL-only tensors;
- K3 straight-through value and gradient behavior;
- exact full-vocabulary normalization for top-k probabilities and aggregate tail mass;
- top-k forward-KL values, temperature scaling, and student-only gradients;
- invalid top-k actor configuration combinations;
- task-specific teacher prompt generation;
- starred keyword biasing and empty-keyword fallback;
- preservation of per-sample teacher prompts through PHIMM RL formatting.

## Current scope

Top-k distillation currently supports the FSDP actor path with fused kernels disabled. Teacher and student must share a vocabulary because teacher token IDs index the student logits. The top-k-plus-tail objective preserves total probability mass while avoiding full-vocabulary transport, but it does not preserve distinctions among individual tail tokens.

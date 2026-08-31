# Online Policy Distillation for ASR

## Summary

This change adds a pure online policy-distillation path for PHIMM ASR training. The student samples responses from its current policy, while a frozen teacher scores those same response tokens under an independently configured prompt context. Training minimizes only the masked teacher KL objective—there is no PPO, GRPO, ReMax, reward-derived advantage, or critic loss.

## Training flow

```text
ASR sample
  ├─ student prompt ──> student rollout ──> sampled response tokens
  └─ teacher prompt + same response tokens ──> frozen teacher log-probabilities

student log-probabilities + teacher log-probabilities
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

The distillation objective is:

$$
\mathcal{L}_{\mathrm{distill}}
= \lambda\,\operatorname{Agg}_{m_t}
\left[\widehat{D}_{\mathrm{KL}}\left(\pi_S(\cdot\mid c_S)\,\Vert\,\pi_T(\cdot\mid c_T)\right)\right],
$$

where $c_S$ and $c_T$ may differ, $m_t$ is the response mask, and the teacher is frozen. The default `k3+` estimator uses the low-variance K3 value with the K2 straight-through gradient.

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
- disabled critic, reward KL, filtering, rollout log-probabilities, and TIS;
- validation and checkpoint defaults.

A concrete 2607 experiment is provided at:

```text
recipe/phimm/config/ver_2607v1/distill_2607v1_bad_mix13k_s200_bs64_lid05_sfl_tm1.yaml
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
- task-specific teacher prompt generation;
- starred keyword biasing and empty-keyword fallback;
- preservation of per-sample teacher prompts through PHIMM RL formatting.

## Current scope

This implementation performs sampled-token online distillation. It does not transfer full-vocabulary teacher logits and therefore is not exact forward-KL distillation. Full-distribution distillation would require a memory-efficient top-k teacher-logit transport path.

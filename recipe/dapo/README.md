# Recipe: Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)

> Open-Source Algorithm Implementation & Expriement Running: [Yuxuan Tong](https://tongyx361.github.io/), [Guangming Sheng](https://hk.linkedin.com/in/guangming-sheng-b50640211)

> [!IMPORTANT]
>
> **🔥 News!!!**
>
> - [2025/04] We reproduced the results of two versions of DAPO ([Full](./run_dapo_qwen2.5_32b.sh) & [w/o Dynamic Sampling](./run_dapo_wo_ds_qwen2.5_32b.sh)), achieving 52% and 50% on AIME 2024 respectively, based on [the latest codebase on `recipe/dapo`](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo). Please check the details in [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n).
> - [2025/03] We published the training record of [an early version of DAPO (w/o Token-level PG Loss & Dynamic Sampling)](./run_dapo_early_qwen2.5_32b.sh), achieving 44% on AIME 2024, in [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n).

🏠 [Homepage](https://dapo-sia.github.io/) | 📝 [Paper@arXiv](https://arxiv.org/abs/2503.14476) | 🤗 [Datasets&Models@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 [Code@GitHub](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo) | 🐱 [Repo@GitHub](https://github.com/BytedTsinghua-SIA/DAPO)

> We propose the **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization (DAPO) algorithm. By making our work publicly available, we provide the broader research community and society with practical access to scalable reinforcement learning, enabling all to benefit from these advancements. Our system is based on the awesome [verl](https://github.com/volcengine/verl) framework. Thanks for their great work! Applying DAPO training to Qwen2.5-32B base model proves to outperform the previous state-of-the-art DeepSeek-R1-Zero-Qwen-32B on AIME 2024, achieving **50%** accuracy with **50%** less training steps.
>
> ![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)

## Quickstart

1. Prepare the datasets **on the Ray cluster**:

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. Submit the job to the Ray cluster **from any machine**:

```bash
cd verl # Repo root
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/dapo/runtime_env.yaml" # This sets environment variables for the Ray cluster
bash recipe/dapo/run_dapo_qwen2.5_32b.sh # or other scripts
```

## Reproduction Runs

| Setup                                        | AIME 2024 Acc. | Hardware  | Image                                                                | Commit                                                                                       | Environment Variables                                                                                                             | Training Script                                                                                                                                             | Training Record                                                                           |
| -------------------------------------------- | -------------- | --------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| DAPO                                         | 52%            | 16x8xH800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_qwen2.5_32b.sh)             | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO w/o Dynamic Sampling                    | 50%            | 16x8xH800 | `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0` | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_wo_ds_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_wo_ds_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |
| DAPO w/o Token-level Loss & Dynamic Sampling | 44%            | 16x8xH20  | `hiyouga/verl:ngc-th2.5.1-cu120-vllm0.7.4-hotfix`                    | [`4f80e4`](https://github.com/volcengine/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490) | [runtime_env.yaml](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/runtime_env.yaml) | [run_dapo_early_qwen2.5_32b.sh](https://github.com/volcengine/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/run_dapo_early_qwen2.5_32b.sh) | [W&B](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl/workspace?nw=wmb4qxfht0n) |

> [!IMPORTANT]
>
> **📢 Call for Contribution!**
>
> Welcome to submit your reproduction runs and setups!

## Configuration

### Bad case replay for ReMax

Enable `algorithm.bad_case_replay` for a ReMax job to revisit inputs whose
greedy baseline has low metric scores. This works in the shared PPO trainer
and DAPO trainer. It is disabled by default and requires
`algorithm.adv_estimator: remax`. All other supported settings follow the
existing trainer path.

```yaml
algorithm:
  adv_estimator: remax
  bad_case_replay:
    enable: true
    capacity: 400
    replay_size: 16
    sampling: random # random | worst
    selection:
      weights: {word: 1.0}
    seed: 42
```

Use these overrides with an existing ASR ReMax config that returns a `word`
metric:

```bash
python -m recipe.dapo.main_dapo \
  --config-path ../phimm/config/v2611_dev/dev \
  --config-name remax_dev_base0_v1h2 \
  algorithm.bad_case_replay.enable=true \
  algorithm.bad_case_replay.replay_size=16 \
  algorithm.bad_case_replay.sampling=random
```

The replay score is `sum(weight * raw_metric) / sum(weight)`. These weights are
independent of training reward weights. Metrics are used exactly as returned;
negative error metrics are neither shifted nor sign-reversed. Lower scores are
selected first. Weights must be finite and nonnegative, with a positive total.
Missing configured metrics raise an error; rows with NaN or infinite scores
are skipped and counted. In particular, the `asr_vote` scorer can return zero
for every greedy total `score`; select meaningful raw components such as
`word` and `char` instead. A scorer must return every selected component.
The default uses only `word`. To combine word and character accuracy, use a
scorer that returns both metrics and explicitly add the character weight:

```bash
algorithm.bad_case_replay.selection.weights.word=0.7 \
  +algorithm.bad_case_replay.selection.weights.char=0.3
```

Let `B` be the nominal `data.gen_batch_size` (or `data.train_batch_size` when
unset), before expanding each prompt to `rollout.n` responses, and `D` the
trainer's data-parallel size. Set the integer `replay_size=K` to reserve replay
slots and fix the DataLoader batch size to `F=B-K` fresh inputs. An enabled
configuration requires `0 <= K < B`, `K % D == 0`, and `B % D == 0`.
At each generation, draw `k=floor(min(K, buffer_size)/D)*D` replay inputs and
immediately remove them from the buffer. Combine them with the fixed F fresh
inputs, giving `N=F+k`. Unselected entries stay in the buffer. Missing replay
inputs leave the batch smaller; they do not increase fresh sampling.
With B=128, K=16, and D=16, the first generation has 112 fresh inputs. All
112 inputs are eligible for the buffer when their scores are finite. Once at
least 16 replay inputs are available, a generation has 112 fresh plus 16 replay
inputs. The loader retains `drop_last=True` and does not generate buffer-only
batches after fresh data is exhausted.

The combined batch contains the same padded raw inputs as the DataLoader
batch, including audio and sample metadata. Replay construction happens before
generation; sampled rollout, greedy baseline decoding, reward processing,
batch balancing, and the actor update then use the original trainer path.
Fresh and replay inputs share the existing ReMax/PPO loss and normalization.
The combined batch retains the trainer's existing row-count divisibility
requirements; replay does not add copied training rows to satisfy them.

`random` samples uniformly without replacement using the configured seed;
`worst` takes the lowest saved scores, preserving order for ties. All inputs
run through new sampled rollouts, greedy baseline generation, reward
computation, and training. After each greedy reward computation, all original
inputs with finite scores are eligible for insertion. Merge these inputs with
the remaining buffer by stable input identity, updating observed entries to
their latest valid scores, then retain the globally lowest-scored `capacity`
entries. There is no per-generation preselection quota. Unobserved entries keep
their last known scores. Expanded rollout rows do not count as additional
inputs. A sampled replay input reenters only if its new score survives global
capacity eviction; being sampled does not automatically reinsert it.

The buffer holds independent CPU copies of original inputs and metadata. It
persists across batches and epochs in one process, starts empty whenever the
process starts, and is not saved or restored with model or DataLoader
checkpoints. DAPO updates the buffer for every generation batch before group
filtering can continue to the next generation; `data.train_batch_size` remains
the filtering target. Automatic total steps use the fresh DataLoader length;
an explicit `trainer.total_training_steps` takes precedence.

Replay logs count original prompts and include requested/actual replay,
fresh/actual generation counts, candidates, invalid scores, duplicates,
capacity evictions, buffer size, and score distributions. Reentry metrics are:

| Metric | Meaning |
| --- | --- |
| `replay/reinsert_candidates` | Sampled replay inputs with finite current scores, eligible for global buffer selection. |
| `replay/reinserted` | Those inputs still retained after ID deduplication and capacity eviction. |
| `replay/reinsert_rate` | Retained reentries divided by actual replay inputs; zero when no inputs were replayed. |

When DAPO accumulates several generation batches for one optimizer step,
counts accumulate and rates use their accumulated numerators and denominators.

### ReMax advantage masking

Select the masking mode directly with:

```yaml
algorithm:
  adv_estimator: remax
  remax_advantage_mask: first_diff
```

- `null` (default): disable advantage masking.
- `edit_boundary`: preserve the existing edit alignment. Select unmatched
  sampled tokens and deletion boundaries, including the last sampled token when a
  deletion occurs at the end.
- `first_diff`: compare sampled and greedy valid tokens positionally. Select
  every sampled token from the first mismatch onward, even if later tokens match.
  If the sampled response extends a matching greedy prefix, select its extra suffix.
  Identical responses, empty sampled responses, and shorter sampled responses that
  match the greedy prefix select no tokens.

For greedy `[1, 2, 9]` and sampled `[1, 3, 9]`, the masks are `[0, 1, 0]`
(`edit_boundary`) and `[0, 1, 1]` (`first_diff`). Padding is never selected.
Existing AM recipes use `remax_advantage_mask: edit_boundary` to preserve their
behavior. Replace previous boolean `true` with `edit_boundary`, and `false` with
`null`; there is no separate `remax_advantage_mask_mode` setting.
Both modes mask final policy advantages only; actor KL, entropy, returns, and loss
denominators remain unchanged. Masking requires the DAPO path and token-level
vanilla policy loss, and cannot be combined with `use_kl_in_reward`.

### Non-finite logprob filtering metrics

The [2609 ASR base configuration](../phimm/config/v2609_asr/base.yaml) enables
`filter_nonfinite_log_probs` for the actor, reference policy, and trainer by
default. Recipes inheriting this base allow non-finite worker logprobs to reach
the trainer for filtering.

With `trainer.filter_nonfinite_log_probs: true`, the trainer reports these
per-step metrics through the configured logging backends, including W&B:

| Metric | Meaning |
| --- | --- |
| `nonfinite_prob/old` | Responses with non-finite old-policy logprobs. |
| `nonfinite_prob/ref` | Responses with non-finite reference-policy logprobs. |
| `nonfinite_prob/rollout` | Responses with non-finite rollout logprobs. |
| `nonfinite_prob/invalid` | Unique responses invalid in any source. |
| `nonfinite_prob/removed` | Total removed responses, including DP-alignment trimming. |

Each source counts whole response trajectories, not tokens or unique prompts,
and only checks tokens selected by `response_mask`. Source counts overlap when
a response is invalid in multiple sources, so their sum can exceed the unique
invalid count. An absent source reports zero. Alignment-only removals do not
increase source counts. If rollout logprobs are reused as old-policy logprobs,
the old and rollout counts match.

### Separated Clip Epsilons (-> Clip-Higher)

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

`clip_ratio_low` and `clip_ratio_high` specify the $\varepsilon_{\text {low }}$ and $\varepsilon_{\text {high }}$ in the DAPO objective.

Core relevant code:

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

### Dynamic Sampling (with Group Filtering)

An example configuration:

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # Non-positive values mean no upper limit
```

Setting `filter_groups.enable` to `True` will filter out groups whose outputs' `metric` are all the same, e.g., for `acc`, groups whose outputs' accuracies are all 1 or 0.

The trainer will repeat sampling with `gen_batch_size` until there are enough qualified groups for `train_batch_size` or reaching the upper limit specified by `max_num_gen_batches`.

Core relevant code:

```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=} < {max_num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # Align the batch
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

### Flexible Loss Aggregation Mode (-> Token-level Loss)

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "token-mean" # / "seq-mean-token-sum" / "seq-mean-token-mean"
    # NOTE: "token-mean" is the default behavior
```

Setting `loss_agg_mode` to `token-mean` will mean the (policy gradient) loss across all the tokens in all the sequences in a mini-batch.

Core relevant code:

```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
else:
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
```

### Overlong Reward Shaping

An example configuration:

```yaml
data:
  max_response_length: 20480 # 16384 + 4096
reward_model:
  overlong_buffer:
    enable: True
    len: 4096
    penalty_factor: 1.0
```

Setting `overlong_buffer.enable` to `True` will penalize the outputs whose lengths are overlong but still within the hard context limit.

Specifically, the penalty increases linearly from `0` to `overlong_buffer.penalty_factor` when the length of the output exceeds the `max_response_length` by `0` to `overlong_buffer.len` tokens.

Core relevant code:

```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
```

## FAQ

### Where is the "Overlong Filtering" in the paper?

Most experiments in the paper, including the best-performant one, are run without Overlong Filtering because it's somehow overlapping with Overlong Reward Shaping in terms of properly learning from the longest outputs. So we don't implement it here.

### What's the difference between [the `recipe/dapo` directory in the `main` branch](https://github.com/volcengine/verl/tree/main/recipe/dapo) and the [`recipe/dapo` branch](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo)?

[The `recipe/dapo` branch](https://github.com/volcengine/verl/tree/recipe/dapo/recipe/dapo) is for **as-is reproduction** and thus won't be updated with new features.

[The `recipe/dapo` directory in the `main` branch](https://github.com/volcengine/verl/tree/main/recipe/dapo) works as an example of how to extend the latest `verl` to implement an algorithm recipe, which will be maintained with new features.

### Why can't I produce similar results after modifications?

RL infrastructures nowadays still have inherent unrobustness, on which we are still working hard to improve.

We strongly recommend to only modify one thing at a time.

We also list some known problems here:

1. Enabling CUDA graph (`enforce_eager=False`) might cause model performance degradation, whose cause is still under investigation.

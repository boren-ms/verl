# Recipe
The examples under `recipes/` are representative extensions to verl for specific end-to-end RL training recipes.
The help the community reproduce experiments, verl team provides a snapshot of the codebase when each recipe is initially PR'ed to verl main. You can find them via [github branches](https://github.com/volcengine/verl/branches/all?query=recipe)

## ASR datasource sampling

Configure training budgets by **datasource name**, not by dataset entry:

```yaml
data:
  use_interleave: false
  data_source:
    earnings_fy27: {num_sample: 10746}
    openml: {num_sample: 10746}
```

The [double-earnings sampling recipe](phimm/config/v2609_asr/remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp2.yaml)
uses 21,492 samples for the combined `earnings_fy27` datasource (including TTS)
and keeps OpenML at 10,746 samples; all other settings match the `_smp` recipe.
The [102.4k/25.6k sampling recipe](phimm/config/v2609_asr/remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp3.yaml)
uses 102,400 earnings samples and 25,600 OpenML samples (128,000 total per pass),
with `shuffle: true` inside each datasource and `data.shuffle: false`.
Rows are shuffled within each source while the datasource blocks remain in order;
all other settings are unchanged from `_smp2`.
The [2609v1 sampling recipe](phimm/config/v2609_asr/remax_2609v1_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp.yaml)
instead keeps both budgets at 10,746 and changes only the model to the v1 shadow
checkpoint at step 54,000, which requires an HF export before training.
The [LibriSpeech rare-keyword variant](phimm/config/v2609_asr/remax_2609v0_earning_ml_ls_verb_hint_s1k_bs128_n8_r256_g32_flr_smp.yaml)
adds rare-keyword training and test-clean/test-other validation using verbatim
English language-hint prompts. It sets all three datasource budgets to 10,746,
sampling `ls_rare_verb` from its verified 256,213-sample manifest,
with a shared 2609 reward including keyword accuracy.
Other training settings remain those of `_smp`, not the 2607 donor.
The shared [2609 base](phimm/config/v2609_asr/base.yaml) routes `ls_clean` and
`ls_other` validation to `openasr_en`.

When `data.data_source` is non-null, before combining training data, `RLHFDataset` calls
`group_datasets(datasets, datasource_settings)` after
loading/preprocessing (or reading a cache).
The helper reads the first sample's `data_source` from each dataset, assuming all
its rows share that label. If the field is absent, its grouping name is `"default"`,
and any `data.data_source.default` settings apply to that group.
Datasets with the same label are concatenated **before applying one budget**.
For example, two earnings datasets of 3,000 and 2,000 rows with `num_sample: 4000`
produce 4,000 rows total, not 4,000 from each dataset.
The function returns a **list of datasets, one per retained datasource**, without
combining different sources. The caller then concatenates or interleaves this
list according to `data.use_interleave`.

- `num_epoch`: non-negative passes, including fractions; selects
  `floor(combined_source_size * num_epoch)` rows.
- `num_sample`: an exact non-negative integer count, repeating the combined
  source when necessary.
- `shuffle`: a boolean, defaulting to `false`. Set `true` to call `Dataset.shuffle()`
  on the sampled dataset **within that datasource**. Shuffling happens after
  repetition/random selection and does not change the order of datasource groups.
  No per-source seed or shuffle-options mapping is supported.

For example, `earnings_fy27: {num_epoch: 2, shuffle: true}` repeats the
combined earnings source twice, then shuffles those rows.

Set only one non-null control per source. Every loaded datasource must have an
entry when grouping is enabled; use `source_name: {}` for one full pass.
If `data.data_source` is absent
or null, grouping and sampling are skipped and the original datasets are
concatenated/interleaved unchanged. An explicit empty mapping (`{}`) rejects
non-empty training datasets because their datasource entries are missing.
Zero removes a source. Configured and non-empty loaded datasource names must match
exactly; a mismatch raises an error listing both configured names without datasets
and loaded sources without configuration. Settings are consumed
when sampling each present source, without an up-front per-source schema check.
Source labels are not prevalidated; an entirely empty result still raises an error.
Empty datasets are skipped with a warning because their source cannot be inferred.
Groups follow only the key order in `data.data_source`, not alphabetical order.
Unlisted loaded sources raise an error rather than being appended or silently dropped.
Datasets within a group retain their
order, including repeated passes. For example, inputs `A1, B1, A2` with two epochs
of A return two datasets: `[A1 + A2 + A1 + A2, B1]`. Concatenating them keeps each
datasource in a contiguous block. Partial passes append a random subset of the
combined source without replacement, using Python's global `random.sample`, even
when the final shuffle is disabled. No fixed seed is applied here, so rebuilding
the dataset can produce a different selection/order. Whole passes retain their
order unless shuffled.
These controls affect training only, leaving validation/generation unchanged.
The dataset loader and its full processed caches are unchanged.

Use `data.use_interleave: false` to concatenate the weighted datasource datasets.
With `true`, interleaving operates on those same weighted groups. Its probabilities
must correspond to the returned datasource groups in configured-key order, not
the original dataset entries. Interleaving's stopping strategy can truncate or
repeat groups, so the final counts need not equal their pre-interleave budgets.
A single retained datasource is used without interleaving.
`data.shuffle` still controls the training dataloader's ordering.
These counts apply **per combined dataset pass**, not to the entire training run:
`trainer.total_epochs` repeats the combined dataset, and step limits or
incomplete dropped batches can stop consumption early.

The [budgeted earnings/OpenML recipe](phimm/config/v2609_asr/remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp.yaml)
uses the settings above: both FY27 batches and Polyus TTS share `earnings_fy27`
and its 10,746-sample budget, matching the OpenML budget. This label is defined in the shared
[earnings datasource YAML](phimm/config/data/train_data/earnings_tts_verb_langhint.yaml)
and used by all recipes that import it.
With `use_interleave: false`, the weighted dataset is built in this order:
`earnings_fy27`, then `openml`. The recipe uses `shuffle: true` to shuffle
training rows, matching the base `flr` recipe. Set `data.shuffle: false` if serial
datasource consumption is desired instead.

## OpenML fixed learning rate

The [OpenML-only fixed-LR recipe](phimm/config/v2609_asr/remax_2609v0_ml_hint_s200_bs128_n8_flr.yaml)
applies a constant `5e-6` learning rate with no warmup to the ML-hint
200-step recipe. Datasets, rewards, batch size 128, eight rollouts, rank 256,
16 GPUs, and 30 epochs are unchanged; redundant `r256` and `g16` tags are omitted.

## Validation batch sizes

The 2609 training and evaluation bases use `data.val_batch_size: 256`.
Each validation dataset can override that default in its own YAML:

```yaml
- dataset_name: jsonl
  val_batch_size: -1
  jsonl_paths: ...
  # Existing preprocessing and parent_audio_path metadata...
```

An absent or null dataset override inherits the global setting. A positive
integer selects that dataset's batch size; `-1` loads the entire dataset in one
batch. Other override values are rejected. The global null/zero fallback to
`data.gen_batch_size` is unchanged.

Nested `data.val_data` lists are flattened in order, with one validation loader
per dataset entry. Final partial batches are retained. The trainer removes the
dataset-level batch option before constructing each validation dataset.
Training batch sizes and the ASR dataset loader's option forwarding are unchanged.

Only the 2609-referenced validation datasets preserving `parent_audio_path` have
explicit `val_batch_size: -1`; short-utterance datasets inherit 256.
`long_audio_grouped` scores within each batch, so all segments of a parent must
stay together. Finite batches for these datasets require parent-aware batching,
which is not implemented here. Full batches can still be large for long-audio
datasets; this change bounds short-utterance batches, not every validation batch.

# Awesome work using verl

- [Logic-RL](https://github.com/Unakar/Logic-RL): a reproduction of DeepSeek R1 Zero on 2K Tiny Logic Puzzle Dataset. ![GitHub Repo stars](https://img.shields.io/github/stars/Unakar/Logic-RL)
- [Seed-Coder](https://github.com/ByteDance-Seed/Seed-Coder): RL training of Seed-Coder boosts performance on competitive programming ![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/Seed-Coder)
- [all-hands/openhands-lm-32b-v0.1](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model): A strong, open coding agent model, trained with [multi-turn fine-tuning](https://github.com/volcengine/verl/pull/195)
- [s3](https://github.com/pat-jj/s3) **Efficient Yet Effective** Search Agent Training via RL ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/s3)
- [Rec-R1](https://arxiv.org/pdf/2503.24289): Bridging Generative Large Language Models and Recommendation Systems via Reinforcement Learning
- [Explore RL Data Scaling](https://arxiv.org/abs/2503.22230): Exploring Data Scaling Trends and Effects in Reinforcement Learning from Human Feedback
- [FIRE](https://arxiv.org/abs/2410.21236): Flaming-hot initiation with regular execution sampling for large language models
- [DQO](https://arxiv.org/abs/2410.09302): Enhancing multi-Step reasoning abilities of language models through direct Q-function optimization
- [ProRL](https://arxiv.org/abs/2505.24864): Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
- [cognition-engineering](https://github.com/gair-nlp/cognition-engineering): Test time scaling drives cognition engineering. ![GitHub Repo stars](https://img.shields.io/github/stars/gair-nlp/cognition-engineering)
- [Trust Region Preference Approximation](https://github.com/XueruiSu/Trust-Region-Preference-Approximation): A simple and stable **reinforcement learning algorithm** for LLM reasoning. ![GitHub Repo stars](https://img.shields.io/github/stars/XueruiSu/Trust-Region-Preference-Approximation)
- [AdaRFT](https://github.com/uscnlp-lime/verl): Efficient Reinforcement Finetuning via **Adaptive Curriculum Learning** ![GitHub Repo stars](https://img.shields.io/github/stars/uscnlp-lime/verl)
- [critic-rl](https://github.com/HKUNLP/critic-rl): LLM critics for code generation ![GitHub Repo stars](https://img.shields.io/github/stars/HKUNLP/critic-rl)
- [self-rewarding-reasoning-LLM](https://arxiv.org/pdf/2502.19613): self-rewarding and correction with **generative reward models** ![GitHub Repo stars](https://img.shields.io/github/stars/RLHFlow/Self-rewarding-reasoning-LLM)
- [DeepEnlighten](https://github.com/DolbyUUU/DeepEnlighten): Reproduce R1 with **social reasoning** tasks and analyze key findings ![GitHub Repo stars](https://img.shields.io/github/stars/DolbyUUU/DeepEnlighten)
- [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing **3D Spatial Reasoning** in **VLMs** for the **Metaverse** ![GitHub Repo stars](https://img.shields.io/github/stars/PzySeere/MetaSpatial)
- [PURE](https://github.com/CJReinforce/PURE): **Credit assignment** is the key to successful reinforcement fine-tuning using **process reward model** ![GitHub Repo stars](https://img.shields.io/github/stars/CJReinforce/PURE)
- [cognitive-behaviors](https://github.com/kanishkg/cognitive-behaviors): Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs ![GitHub Repo stars](https://img.shields.io/github/stars/kanishkg/cognitive-behaviors)
- [deepscaler](https://github.com/agentica-project/rllm/tree/deepscaler): iterative context scaling with GRPO ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/deepscaler)
- [DAPO](https://dapo-sia.github.io/): the fully open source SOTA RL algorithm that beats DeepSeek-R1-zero-32B ![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)
- [NoisyRollout](https://github.com/NUS-TRAIL/NoisyRollout): Reinforcing Visual Reasoning with Data Augmentation ![GitHub Repo stars](https://img.shields.io/github/stars/NUS-TRAIL/NoisyRollout)

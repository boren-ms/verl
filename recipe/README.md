# Recipe
The examples under `recipes/` are representative extensions to verl for specific end-to-end RL training recipes.
The help the community reproduce experiments, verl team provides a snapshot of the codebase when each recipe is initially PR'ed to verl main. You can find them via [github branches](https://github.com/volcengine/verl/branches/all?query=recipe)

## ASR datasource sampling

Configure training budgets by **datasource name**, not by dataset entry:

```yaml
data:
  use_interleave: false
  data_source:
    earnings_fy27: {num_epoch: 2}
    openml: {num_sample: 20000}
```

Before combining training data, `RLHFDataset` calls
`group_weighted_datasource(datasets, datasource_settings)` after
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

Set only one non-null control per source. Unlisted sources and sources with
neither control use the default of one full pass. If `data.data_source` is absent,
null, or empty, all sources are still grouped and retain all rows.
Zero removes a source. Configured names absent from the loaded datasets are logged
as warnings and ignored; their settings are not prechecked. Settings are consumed
when sampling each present source, without an up-front per-source schema check.
Source labels are not prevalidated; an entirely empty result still raises an error.
Empty datasets are skipped with a warning because their source cannot be inferred.
Groups follow their first appearance in `data.train_data`, not alphabetical order
or the order of keys in `data.data_source`. Datasets within a group retain their
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
must correspond to the returned datasource groups in first-appearance order, not
the original dataset entries. Interleaving's stopping strategy can truncate or
repeat groups, so the final counts need not equal their pre-interleave budgets.
A single retained datasource is used without interleaving.
`data.shuffle` still controls the training dataloader's ordering.
These counts apply **per combined dataset pass**, not to the entire training run:
`trainer.total_epochs` repeats the combined dataset, and step limits or
incomplete dropped batches can stop consumption early.

The [budgeted earnings/OpenML recipe](phimm/config/v2609_asr/remax_2609v0_earning_ml_verb_hint_s1k_bs128_n8_r256_g32_flr_smp.yaml)
uses the settings above: both FY27 batches and Polyus TTS share `earnings_fy27`
and its two-epoch budget. This label is defined in the shared
[earnings datasource YAML](phimm/config/data/train_data/earnings_tts_verb_langhint.yaml)
and used by all recipes that import it.
With `use_interleave: false`, the weighted dataset is built in this order:
`earnings_fy27`, then `openml`. The recipe uses `shuffle: true` to shuffle
training rows, matching the base `flr` recipe. Set `data.shuffle: false` if serial
datasource consumption is desired instead.

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

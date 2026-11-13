# Base3a Qwen ASR on main's agent loop trainer

This package imports only the model, audio/data, scoring, and config dependency
closure of `boren/rl_2611_dev` at
`dffabc00d8261dd9abf3087958259d5c8944f5a2`. It keeps the original
`recipe.phimm` imports while leaving main's `recipe` submodule intact.
The legacy DAPO trainer and synchronous vLLM rollout are not imported.

Install into the existing GPU environment without replacing its dependencies:

```bash
uv pip install --system --no-deps -e . -e plugins/phimm -e plugins/qwen35_audio
uv pip install --system opencc-python-reimplemented==0.1.7 wetext==0.2.1
python -m pytest -q plugins/phimm/tests/test_adapter.py
python -m recipe.phimm.preflight trainer.experiment_name=<unique-name>
```

The two pinned text-normalizer dependencies match the source requirements.
Preflight exercises the Chinese, Vietnamese, and French scorers before model
initialization, including dependencies loaded lazily by the Chinese scorer.

`ASRAgentLoop` uses main's LLM client and TransferQueue worker. Each training
prompt gets one shared greedy baseline before its eight sampled trajectories.
The unchanged source scorer computes the delta/absolute vote reward. The
trainer passes the baseline score to the standard ReMax advantage estimator.
Validation groups and sorts segments before scoring full parent references;
corpus error rates use summed error/reference counts.

The adapter retains batch128, LoRA320/640, LR6e-7, two epochs, six-source
mix, validation20/save10, and auto resume. Runtime checks require 55,664 rows
and 868 updates. Main's decoupled token importance correction has cap5 without
batch normalization. Its on-policy loss and gradient are tested against the
source objective; end-to-end quality still requires actual experiment results.

Sampling preserves the source's effective no-repeat behavior: it passes
`ngram_size=15` and `ngram_window=512` to the registered DeepSeek processor.
That processor reads `window_size`, so the effective window remains its
100-token default. Correcting the source spelling here would change the
experiment. Source validation retains the training `extra_args`, including
when its validation no-repeat size is zero or absent. DAPO's greedy ReMax
baseline uses that same validation path and its 512-token response limit.
The adapter preserves these effective settings for all three phases.
It also preserves the source LLM's seed0 on every replica and raw rollout
log probabilities, rather than inheriting main's seed42 and processed scores.

For an already synced two-node Ray pool:

```bash
python plugins/phimm/submit.py --submission-id <unique-id> \
  trainer.experiment_name=<unique-name>
```

Initial validation can be checked separately with `trainer.val_only=true`
and a separate experiment name. It does not replace the training run.
After the final checkpoint is exported and verified, full evaluation uses:

```bash
python plugins/phimm/submit.py --evaluate --submission-id <unique-eval-id> \
  trainer.experiment_name=<unique-eval-name> \
  actor_rollout_ref.model.path=<verified-merged-HF-path>
```

The evaluation manifest retains the reference run's 35 datasets, including
`eslo_telephony_fr` and `accueil_ubs_fr` from its producer revision
`965a025ac220c91305aa762bdb176697b55e0fb5`. The current development branch's
generic manifest omits these two reference datasets.

The recipe selects the installed NVIDIA CUDA runtime matching PyTorch's CUDA
major version. The source branch's explicit-runtime helper is reused in the
Qwen general plugin because TileLang can load an incomplete `libcudart_stub.so`
that vLLM 0.24 would otherwise select for its sleep allocator.
The HF package initializes optional FlashInfer IPC before importing the model,
because spawned workers unpickle its config before vLLM loads general plugins.
`plugins/qwen35_audio/tests/test_cuda_import.py` exercises that spawn order and
checks both CUDA consumers against the installed CUDA major version.

Azure model downloads are cached atomically per node. Every FSDP rank uploads
its own checkpoint files before the driver publishes the latest-step marker.
The callback also uploads dataloader state; auto resume prefers the committed
remote step. Validation JSONL and metric JSON files are uploaded to the unique
experiment's `val_data_gen` directory.

Required quality gate: at every compared dataset, the reproduced error rate
may regress by at most **0.1 percentage point**. Missing datasets or mismatched
reference denominators do not count as a pass. Reference scores must use the
same normalization as reproduced outputs.

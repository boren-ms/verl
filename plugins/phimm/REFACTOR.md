# Base3a integration refactor

Starting point: `6a7fd562b66a35e213521e13e583c5d7d83a2a8c` on a clean,
separate worktree. Design authority: Main at its parent
`662507f9582e16d09c12484387f27727b5d0f947`, as implemented in the current
trainer, worker, dataset factory and Hydra component configs.

## Architecture

- Main's `ppo_trainer` composes its actor, reference, rollout, model, data,
  optimizer and rollout-correction configs. The recipe overrides those native
  sections in YAML; it no longer composes twice, manipulates GlobalHydra state,
  resolves a legacy config early, or copies only recognized fields.
- The recipe uses Main's package-based dataset import. `ASRDataset` returns raw
  prompts and segment rows directly, preserving nested manifests, null-feature
  alignment, source budgets, deterministic mixing and validation grouping.
  The legacy dataset's audio retry/tokenization/truncation path was never used
  by this adapter and has been removed. Audio loading remains in the agent loop.
- Main's `@register("asr")` selects the loop without an auxiliary registry YAML.
  `ASRWorker` adds one shared greedy baseline before Main starts sampled sessions,
  and the audio-specific processor-input bridge. Main owns session fan-out,
  queue writes, position IDs, scoring handoff and worker lifecycle.
- `ASRTrainerSync` retains reward-component metrics, corpus validation metrics,
  required row/update checks, validation artifact paths and Azure resume.
  `ASRCheckpointCallback` publishes dataloader state and then the committed-step
  marker. Existing rank-shard publication and barriers remain unchanged.
- The integration's small core extension points remain: a subclassable
  TransferQueue worker, a trainer class selection independent of scheduling
  mode, and loading ReMax baseline tensors from TransferQueue. The core also
  retains Azure IO, ragged-audio collation and audio processor position handling.
  Replacing these with Ray private metadata, global registry overrides or a
  duplicated training driver would weaken the extension boundary.
- Shared audio cache, ZIP byte readers and chunk byte readers stay in core:
  both the recipe and the existing audio-subset skill use them. This avoids
  introducing a dependency from core or the standalone skill to the optional
  recipe package. Generic Azure checkpoint IO and model-download behavior
  remain unchanged.

## Removed code and configuration

Removed the Python config translator, Base1/Base1a/Base2a inheritance ladder,
auxiliary agent registry YAML, unused earnings-AA validation manifest and legacy
RL dataset. The Base3a experiment now declares its manifest inventory directly.

Reduced the data pipeline to JSONL loading, path mapping, field renaming,
entity-tag cleaning, task/prompt formatting, ranked-prefix limits, constant
fields and background audio prefetch. Unsupported transforms raise an error.
Removed unused data-generation, preference-simulation, biasing, audio
augmentation, VAD and silence-trimming paths and resources. Chunk sample readers
remain; unused chunk dataset builders were removed. Removed unused storage,
export and duplicate remote-cache helpers from the recipe's shared utilities.

Removed the model's original checkpoint conversion and standalone HF verification
scripts. Retained `convert_verl_to_pt.py` because the documented final-checkpoint
export and LoRA workflow calls it. Retained `cache_eval_audio.py` because the
managed ASR workflow uses it for optional prefetch. Model frontend implementations,
full evaluation manifests, imported skills and historical reports are preserved.

## Training controls and requested behavior change

| Control | Value |
| --- | --- |
| Training batch / samples per prompt | 128 / 8 |
| LoRA rank / alpha | 320 / 640 |
| Learning rate / schedule | 6e-7 / constant, no warmup |
| Epochs / rows / updates | 2 / 55,664 / 868 |
| Source budgets | noisy_ls 20,424; earn_fy27 9,568; quartr 3,125 + 6,392 + 6,155; aishell 10,000 |
| Validation / checkpoint cadence | every 20 / every 10 updates |
| Resume | auto, committed Azure marker |
| Rollout seed / log probabilities | 0 per replica / raw |
| Sampled / greedy / validation response limit | 768 / 512 / 512 |
| Token importance correction | cap 5, no batch normalization |
| No-repeat request | ngram_size 15, window_size 512; configured window now takes effect |

The source data mix still uses its effective seed 42 when `data.seed` is null
or zero. This is separate from the rollout seed 0. No seed behavior is changed.

The unseeded greedy vote scorer intentionally returns a zero score with component
measurements. Those measurements feed sampled delta/vote rewards; Main subtracts
only the baseline score, avoiding a second subtraction of greedy error measures.
Validation still groups by source/parent, orders segments by start time and
computes corpus rates from summed errors and reference units.

Inference still uses text placeholders plus waveforms; training keeps expanded
HF prompt IDs and the exact generated IDs/log probabilities. Ragged audio features
and masks retain time padding and contiguous feature tensors.

The original effective training config was captured before editing and compared
with the native composition before the requested no-repeat fix. Training values
matched at that stage; the subsequent change deliberately makes the configured
512-token window effective instead of the source default of 100. Removed fields are obsolete
translation diagnostics, unused data-loader/examination declarations, legacy
reward-manager declarations and the deprecated optimizer `warmup_style` alias
(the native `lr_scheduler_type` remains constant). Static runtime environment
settings moved from the entry point into YAML. Dataset/agent loading now uses
package imports and Main's registry. Stale source mappings for excluded training
data were removed.

Native overrides now propagate through model names, tracing, prompt/token budgets,
batches, LoRA alpha, seeds and artifact paths. Evaluation defaults explicitly
require a model path and experiment name at launch; the old translation could
fall back to inherited model/default-name values when Hydra merged `???`.
Evaluation retains all 35 manifest entries, including the two French OpenExt sets.

## Local validation

Use the existing environment; no environment installation or remote job is
required for the local tests. From the repository root:

```bash
export PYTHONPATH=plugins/phimm/src:plugins/qwen35_audio/src:.
/home/boren/.virtualenvs/openai/bin/python -m pytest -q \
  plugins/phimm/tests/test_native_recipe.py plugins/phimm/tests/test_adapter.py \
  -k 'not vllm_no_repeat and not remax_advantage and not training_reward_metrics and not source_on_policy'
```

Result: **16 passed, 4 deselected**. Tests cover native composition, ordinary
Main component targets, strict overrides, interpolation propagation, composition
outside the repository working directory, training budgets, all 35 evaluation
entries/scorer dispatches, explicit evaluation inputs, raw dataset mixing,
parent ordering, real local JSONL formatting, removed-transform errors, byte
readers, cache selectors, scorer baselines and corpus/reward metrics.

The complete original adapter test command was also attempted:

```bash
/home/boren/.virtualenvs/openai/bin/python -m pytest -q plugins/phimm/tests/test_adapter.py
```

Result: **5 passed, 4 failed** during dependency imports. The environment has
`operator torchvision::nms does not exist`, cannot import Transformers'
`MistralForSequenceClassification`, and lacks `transfer_queue`. The affected
checks are vLLM's no-repeat processor, Main's ReMax estimator, TransferQueue
training metrics and policy loss/gradient equivalence. They remain in the suite;
no dependency failures are hidden or converted to skips.

Actual training/evaluation Hydra entry points were checked with `--cfg job
--resolve` from `/tmp`, using absolute PYTHONPATH entries, an explicit test
experiment name and a placeholder local evaluation model path. This composes
configuration without initializing workers. The entry-point commands were:

```bash
export PYTHONPATH=/home/boren/.codex/worktrees/ecef/verl/plugins/phimm/src:/home/boren/.codex/worktrees/ecef/verl/plugins/qwen35_audio/src:/home/boren/.codex/worktrees/ecef/verl
cd /tmp
/home/boren/.virtualenvs/openai/bin/python -m recipe.phimm.main_asr_remax --cfg job --resolve
/home/boren/.virtualenvs/openai/bin/python -m recipe.phimm.main_asr_eval --cfg job --resolve \
  trainer.experiment_name=local-config-check actor_rollout_ref.model.path=/models/merged
```

Both exited 0. Changed Python files pass `ruff check`, `ruff format --check` and
`python -m compileall -q`; `git diff --check` passes.

GPU model/dataclass initialization, remote data loading, distributed checkpoint
publication/resume and end-to-end training remain unvalidated in this local
refactor. No quality improvement is claimed. Historical quality parity failed;
the scorer change after reference step 120 remains a confounder, not an
established cause. No remote training was launched.

## Follow-up: make the configured no-repeat window effective

At the user's request, sampling now passes `window_size` instead of the ignored
`ngram_window` key. The YAML value 512 now controls the processor for sampled
rollouts, greedy baselines and validation. This intentionally changes decoding
behavior relative to the historical run. Request construction was extracted to
`agentic/sampling.py` so configuration overrides can be tested without importing
the GPU worker stack. The vLLM integration test now expects matches beyond
100 tokens to be blocked and matches beyond 512 tokens to remain allowed.


Follow-up sampling checks: **20 passed, 4 deselected** using the focused command
above. The production request builder was also exercised with isolated upstream
vLLM v0.24.0 processor classes on CPU: windows 100 and 512 blocked or allowed
matches at gaps 30, 150 and 600 as expected, for training and validation paths.
This does not validate the full vLLM worker stack.

## Follow-up: remove dataset disk caching

Removed `cache_ds`, `load_cached_ds`, `save_cached_ds`, the cache-only storage
module and hash helper, and all 44 `cache_name` declarations in retained dataset
manifests. Each dataset load now reads JSONL and applies the configured transforms
directly. Existing on-disk caches are not deleted. Audio-file prefetch/cache,
Azure model caching and training checkpoint/dataloader state remain supported.


Dataset-cache removal verification: **21 passed, 4 deselected** using the same
focused command. A new local-manifest test verifies that subsequent loads see
updated JSONL content without calling Dataset.load_from_disk/save_to_disk.

## Additional review findings

On local datasets 4.3.0, null-list normalization remains necessary:
`List(null)` plus `List(string)` fails concatenation before normalization and
passes afterward. However, the existing helper also promotes bare null leaves
to strings: `Value(null)` plus `Value(int64)` concatenates successfully without
the helper and fails after its string promotion. This inherited implementation
needs narrower or peer-schema-aware type alignment; it has not been changed as
part of the user's explanation request.

Validation max_samples is applied to segment rows before parent grouping. A
local three-segment parent with max_samples=2 returns only two segments but
retains the full-parent reference. Default max_samples=-1 is unaffected;
partial validation limits can produce misleading corpus metrics. The user
requested reverting the sample-limit removal, so the original Main-compatible
sample limiting behavior is retained.

No retained Base3a/evaluation manifest enables online VAD or silence trimming.
Removing those implementations does not remove existing segment metadata or
change the configured offline-presegmented input paths. Remote manifest/audio
contents have not been re-audited locally.

## Commit-time checks

Full pre-commit was run. Its pinned Ruff formatting/type-style findings were
fixed, and the four generated PPO reference configs were synchronized to include
the existing trainer_class extension. The repository-wide docs timestamp,
license and test-layout hooks report pre-existing imported documentation/skill
issues outside this change. These failures are not suppressed in the report.

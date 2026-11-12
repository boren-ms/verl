# Distributed ASR training loading

The phimm ReMax/DAPO training entry points can distribute audio reading,
tokenization and processor features over their actor/rollout nodes. This uses
`DistributedTrainingLoader` in `recipe/phimm/data/training_loader.py`.
The default remains the existing local loader. For a `v2611_dev/base` recipe:

```yaml
data:
  distributed_training_loading:
    enabled: true
    workers_per_node: 4
    prefetch_batches: 2
    timeout_s: 300
```

Or append `data.distributed_training_loading.enabled=true` at launch. This flag
is independent of the standalone generation loader flag. It changes fresh
training input preprocessing; validation loaders retain their existing behavior.
It does not launch a new training job, change the optimizer, or restart a job.

## Data flow and batch semantics

A local StatefulDataLoader now samples **indices only**, preserving its original
sampler, batch size, `drop_last`, worker count and torchdata checkpoint layout.
Its subprocesses no longer read audio or run the processor. The resolved dataset
mix and any source budgets remain on the coordinator. BCR continues to use
`bad_case_replay.fresh_batch_size`; replay merging and selection are unchanged.

The index batches are split into ordered shards. CPU actors are hard-pinned to
the nodes from the actor/rollout placement groups, with one CPU and zero GPUs
per actor. Node selection is lazy, after GPU workers have been placed. Only a
table-free dataset template and small metadata slices are serialized. Local
coordinator audio-cache paths are converted back to portable remote URIs.
The existing collator merges processed samples in sampler order before the
trainer's existing rollout, reward and optimizer logic.

For unreadable audio, training retains RLHFDataset's original global retry
indices and stride. Each shard includes the primary rows and possible retry
rows. A sparse table retains the **full dataset length and original indexes**,
so a failure cannot accidentally substitute a different shard-local sample.
Retries add metadata work (up to `max_audio_load_retries + 1` candidate records
per sampled input), while audio is read only for candidates actually attempted.

`prefetch_batches` bounds speculative global audio batches. Startup and fetches
have timeouts; epoch completion, an early-closed iterator, and exceptions kill
CPU actors. The phimm training entry point also closes the loader in `finally`.
Logs identify actual worker nodes, per-node sample counts and exposed input wait.
Training metrics include `timing_s/data_loading`; DAPO accumulates loading wait
across generation/filtering batches for an optimizer step. Startup is separate
from the per-batch exposed wait. These measurements do not redefine the
existing step-throughput metric.

## Checkpoints and order

Prefetching advances the index loader ahead of the trainer. The wrapper saves
a separate **committed** torchdata state after each batch delivered to the
training loop. Checkpointing reads that state, rather than the speculative
index-loader position. On early close or failure, unconsumed prefetch is
discarded and the index loader is restored to the committed position.

Ordinary torch RandomSampler does not itself persist its generator state.
New distributed checkpoints therefore also store the epoch's initial sampler
RNG under `_asr_training_sampler_epoch_generator`. This preserves raw PyTorch
sampling order, including its RNG consumption at epoch boundaries. The trainer
restores that extra state even when switching back to a local loader.

Existing local checkpoints are supported with their original matching
`dataloader_num_workers`, sampler and batch configuration. Tests cover local
worker counts 0 and 2, and real-data validation covers worker count 1. Historical
random-sampler checkpoints without the epoch RNG retain their original restore
semantics: a missing RNG state cannot reconstruct an unknown past epoch's
shuffle. This optimization does not repair information absent from those
checkpoints. Sequential checkpoints do not have that shuffle ambiguity.

The supported scope is the deterministic phimm RLHFDataset path with standard
SequentialSampler or explicitly seeded RandomSampler. Custom curriculum
samplers and datasets changed by `on_batch_end` are rejected, because speculative
prefetch could change their meaning. Arbitrary stochastic dataset transforms
and dataset/worker RNG checkpoint migration are outside this implementation.
Initial metadata construction, collation, feature transfer through the
coordinator, validation loading and replay-buffer persistence are unchanged.

## Small validation

```bash
PYTHONPATH=. python -m pytest -q \
  tests/recipe/phimm/test_training_loader.py \
  tests/trainer/ppo/test_distributed_asr_loading.py

PYTHONPATH=. python -m scripts.validate_asr_training_loading
```

The script defaults to 17 synthetic WAV rows and two local logical Ray nodes.
It compares two shuffled epochs to the local StatefulDataLoader, serializes
and reloads a checkpoint while three batches are prefetched, checks legacy
local-state restore, and checks actor cleanup. All runs are CPU-only.

For two authorized physical nodes and a small actual dataset:

```bash
VERL_USE_EXTERNAL_MODULES=hf_qwen35_audio \
python -m scripts.validate_asr_training_loading \
  --address auto --node-ids NODE_ID_1 NODE_ID_2 \
  --input-jsonl EXISTING_GENERATION_PART.jsonl \
  --processor-path '/root/.blobfile/*/PATH_TO_MATCHING/qwen_hf' \
  --rows 17 --batch-size 4 --num-workers 1 \
  --report validation-training-real.json
```

This loads the matching tokenizer/processor, **no model weights**, and performs
no GPU inference or optimizer updates. Small correctness checks do not establish
an end-to-end training speedup. Compare actual exposed loading wait and complete
optimizer-cycle wall times on comparable data when benchmarking training.

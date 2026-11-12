# Distributed generation loading

`main_asr_gen.py` can run audio reading and `RLHFDataset.__getitem__` (tokenization,
waveforms and processor features) in CPU actors on the nodes allocated to the
rollout worker group. Enable it for a config inheriting `v2611_dev/gen/gen_base`:

```yaml
data:
  distributed_loading:
    enabled: true
    workers_per_node: 4
    prefetch_batches: 2
    timeout_s: 300
```

The equivalent launch override is `data.distributed_loading.enabled=true`.
The default is disabled. Local `data.num_workers`, `prefetch_factor`,
`persistent_workers` and `pin_memory` apply to the local DataLoader only.
Distributed workers each reserve one CPU and zero GPUs. Four workers per node
across four nodes give 16 workers total; `torch` uses one intra-op thread in
each worker. Runtime node-affinity checks and per-batch `rows_by_node` logs
show where loading actually happened. Generation timing logs separate input
wait, dispatch/decode and output queue wait.

Each batch's metadata is split into contiguous ranges across workers. The
coordinator collects results in source order before calling the existing
collator and generation pipeline. A partial final batch is retained. Resume
starts at the original dataset offset; shuffle is rejected. Unreadable samples
fail the run rather than substituting a different sample. Loader startup and
batch fetch have timeouts, and actors are killed on exhaustion, early close,
or a generation pipeline failure.

The coordinator still loads dataset metadata once and performs collation,
GPU dispatch, scoring and output writing. It sends a dataset template without
its Arrow table to each actor, then sends only bounded metadata slices.
This avoids remote references to coordinator-local Arrow mmap files and
repeating full metadata initialization on every node. Paths under
`VERL_AUDIO_CACHE_ROOT` (default `~/data`) are restored to Orange URIs before
dispatch, preserving byte offsets, chunk indexes, concatenation and time
selectors. Other local audio files must be available on the selected nodes.

`prefetch_batches` bounds outstanding global loading batches, in addition to
the existing prepared/output queues controlled by `data.prefetch_depth`.
Account for waveform and feature sizes when increasing either setting.
This implementation still transfers processed CPU features through the
coordinator. It does not claim to remove that network/CPU cost or accelerate
initial metadata loading.

## Small validation

From this worktree:

```bash
PYTHONPATH=. python -m pytest -q tests/recipe/phimm/test_generation_loader.py
PYTHONPATH=. python -m scripts.validate_asr_generation_loading
```

The default smoke test uses 37 short synthetic WAVs and two logical Ray nodes
on the local host. It checks waveforms/features against a serial baseline,
row order, the final partial batch, resume at row 16, bad-audio failure, and
actor cleanup. Tests additionally check timeouts, explicit close, node
selection from placement groups, portable cache paths and invalid options.

For **two physical nodes** in an authorized Ray cluster, use a runtime working
directory containing this worktree's code and pass the exact target node IDs:

```bash
python -m scripts.validate_asr_generation_loading \
  --address auto --node-ids NODE_ID_1 NODE_ID_2 \
  --rows 37 --batch-size 8 --workers-per-node 1
```

To validate real `RLHFDataset` and matching tokenizer/processor objects, sample
an existing generation JSONL (preserving prompt, reference and audio selectors)
and point to a complete **local cached** processor directory. This does not
load model weights or perform GPU inference:

```bash
VERL_USE_EXTERNAL_MODULES=hf_qwen35_audio \
python -m scripts.validate_asr_generation_loading \
  --address auto --node-ids NODE_ID_1 NODE_ID_2 \
  --input-jsonl EXISTING_GENERATION_PART.jsonl \
  --processor-path '/root/.blobfile/*/PATH_TO_MATCHING/qwen_hf' \
  --rows 17 --batch-size 4 --workers-per-node 1 \
  --report validation-real.json
```

These are correctness and placement checks. Their startup-dominated wall times
are not production throughput measurements. Measure an actual generation run's
input wait and end-to-end throughput on comparable audio before claiming a
speedup. Use a separate output directory for any such generation experiment.

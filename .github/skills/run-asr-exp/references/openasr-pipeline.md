# OpenASR pipeline, checkpoint export and prefetch

Apply the shared job, resource, authorization, monitoring and completion rules
in [SKILL.md](../SKILL.md). Execute only requested stages:

`optional training -> requested or last complete checkpoint -> verified HF export -> standalone evaluation -> requested workbook`.

For an evaluation-only request recover producer config, base model, rank/alpha,
topology and output root from checkpoint provenance. Default eval config is
`recipe/phimm/config/v2611_dev/eval/eval_2609r2_openall_zip.yaml`; default
workbook path is `tmp/openasr_report/<train-stem>_step<step>.xlsx`. Never
stop or change source training to evaluate a complete checkpoint.

## State and checkpoint selection

Keep durable pipeline state under ignored `tmp/run_openasr_exp/<train-stem>/`:
effective train/eval configs, restrictions and selected pools/topology, names,
overrides, revisions, output roots, checkpoint policy and producer provenance
(base, rank/alpha, world size, cadence), desired training target `T`, verified
resume/current steps and independent completion status for each requested stage;
submission IDs, links and replacement
authorization; checkpoint fingerprints and shard readiness, export proof,
evaluation detail paths, report inputs and verification; queue order and fresh
readiness observations; prefetch inventory/hash, main pod, user/HOME/cache,
process, logs and summary; and one shared monitor identity. Never store
credentials or signed URLs. Reconcile live jobs and artifacts before resuming;
same names or old files alone do not prove identity.

Enumerate the producer's `global_step_<N>` directories numerically. A
checkpoint is ready only when provenance matches, all expected nonempty
`model_world_size_<W>_rank_<R>.pt` shards for `R=0..W-1` exist and writes/uploads
are complete. For an explicit step/path never substitute. Otherwise select the
largest complete step, even if a newer directory is incomplete. A verified HF
export for that exact checkpoint can be reused; missing HF weights with complete
actor shards means export is needed.

For a queue, reconcile active work first and inspect each unfinished request at
most once per tick. Move a genuinely missing/incomplete checkpoint to the tail,
preserve the relative order of others and inspect the next immediately. Start
the first ready request. Diagnose authentication, unreadable artifacts and
unresolved provenance as errors, not missing checkpoints. Never duplicate an
active export, evaluation or report.

## Exact checkpoint export

Derive rank/alpha and base model from the effective producer, and use the
repository converter:

```bash
python3 plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py \
  --input <checkpoint-root> --output <hf-dir>/model.pt \
  --match-lora-merged --lora-alpha <effective-alpha> --lora-rank <effective-rank>
```

Use a dedicated local cache for `az://` shards. Verify rank coverage, merged
adapter-pair count and rank/alpha provenance. Convert the top-level tensor
dictionary to `model.safetensors`, removing `.base_layer.` from keys with
collision checks and confirming no adapter tensors remain. Copy config,
tokenizer, preprocessor and required custom model code from the actual base,
not base weights or a stale sharded index. Verify readability, keys/shapes,
assets and uploaded files before recording the URI. Invalidate only identified
stale cache/export files.

## Nonblocking audio prefetch

After training starts, prefetch as an optional optimization, never an
evaluation gate. Resolve the actual evaluation config's complete `data.val_data`
inventory, including nested lists, `path_map` and `rename_fields` in loader
order. Build a cache-only JSONL/YAML in the state directory preserving complete
`archive.zip!offset:length` and segment selectors; deduplicate physical archives
without losing distinct members. Verify every configured source is covered.
Treat unsupported inputs, missing paths, empty inventory, auth or disk errors
as prefetch failures, never remove evaluation datasets to make it work.

On the verified evaluation main/head pod only, under the evaluator's user/HOME
and cache root, start one background command from `/root/code/verl`:

```bash
python -m recipe.phimm.cache_eval_audio <verified-cache-dataset.yaml> \
  --workers 16 --audio-fields <fields>
```

Capture process identity, logs, exit status and JSON summary. Do not run worker
pod prefetchers or await prefetch before evaluation. Reconcile identity and
hashes before retrying; reduce concurrency if I/O harms decoding. At pipeline
completion stop only this pipeline's exact remaining prefetch process; it must
not keep the pipeline open. A `_zip` filename alone does not prove ZIP-backed
audio: inspect manifest rows and verify a byte-range read for new ZIP manifests.

## Evaluation and reporting

Default evaluation topology is the producer's effective `trainer.nnodes`, not
the eval YAML default, pool-name count or currently idle node count; changing it
requires authorization. Require the selected evaluation pool and required GPU
Ray nodes to be available. Source training on separate capacity may continue.
Submit the exact eval YAML and verified HF export with candidate-specific name,
`trainer.resume_mode=disable`, verified topology, and
`actor_rollout_ref.model.lora_rank=0` for a merged model. Reconcile but do not
await prefetch. Verify runtime topology and require final `p_err` metrics and
exact detail artifacts for every configured dataset, with Ray `SUCCEEDED`.
Preserve explicitly requested extra datasets separately from canonical rows.

For a requested workbook, create these state-directory inputs:

- `metrics.json`: candidate `val-aux/<source>/p_err/mean@1` fractions.
- `model_info.json`: export URI, eval config, pool, Ray ID, W&B URL and revision.
- `dataset_results.json`: canonical dataset key to exact detail-result file.

Resolve canonical identities from composed manifests; do not alias cleaned and
uncleaned or short and long datasets by name. Use `openasr-report` for baseline
values, rows, aggregation, formulas, formatting and quality gates. Reopen and
verify the workbook and model/result metadata. Complete the requested pipeline
only when every requested stage is verified, including training reaching `T`
under the shared completion rule when training was requested. Evaluation and
reporting may finish earlier for an explicit intermediate checkpoint, but keep
training and the shared monitor active until its own target is verified. For a
default final-checkpoint report, resolve the last complete checkpoint after
training completion rather than freezing an early checkpoint as the final one;
report checkpoint step/URI, training and evaluation identities/links, report
path, English and ML metrics, extra-suite results and provenance limitations.

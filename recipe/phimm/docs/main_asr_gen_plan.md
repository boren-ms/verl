# Plan: `main_asr_gen.py` — ASR Generation with vLLM Async Servers

## Overview

Create `recipe/phimm/main_asr_gen.py` that uses **standalone vLLM server replicas** (from the fully-async infrastructure) for ASR batch generation/inference. This replaces the old FSDP-based `ActorRolloutRefWorker.generate_sequences()` approach which is no longer supported in verl-mirror (sync `generate_sequences` on `ServerAdapter` raises `NotImplementedError`).

**Example config**: `recipe/phimm/config/gen/gen_oss_ls.yaml` (LibriSpeech evaluation).

---

## Background: Old vs New Approach

| Aspect | Old (reference `/home/boren/code/verl/`) | New (verl-mirror) |
|--------|------------------------------------------|--------------------|
| **Worker** | `ActorRolloutRefWorker` + `create_colocated_worker_cls` + `RayWorkerGroup` | Standalone `vLLMReplica` via `get_rollout_replica_class("vllm")` |
| **Model loading** | Full FSDP model on each worker | vLLM `AsyncLLM` engine per replica |
| **Generation API** | `wg.generate_sequences(DataProto)` — sync, batch-level | `LLMServerClient.generate(prompt_ids, audio_data)` — async, per-sample |
| **Multimodal** | Pre-computed `multi_modal_inputs` via `DataProto` | Raw audio arrays passed to vLLM, which computes embeddings internally |
| **Parallelism** | Threaded producer/consumer | Threaded producer/consumer + asyncio concurrent requests within batch |
| **Load balancing** | Implicit (single worker group) | `GlobalRequestLoadBalancer` across N replicas |

---

## Files to Create

### 1. `recipe/phimm/config/gen/gen_oss_ls.yaml`

Config for LibriSpeech generation. Structure:

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config
defaults:
  - ppo_trainer       # reuse existing defaults chain
  - _self_

data:
  custom_cls:
    path: recipe/phimm/data/rl_dataset.py
    name: RLHFDataset
  train_data:           # inline LibriSpeech TSV config
    - dataset_name: tsv
      tsv_paths: az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe.tsv
      add_task_info:
        task: lang_asr
      post_process:
        add_field:
          fields:
            data_source: asr
        verl_format:
          prompt_key: prompt
  batch_size: 512
  output_path: az://orngwus2cresco/data/boren/data/verl/gen_qwen/en_ls
  output_split_size: 20000
  prefetch_depth: 3
  max_prompt_length: 1024
  max_response_length: 768
  max_audio_dur: 40
  num_workers: 4
  return_multi_modal_inputs: false   # not needed for vLLM server path

actor_rollout_ref:
  model:
    path: az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/fast-llm-2605-qwen3-5-9b-s2-st-example/90000/qwen_hf/
    trust_remote_code: true
  rollout:
    name: vllm
    temperature: 0.0
    top_p: 1.0
    response_length: 768
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.85
    max_num_seqs: 24
    enforce_eager: false
    enable_chunked_prefill: true
    load_format: auto               # load real weights (not dummy)
    no_repeat_ngram_size: 15
    stop: ["<|im_end|>"]
    stop_token_ids: [248044, 248046]

trainer:
  n_gpus_per_node: 8
  nnodes: 1
```

### 2. `recipe/phimm/main_asr_gen.py`

Main generation script. See detailed flow below.

---

## Detailed Flow

### Startup

```
Hydra main → run_generation(config)
  ├── Init Ray with env vars
  ├── Load tokenizer + processor (for decoding)
  ├── Create RLHFDataset + StatefulDataLoader
  ├── Resume check (scan existing parquet parts)
  └── ray.get(main_task.remote(config))  # run in Ray task for resource management
```

### Server Launch (inside Ray task)

```python
# Same pattern as main_generation_server.py
rollout_server_class = get_rollout_replica_class("vllm")
num_replicas = (n_gpus_per_node * nnodes) // tp_size

replicas = [rollout_server_class(
    replica_rank=i, config=rollout_config,
    model_config=model_config, gpus_per_node=n_gpus_per_node
) for i in range(num_replicas)]

await asyncio.gather(*[r.init_standalone() for r in replicas])
server_handles = [r._server_handle for r in replicas]
```

### Load Balancer Setup

```python
load_balancer = GlobalRequestLoadBalancer.options(
    name="gen_load_balancer"
).remote(server_handles)
client = LLMServerClient(load_balancer)
```

### Generation Loop (pipelined)

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Producer    │────▶│  Main Thread     │────▶│  Consumer        │
│  (threading) │     │  (asyncio gen)   │     │  (threading)     │
│              │     │                  │     │                  │
│ DataLoader   │     │ For each sample: │     │ Decode tokens    │
│ → batch_dict │     │  1. Unpad IDs    │     │ eval_score(WER)  │
│ → extract    │     │  2. Extract audio│     │ Write parquet    │
│   metadata   │     │  3. client.gen() │     │ Log examples     │
└─────────────┘     │  (async gather)  │     └──────────────────┘
                    └──────────────────┘
```

### Per-Sample Generation

```python
# Extract unpadded prompt_ids from left-padded input_ids
attention_mask = sample["attention_mask"]
input_ids = sample["input_ids"]
valid_start = attention_mask.sum()  # number of valid tokens
prompt_ids = input_ids[-valid_start:].tolist()

# Raw audio from dataset
audio_data = sample["multi_modal_data"]["audio"]  # [(wav_array, sample_rate), ...]

# Generate via load-balanced client
result = await client.generate(
    request_id=uuid.uuid4().hex,
    prompt_ids=prompt_ids,
    sampling_params={"temperature": 0.0, "max_tokens": 768, ...},
    audio_data=audio_data,
    mm_processor_kwargs={"sampling_rate": processor.feature_extractor.sampling_rate},
)
# result.token_ids → decode → eval_score → WER
```

### Output Format

Same as reference: parquet splits (`part-000.parquet`, `part-001.parquet`, ...) with columns:
- `prompt`, `text` (ground truth), `audio_path`, `audio_chunk`
- `response` (parsed ASR text), `raw_response` (full decoded output)
- `n_err`, `n_ref`, `wer`, `n_edge` (WER metrics from `eval_score`)
- Any `extra_info` fields from the dataset

### Resume Support

On restart, scans `output_path/part-*.parquet` for contiguous parts, counts existing examples, and skips the corresponding number of dataset samples. Requires `batch_size`-aligned boundaries.

---

## Key Dependencies (existing code, read-only)

| File | What we use |
|------|-------------|
| `verl/workers/rollout/replica.py` | `get_rollout_replica_class()`, `RolloutReplica.init_standalone()` |
| `verl/workers/rollout/llm_server.py` | `GlobalRequestLoadBalancer`, `LLMServerClient` |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `vLLMHttpServer.generate(prompt_ids, audio_data, ...)` |
| `recipe/phimm/data/rl_dataset.py` | `RLHFDataset` — produces `input_ids` (with audio placeholders) + `multi_modal_data["audio"]` |
| `recipe/phimm/reward/asr_edge.py` | `eval_score()` for WER computation |
| `recipe/phimm/utils/shared.py` | `parse_asr_response()` for response parsing |
| `recipe/phimm/utils/env.py` | `EnvMgr` for cluster environment variables |
| `verl/utils/dataset/rl_dataset.py` | `collate_fn` for DataLoader |

---

## Design Decisions

1. **`LLMServerClient` (Ray actor-level), not HTTP API**: The HTTP `/v1/chat/completions` endpoint cannot handle raw audio arrays. The Ray actor-level `generate()` method on `vLLMHttpServer` accepts `audio_data` directly.

2. **No `AgentLoopWorker`**: Direct client calls are simpler for single-turn batch generation. AgentLoopWorker adds overhead (Ray actor setup, agent loop registry, tools) unnecessary here.

3. **No `pad_dataproto_to_divisor`**: The old approach needed padding for FSDP's world_size alignment. vLLM handles variable-length inputs natively—each request is independent.

4. **`return_multi_modal_inputs: false`**: The dataset doesn't need to compute `input_audio_embeds` (which vLLM computes internally from raw audio). We only need `input_ids` (tokenized with audio placeholders) and `multi_modal_data["audio"]` (raw waveforms).

5. **`load_format: auto`**: Standalone replicas must load real model weights (not `dummy` which is for sleep/wake hybrid mode during training).

6. **Config via `ppo_trainer` defaults**: Reuses the existing verl-mirror Hydra config composition chain. No separate `generation.yaml` base config needed.

---

## Verification Steps

1. **Imports**: `python3 -c "from recipe.phimm.main_asr_gen import main"`
2. **Config resolution**: `python3 -m recipe.phimm.main_asr_gen --config-path=../../recipe/phimm/config/gen --config-name=gen_oss_ls --cfg job`
3. **Full run**: Execute on LibriSpeech, check parquet output + WER stats in stdout
4. **Baseline comparison**: Compare WER with known baseline for the model/dataset

---

## Usage

```bash
# From repo root
python3 -m recipe.phimm.main_asr_gen \
    --config-path=../../recipe/phimm/config/gen \
    --config-name=gen_oss_ls

# Override model path
python3 -m recipe.phimm.main_asr_gen \
    --config-path=../../recipe/phimm/config/gen \
    --config-name=gen_oss_ls \
    actor_rollout_ref.model.path=/path/to/model

# Override output path and batch size
python3 -m recipe.phimm.main_asr_gen \
    --config-path=../../recipe/phimm/config/gen \
    --config-name=gen_oss_ls \
    data.output_path=/tmp/gen_output \
    data.batch_size=64
```

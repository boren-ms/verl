"""ASR batch generation using standalone vLLM server replicas.

Replaces the old FSDP-based ActorRolloutRefWorker.generate_sequences()
approach. Uses the fully-async vLLM infrastructure with a
GlobalRequestLoadBalancer for multi-replica inference.

Usage:
    python3 -m recipe.phimm.main_asr_gen \
        --config-path=../../recipe/phimm/config/gen \
        --config-name=gen_oss_ls
"""

import asyncio
import logging
import os
import time
import uuid

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _scan_existing_parts(output_path: str) -> tuple[int, int]:
    """Scan existing parquet parts and return (total_examples, next_part_idx).

    Only counts contiguous parts starting from part-00000.
    """
    try:
        import blobfile as bf

        files = list(bf.listdir(output_path))
    except (FileNotFoundError, OSError):
        return 0, 0

    parts = sorted(f for f in files if f.startswith("part-") and f.endswith(".parquet"))
    total = 0
    for i, part in enumerate(parts):
        expected = f"part-{i:05d}.parquet"
        if part != expected:
            break
        try:
            part_path = os.path.join(output_path, part)
            df = pd.read_parquet(part_path)
            total += len(df)
        except Exception:
            break
    else:
        i = len(parts)
    return total, i


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

async def _generate_batch(server_handles, batch_items, sampling_params, processor_sr):
    """Generate for all items in a batch concurrently via round-robin servers."""
    tasks = []
    for i, item in enumerate(batch_items):
        request_id = uuid.uuid4().hex
        server = server_handles[i % len(server_handles)]
        multimodal_kwargs = {}
        if item.get("audio_data") is not None:
            multimodal_kwargs["audio_data"] = item["audio_data"]
            multimodal_kwargs["mm_processor_kwargs"] = {"sampling_rate": processor_sr}
        tasks.append(
            server.generate.remote(
                request_id=request_id,
                prompt_ids=item["prompt_ids"],
                sampling_params=dict(sampling_params),
                **multimodal_kwargs,
            )
        )
    # Gather ray remote calls
    results = []
    for task in tasks:
        try:
            result = await task
            results.append(result)
        except Exception as e:
            results.append(e)
    return results


def _extract_scalar(arr, idx):
    """Safely extract a scalar from a numpy object array or tensor batch."""
    if arr is None:
        return None
    val = arr[idx]
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return val.item()
    if isinstance(val, np.ndarray) and val.size == 1:
        return val.flat[0]
    return val


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def _normalize_config(config):
    """Ensure config values are compatible with downstream consumers.

    Hydra may unwrap single-element YAML lists into a DictConfig.
    The dataset loader expects ``data.train_data`` to be a list of dicts.
    """
    from omegaconf import DictConfig, ListConfig

    for key in ("train_data", "val_data"):
        td = OmegaConf.select(config, f"data.{key}", default=None)
        if td is not None and isinstance(td, DictConfig):
            OmegaConf.update(config, f"data.{key}", [OmegaConf.to_container(td, resolve=True)])


async def _run_generation_async(config):
    """Async entry: launch servers, iterate dataset, generate, write parquet."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    from recipe.phimm.reward.asr_edge import eval_score
    from recipe.phimm.utils.shared import parse_asr_response
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.hdfs_io import makedirs
    from verl.workers.rollout.replica import get_rollout_replica_class

    OmegaConf.resolve(config)
    _normalize_config(config)

    # ── tokenizer / processor ──────────────────────────────────────────
    from verl.utils.fs import copy_to_local

    model_path = config.actor_rollout_ref.model.path.rstrip("/")
    # Resolve az:// blob paths to local cache
    if model_path.startswith("az://"):
        import hashlib
        import subprocess

        cache_key = hashlib.md5(model_path.encode()).hexdigest()
        local_model_path = os.path.join(
            os.path.expanduser("~"), ".blobfile", cache_key,
            model_path.split("/")[-1],
        )
        if not os.path.isdir(local_model_path) or not os.path.exists(
            os.path.join(local_model_path, "config.json")
        ):
            os.makedirs(local_model_path, exist_ok=True)
            logger.info("Syncing model from %s to %s", model_path, local_model_path)
            subprocess.run(
                ["bbb", "sync", "--concurrency", "64", model_path + "/", local_model_path + "/"],
                check=True,
            )
        logger.info("Using local model path: %s", local_model_path)
    else:
        local_model_path = copy_to_local(model_path)

    # Update config so downstream replicas also use the resolved local path
    OmegaConf.update(config, "actor_rollout_ref.model.path", local_model_path)

    trust_remote_code = config.actor_rollout_ref.model.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_remote_code)
    # Use AutoProcessor directly — verl-mirror's hf_processor only handles VL models.
    # Newer transformers (4.57+) has strict class-identity checks for custom
    # processors that break when the same class is imported from two paths.
    # Patch ProcessorMixin to relax the type check.
    from transformers import AutoProcessor
    from transformers.processing_utils import ProcessorMixin

    _orig_check = ProcessorMixin.check_argument_for_proper_class

    def _relaxed_check(self, attribute_name, arg):
        try:
            _orig_check(self, attribute_name, arg)
        except (TypeError, ValueError):
            pass  # allow name-matched custom classes loaded via trust_remote_code

    ProcessorMixin.check_argument_for_proper_class = _relaxed_check
    try:
        processor = AutoProcessor.from_pretrained(
            local_model_path, trust_remote_code=trust_remote_code
        )
    finally:
        ProcessorMixin.check_argument_for_proper_class = _orig_check
    processor_sr = getattr(
        processor, "feature_extractor",
        getattr(processor, "audio_feature_extractor", None),
    ).sampling_rate

    # ── dataset / dataloader ───────────────────────────────────────────
    from recipe.phimm.data.rl_dataset import RLHFDataset

    # Pass train_data directly as data_files (gen uses train_data by convention)
    ds_conf = OmegaConf.select(config, "data.gen_data",
              default=OmegaConf.select(config, "data.train_data",
              default=OmegaConf.select(config, "data.val_data")))
    # Support both old verl (data_confs) and verl-mirror (data_files) RLHFDataset API
    try:
        dataset = RLHFDataset(
            data_files=ds_conf,
            tokenizer=tokenizer,
            config=config.data,
            processor=processor,
        )
    except TypeError:
        dataset = RLHFDataset(
            data_confs=ds_conf,
            tokenizer=tokenizer,
            config=config.data,
            processor=processor,
        )
    logger.info("Dataset loaded: %d examples", len(dataset))

    batch_size = config.data.batch_size
    num_workers = config.data.get("num_workers", 4)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # ── output / resume ────────────────────────────────────────────────
    output_path = config.data.output_path
    makedirs(output_path, exist_ok=True)
    split_size = config.data.get("output_split_size", 20000)

    skip_count, part_idx = _scan_existing_parts(output_path)
    if skip_count > 0:
        logger.info("Resuming: skipping %d examples, starting from part %d", skip_count, part_idx)

    # ── sampling params ────────────────────────────────────────────────
    rollout_cfg = config.actor_rollout_ref.rollout
    sampling_params = {
        "temperature": float(rollout_cfg.temperature),
        "top_p": float(rollout_cfg.top_p),
        "max_tokens": int(rollout_cfg.response_length),
    }
    stop_token_ids = OmegaConf.select(config, "data.stop_token_ids", default=None)
    if stop_token_ids is not None:
        sampling_params["stop_token_ids"] = list(stop_token_ids)

    # ── launch vLLM server replicas ────────────────────────────────────
    logger.info("Launching vLLM server replicas …")
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    tp_size = rollout_config.tensor_model_parallel_size
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    num_replicas = n_gpus // tp_size

    rollout_server_class = get_rollout_replica_class(rollout_config.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=i,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for i in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])
    server_handles = [server._server_handle for server in rollout_servers]
    logger.info("Server replicas ready: %d replicas", len(server_handles))

    # Simple round-robin load balancer for direct server access
    _server_idx = [0]

    def _next_server():
        idx = _server_idx[0] % len(server_handles)
        _server_idx[0] += 1
        return server_handles[idx]

    def _write_parquet(df, dest_path):
        """Write parquet to local temp file then copy to blob if az://."""
        if dest_path.startswith("az://"):
            import blobfile as bf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                df.to_parquet(tmp.name, index=False)
                tmp_path = tmp.name
            bf.makedirs(os.path.dirname(dest_path))
            bf.copy(tmp_path, dest_path, overwrite=True)
            os.remove(tmp_path)
        else:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            df.to_parquet(dest_path, index=False)

    # ── generation loop ────────────────────────────────────────────────
    all_results: list[dict] = []
    total_n_err = 0
    total_n_ref = 0
    examples_done = 0
    skipped = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        actual_bs = batch["input_ids"].shape[0]

        # handle resume: skip already-processed examples
        if skipped + actual_bs <= skip_count:
            skipped += actual_bs
            continue
        offset = max(0, skip_count - skipped)
        skipped = max(skipped, skip_count)

        # ── prepare per-sample requests ────────────────────────────────
        batch_items = []
        batch_meta = []
        for i in range(offset, actual_bs):
            # Build prompt_ids with <audio> placeholder for vLLM's audio plugin.
            # The dataset's input_ids have pre-expanded audio_pad tokens which
            # the standalone vLLM plugin can't process — it needs the <audio>
            # text placeholder that it will replace during input processing.
            raw_ids = _extract_scalar(batch.get("raw_prompt_ids"), i)
            if raw_ids is not None:
                prompt_ids = list(raw_ids) if not isinstance(raw_ids, list) else raw_ids
                # Insert <audio> token sequence if not already present
                audio_tok_ids = tokenizer.encode("<audio>", add_special_tokens=False)
                has_audio = any(
                    prompt_ids[j:j + len(audio_tok_ids)] == audio_tok_ids
                    for j in range(len(prompt_ids) - len(audio_tok_ids) + 1)
                ) if len(audio_tok_ids) <= len(prompt_ids) else False
                if not has_audio:
                    # Find the position after <|im_start|>user\n and inject <audio>\n
                    newline_tok = tokenizer.encode("\n", add_special_tokens=False)
                    inject_pos = None
                    for j in range(len(prompt_ids)):
                        # After the first newline token (end of <|im_start|>user\n)
                        if prompt_ids[j:j + len(newline_tok)] == newline_tok:
                            inject_pos = j + len(newline_tok)
                            break
                    if inject_pos is not None:
                        prompt_ids = prompt_ids[:inject_pos] + audio_tok_ids + newline_tok + prompt_ids[inject_pos:]
            else:
                attn = batch["attention_mask"][i]
                ids = batch["input_ids"][i]
                valid_len = int(attn.sum().item())
                prompt_ids = ids[-valid_len:].tolist()

            audio_data = _extract_scalar(batch.get("multi_modal_data"), i)
            if isinstance(audio_data, dict):
                audio_data = audio_data.get("audio")

            # Extract ground truth from reward_model dict or text field
            gt_raw = _extract_scalar(batch.get("reward_model"), i)
            if isinstance(gt_raw, dict):
                ground_truth = gt_raw.get("ground_truth", gt_raw.get("gt_output", ""))
            else:
                ground_truth = _extract_scalar(batch.get("text"), i)

            batch_items.append({"prompt_ids": prompt_ids, "audio_data": audio_data})
            batch_meta.append({
                "ground_truth": ground_truth,
                "audio_path": _extract_scalar(batch.get("audio_path"), i),
                "data_source": _extract_scalar(batch.get("data_source"), i),
                "prompt": _extract_scalar(batch.get("raw_prompt"), i),
                "extra_info": _extract_scalar(batch.get("extra_info"), i),
            })

        if not batch_items:
            continue

        # ── generate ───────────────────────────────────────────────────
        t_batch = time.time()
        results = await _generate_batch(server_handles, batch_items, sampling_params, processor_sr)
        gen_time = time.time() - t_batch

        # ── decode + WER ───────────────────────────────────────────────
        for result, meta in zip(results, batch_meta):
            if isinstance(result, Exception):
                logger.error("Generation failed: %s", result)
                all_results.append({
                    "text": meta["ground_truth"],
                    "audio_path": meta["audio_path"],
                    "data_source": meta["data_source"],
                    "response": "",
                    "raw_response": f"ERROR: {result}",
                    "n_err": 0, "n_ref": 0, "wer": 0.0, "n_edge": 0,
                })
            else:
                raw_resp = tokenizer.decode(result.token_ids, skip_special_tokens=False)
                parsed = parse_asr_response(raw_resp)
                gt = meta["ground_truth"]
                gt_str = str(gt) if gt is not None else ""
                if gt_str:
                    scores = eval_score(raw_resp, gt_str)
                else:
                    scores = {"n_err": 0, "n_ref": 0, "wer": 0.0, "n_edge": 0}
                total_n_err += scores["n_err"]
                total_n_ref += scores["n_ref"]
                row = {
                    "text": gt,
                    "audio_path": meta["audio_path"],
                    "data_source": meta["data_source"],
                    "response": parsed["text"],
                    "raw_response": raw_resp,
                    **{k: scores[k] for k in ("n_err", "n_ref", "wer", "n_edge")},
                }
                ei = meta.get("extra_info")
                if isinstance(ei, dict):
                    for k, v in ei.items():
                        if k not in row:
                            row[k] = v
                all_results.append(row)
            examples_done += 1

        # ── flush parquet parts ────────────────────────────────────────
        while len(all_results) >= split_size:
            chunk = all_results[:split_size]
            all_results = all_results[split_size:]
            part_file = os.path.join(output_path, f"part-{part_idx:05d}.parquet")
            _write_parquet(pd.DataFrame(chunk), part_file)
            logger.info("Wrote %s (%d rows)", part_file, len(chunk))
            part_idx += 1

        # ── progress ───────────────────────────────────────────────────
        elapsed = time.time() - t0
        running_wer = (total_n_err / total_n_ref * 100) if total_n_ref > 0 else 0.0
        logger.info(
            "Batch %d | %d examples | batch %.1fs | elapsed %.1fs | WER %.2f%%",
            batch_idx, examples_done, gen_time, elapsed, running_wer,
        )

    # ── write remainder ────────────────────────────────────────────────
    if all_results:
        part_file = os.path.join(output_path, f"part-{part_idx:05d}.parquet")
        _write_parquet(pd.DataFrame(all_results), part_file)
        logger.info("Wrote %s (%d rows)", part_file, len(all_results))

    elapsed = time.time() - t0
    avg_wer = (total_n_err / total_n_ref * 100) if total_n_ref > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"Generation complete: {examples_done} examples in {elapsed:.1f}s")
    print(f"Average WER: {avg_wer:.2f}% ({total_n_err}/{total_n_ref})")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="config/gen", config_name="gen_oss_ls", version_base=None)
def main(config):
    from pathlib import Path
    from pprint import pprint

    from recipe.phimm.utils.env import EnvMgr

    # Register 'eval' resolver needed by rollout config interpolations
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

    env_mgr = EnvMgr()
    env_vars = env_mgr.envs()
    print(f"Cluster Env: {env_vars}")

    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                **env_vars,
            },
            "excludes": [str(Path(__file__).parents[2] / ".git")],
        }
        ray_init_kwargs = OmegaConf.to_container(
            config.get("ray_kwargs", {}).get("ray_init", {}), resolve=True
        ) or {}
        runtime_env = {**default_runtime_env, **ray_init_kwargs.pop("runtime_env", {})}
        ray_init_kwargs["runtime_env"] = runtime_env
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_generation_async(config))


if __name__ == "__main__":
    main()

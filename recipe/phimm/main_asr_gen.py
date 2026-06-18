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

async def _generate_batch_client(client, batch_items, sampling_params, processor_sr):
    """Generate a batch of requests via LLMServerClient."""
    tasks = []
    for item in batch_items:
        request_id = uuid.uuid4().hex
        tasks.append(
            client.generate(
                request_id=request_id,
                prompt_ids=item["prompt_ids"],
                sampling_params=dict(sampling_params),
                audio_data=item.get("audio_data"),
                mm_processor_kwargs={"sampling_rate": processor_sr},
            )
        )
    return await asyncio.gather(*tasks, return_exceptions=True)



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
    from recipe.phimm.reward.asr_edge import eval_score
    from recipe.phimm.utils.shared import parse_asr_response
    from verl.utils import hf_tokenizer
    from verl.utils.hdfs_io import makedirs

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
    dataset = RLHFDataset(
        data_files=ds_conf,
        tokenizer=tokenizer,
        config=config.data,
        processor=processor,
    )
    logger.info("Dataset loaded: %d examples", len(dataset))

    batch_size = config.data.batch_size

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
    from verl.workers.rollout.llm_server import LLMServerManager

    logger.info("Launching vLLM server replicas …")
    server_manager = await LLMServerManager.create(config=config)
    client = server_manager.get_client()
    logger.info("Server replicas ready (LLMServerManager)")

    def _write_parquet(df, dest_path):
        """Write parquet to local temp file then copy to blob if az://."""
        if dest_path.startswith("az://"):
            import tempfile

            import blobfile as bf

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
    n_total = len(dataset)
    audio_tok_ids = tokenizer.encode("<audio>", add_special_tokens=False)
    newline_tok = tokenizer.encode("\n", add_special_tokens=False)
    t0 = time.time()

    for batch_start in range(skip_count, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_idx = batch_start // batch_size

        # ── prepare per-sample requests ────────────────────────────────
        batch_items = []
        batch_meta = []
        for idx in range(batch_start, batch_end):
            item = dataset[idx]

            # Build prompt_ids with <audio> placeholder for vLLM's audio plugin.
            raw_ids = item.get("raw_prompt_ids")
            if raw_ids is not None:
                prompt_ids = list(raw_ids) if not isinstance(raw_ids, list) else raw_ids
                has_audio = any(
                    prompt_ids[j:j + len(audio_tok_ids)] == audio_tok_ids
                    for j in range(len(prompt_ids) - len(audio_tok_ids) + 1)
                ) if len(audio_tok_ids) <= len(prompt_ids) else False
                if not has_audio:
                    inject_pos = None
                    for j in range(len(prompt_ids)):
                        if prompt_ids[j:j + len(newline_tok)] == newline_tok:
                            inject_pos = j + len(newline_tok)
                            break
                    if inject_pos is not None:
                        prompt_ids = prompt_ids[:inject_pos] + audio_tok_ids + newline_tok + prompt_ids[inject_pos:]
            else:
                attn = item["attention_mask"]
                ids = item["input_ids"]
                valid_len = int(attn.sum().item())
                prompt_ids = ids[-valid_len:].tolist()

            audio_data = item.get("multi_modal_data")
            if isinstance(audio_data, dict):
                audio_data = audio_data.get("audio")

            gt_raw = item.get("reward_model")
            if isinstance(gt_raw, dict):
                ground_truth = gt_raw.get("ground_truth", gt_raw.get("gt_output", ""))
            else:
                ground_truth = item.get("text")

            batch_items.append({"prompt_ids": prompt_ids, "audio_data": audio_data})
            batch_meta.append({
                "ground_truth": ground_truth,
                "audio_path": item.get("audio_path"),
                "data_source": item.get("data_source"),
                "prompt": item.get("raw_prompt"),
                "extra_info": item.get("extra_info"),
            })

        # ── generate ───────────────────────────────────────────────────
        t_batch = time.time()
        results = await _generate_batch_client(client, batch_items, sampling_params, processor_sr)
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

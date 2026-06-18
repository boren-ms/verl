"""ASR generation using standalone vLLM server replicas (item-by-item).

Usage:
    python3 -m recipe.phimm.main_asr_gen \
        --config-path=../../recipe/phimm/config/gen \
        --config-name=gen_oss_ls
"""

import asyncio
import hashlib
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin

from recipe.phimm.data.rl_dataset import RLHFDataset
from recipe.phimm.reward.asr_edge import eval_score
from recipe.phimm.utils.env import EnvMgr
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.llm_server import LLMServerManager

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _normalize_config(config):
    """Unwrap single-element YAML lists that Hydra collapses to DictConfig."""
    for key in ("train_data", "val_data"):
        td = OmegaConf.select(config, f"data.{key}", default=None)
        if td is not None and isinstance(td, DictConfig):
            OmegaConf.update(config, f"data.{key}", [OmegaConf.to_container(td, resolve=True)])


def _build_sampling_params(config):
    """Extract sampling params dict from rollout config."""
    r = config.actor_rollout_ref.rollout
    params = {
        "temperature": float(r.temperature),
        "top_p": float(r.top_p),
        "max_tokens": int(r.response_length),
    }
    stop_ids = OmegaConf.select(config, "data.stop_token_ids", default=None)
    if stop_ids is not None:
        params["stop_token_ids"] = list(stop_ids)
    return params


# ---------------------------------------------------------------------------
# Model / processor loading
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str) -> str:
    """Resolve az:// blob paths to a local cache; copy_to_local for others."""
    if not model_path.startswith("az://"):
        return copy_to_local(model_path)

    cache_key = hashlib.md5(model_path.encode()).hexdigest()
    local = os.path.join(os.path.expanduser("~"), ".blobfile", cache_key, model_path.split("/")[-1])
    if not os.path.isdir(local) or not os.path.exists(os.path.join(local, "config.json")):
        os.makedirs(local, exist_ok=True)
        logger.info("Syncing model from %s to %s", model_path, local)
        subprocess.run(["bbb", "sync", "--concurrency", "64", model_path + "/", local + "/"], check=True)
    logger.info("Using local model path: %s", local)
    return local


def _load_processor(model_path: str, trust_remote_code: bool):
    """Load AutoProcessor with a relaxed type-check monkey-patch for custom processors."""
    orig = ProcessorMixin.check_argument_for_proper_class

    def _relaxed(self, attribute_name, arg):
        try:
            orig(self, attribute_name, arg)
        except (TypeError, ValueError):
            pass

    ProcessorMixin.check_argument_for_proper_class = _relaxed
    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    finally:
        ProcessorMixin.check_argument_for_proper_class = orig


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _scan_existing_parts(output_path: str) -> tuple[int, int]:
    """Return (total_examples, next_part_idx) from contiguous parquet parts."""
    try:
        files = list(bf.listdir(output_path))
    except (FileNotFoundError, OSError):
        return 0, 0

    parts = sorted(f for f in files if f.startswith("part-") and f.endswith(".parquet"))
    total = 0
    for i, part in enumerate(parts):
        if part != f"part-{i:05d}.parquet":
            break
        try:
            total += len(pd.read_parquet(os.path.join(output_path, part)))
        except Exception:
            break
    else:
        i = len(parts)
    return total, i


def _write_parquet(df: pd.DataFrame, dest_path: str):
    """Write parquet, transparently handling az:// blob destinations."""
    if dest_path.startswith("az://"):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            tmp_path = tmp.name
        bf.makedirs(os.path.dirname(dest_path))
        bf.copy(tmp_path, dest_path, overwrite=True)
        os.remove(tmp_path)
    else:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        df.to_parquet(dest_path, index=False)


# ---------------------------------------------------------------------------
# Item preparation
# ---------------------------------------------------------------------------

def _prepare_prompt_ids(item, audio_tok_ids, newline_tok):
    """Extract prompt token IDs from a dataset item, injecting <audio> if missing."""
    raw_ids = item.get("raw_prompt_ids")
    if raw_ids is None:
        attn = item["attention_mask"]
        valid_len = int(attn.sum().item())
        return item["input_ids"][-valid_len:].tolist()

    prompt_ids = list(raw_ids) if not isinstance(raw_ids, list) else raw_ids
    if len(audio_tok_ids) <= len(prompt_ids) and any(
        prompt_ids[j:j + len(audio_tok_ids)] == audio_tok_ids
        for j in range(len(prompt_ids) - len(audio_tok_ids) + 1)
    ):
        return prompt_ids

    # Inject <audio>\n after the first newline
    for j in range(len(prompt_ids)):
        if prompt_ids[j:j + len(newline_tok)] == newline_tok:
            pos = j + len(newline_tok)
            return prompt_ids[:pos] + audio_tok_ids + newline_tok + prompt_ids[pos:]
    return prompt_ids


def _prepare_item(item, audio_tok_ids, newline_tok):
    """Build (gen_input, meta) for a single dataset item."""
    prompt_ids = _prepare_prompt_ids(item, audio_tok_ids, newline_tok)

    audio_data = item.get("multi_modal_data")
    if isinstance(audio_data, dict):
        audio_data = audio_data.get("audio")

    gt_raw = item.get("reward_model")
    ground_truth = (
        gt_raw.get("ground_truth", gt_raw.get("gt_output", ""))
        if isinstance(gt_raw, dict) else item.get("text")
    )

    gen_input = {"prompt_ids": prompt_ids, "audio_data": audio_data}
    meta = {
        "ground_truth": ground_truth,
        "audio_path": item.get("audio_path"),
        "data_source": item.get("data_source"),
        "prompt": item.get("raw_prompt"),
        "extra_info": item.get("extra_info"),
    }
    return gen_input, meta


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------

async def _generate_one(client, gen_input, sampling_params, processor_sr):
    """Generate for a single item. Returns raw token IDs or an Exception."""
    return await client.generate(
        request_id=uuid.uuid4().hex,
        prompt_ids=gen_input["prompt_ids"],
        sampling_params=dict(sampling_params),
        audio_data=gen_input.get("audio_data"),
        mm_processor_kwargs={"sampling_rate": processor_sr},
    )


def _score_one(result, meta, tokenizer, eval_score):
    """Decode + score a single result. Returns (row, n_err, n_ref)."""
    gt = meta["ground_truth"]
    base = {"text": gt, "audio_path": meta["audio_path"], "data_source": meta["data_source"]}

    if isinstance(result, Exception):
        logger.error("Generation failed: %s", result)
        return {**base, "response": "", "raw_response": f"ERROR: {result}",
                "n_err": 0, "n_ref": 0, "wer": 0.0, "n_edge": 0}, 0, 0

    raw_resp = tokenizer.decode(result.token_ids, skip_special_tokens=False)
    gt_str = str(gt) if gt is not None else ""
    scores = eval_score(raw_resp, gt_str) if gt_str else {"n_err": 0, "n_ref": 0, "wer": 0.0, "n_edge": 0}

    row = {**base, "response": raw_resp, "raw_response": raw_resp,
           "n_err": scores["n_err"], "n_ref": scores["n_ref"], "wer": scores["wer"], "n_edge": scores["n_edge"]}
    ei = meta.get("extra_info")
    if isinstance(ei, dict):
        row.update({k: v for k, v in ei.items() if k not in row})
    return row, scores["n_err"], scores["n_ref"]


# ---------------------------------------------------------------------------
# Main async orchestration
# ---------------------------------------------------------------------------

async def _run_generation_async(config):
    """Launch servers, iterate dataset, generate, score, write parquet."""
    OmegaConf.resolve(config)
    _normalize_config(config)

    # ── model / tokenizer / processor ──────────────────────────────────
    local_model_path = _resolve_model_path(config.actor_rollout_ref.model.path.rstrip("/"))
    OmegaConf.update(config, "actor_rollout_ref.model.path", local_model_path)

    trust_rc = config.actor_rollout_ref.model.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_rc)
    processor = _load_processor(local_model_path, trust_rc)
    processor_sr = getattr(
        processor, "feature_extractor",
        getattr(processor, "audio_feature_extractor", None),
    ).sampling_rate

    # ── dataset ────────────────────────────────────────────────────────
    ds_conf = (OmegaConf.select(config, "data.gen_data", default=None)
               or OmegaConf.select(config, "data.train_data", default=None)
               or OmegaConf.select(config, "data.val_data"))
    dataset = RLHFDataset(data_files=ds_conf, tokenizer=tokenizer, config=config.data, processor=processor)
    logger.info("Dataset loaded: %d examples", len(dataset))

    # ── output / resume ────────────────────────────────────────────────
    output_path = config.data.output_path
    makedirs(output_path, exist_ok=True)
    split_size = config.data.get("output_split_size", 20000)
    skip_count, part_idx = _scan_existing_parts(output_path)
    if skip_count > 0:
        logger.info("Resuming: skipping %d examples, starting from part %d", skip_count, part_idx)

    # ── server replicas ────────────────────────────────────────────────
    sampling_params = _build_sampling_params(config)
    logger.info("Launching vLLM server replicas …")
    server_manager = await LLMServerManager.create(config=config)
    client = server_manager.get_client()
    logger.info("Server replicas ready")

    # ── generation loop (semaphore-bounded continuous feed) ───────────────
    n_total = len(dataset)
    audio_tok_ids = tokenizer.encode("<audio>", add_special_tokens=False)
    newline_tok = tokenizer.encode("\n", add_special_tokens=False)
    concurrency = config.data.get("concurrency", config.data.batch_size)
    log_interval = config.data.get("log_interval", 100)

    all_results: list[dict] = []
    total_n_err = total_n_ref = examples_done = 0
    t0 = time.time()
    sem = asyncio.Semaphore(concurrency)

    async def _worker(idx):
        async with sem:
            gen_input, meta = _prepare_item(dataset[idx], audio_tok_ids, newline_tok)
            try:
                result = await _generate_one(client, gen_input, sampling_params, processor_sr)
            except Exception as exc:
                result = exc
            return idx, _score_one(result, meta, tokenizer, eval_score)

    # Collect results in submission order via dict keyed by index
    pending: dict[int, asyncio.Task] = {}
    next_flush_idx = skip_count  # tracks the next index to flush in order

    async def _collect(task, idx):
        nonlocal total_n_err, total_n_ref, examples_done, next_flush_idx, part_idx
        _, (row, n_err, n_ref) = await task
        pending[idx] = row
        total_n_err += n_err
        total_n_ref += n_ref
        examples_done += 1

        # Flush results that are contiguous from next_flush_idx (preserves order)
        while next_flush_idx in pending:
            all_results.append(pending.pop(next_flush_idx))
            next_flush_idx += 1

        # Write parquet parts
        while len(all_results) >= split_size:
            chunk, all_results[:] = all_results[:split_size], all_results[split_size:]
            part_file = os.path.join(output_path, f"part-{part_idx:05d}.parquet")
            _write_parquet(pd.DataFrame(chunk), part_file)
            logger.info("Wrote %s (%d rows)", part_file, len(chunk))
            part_idx += 1

        if examples_done % log_interval == 0:
            running_wer = (total_n_err / total_n_ref * 100) if total_n_ref > 0 else 0.0
            logger.info(
                "%d/%d done | elapsed %.1fs | WER %.2f%%",
                examples_done, n_total - skip_count, time.time() - t0, running_wer,
            )

    # Feed tasks continuously — keep up to `concurrency` in-flight at all
    # times so server replicas never sit idle between chunks.
    pending_tasks: set[asyncio.Task] = set()
    submit_idx = skip_count

    def _submit_one():
        nonlocal submit_idx
        if submit_idx < n_total:
            t = asyncio.create_task(_worker(submit_idx))
            t._item_idx = submit_idx
            pending_tasks.add(t)
            submit_idx += 1

    # Seed initial batch
    for _ in range(min(concurrency, n_total - skip_count)):
        _submit_one()

    while pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for t in done:
            await _collect(t, t._item_idx)
            _submit_one()  # replace each finished task immediately

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
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

    env_mgr = EnvMgr()
    env_vars = env_mgr.envs()
    print(f"Cluster Env: {env_vars}")

    if not ray.is_initialized():
        ray_init_kwargs = OmegaConf.to_container(
            config.get("ray_kwargs", {}).get("ray_init", {}), resolve=True
        ) or {}
        runtime_env = {
            **{
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN", "HF_HUB_OFFLINE": "1",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    **env_vars,
                },
                "excludes": [str(Path(__file__).parents[2] / ".git")],
            },
            **ray_init_kwargs.pop("runtime_env", {}),
        }
        ray_init_kwargs["runtime_env"] = runtime_env
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**ray_init_kwargs)

    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_generation_async(config))


if __name__ == "__main__":
    main()

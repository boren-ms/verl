"""ASR generation using FullyAsyncRollouter + MessageQueue consumer.

Usage:
    python3 -m recipe.phimm.asr_rollout \
        --config-path=../../recipe/phimm/config/rollout \
        --config-name=rollout_oss_ls
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf

from recipe.phimm.utils.env import EnvMgr
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _normalize_config(config):
    for key in ("train_data", "val_data"):
        td = OmegaConf.select(config, f"data.{key}", default=None)
        if td is not None and isinstance(td, DictConfig):
            OmegaConf.update(config, f"data.{key}", [OmegaConf.to_container(td, resolve=True)])


def _resolve_model_path(model_path: str) -> str:
    if not model_path.startswith("az://"):
        return copy_to_local(model_path)
    cache_key = hashlib.md5(model_path.encode()).hexdigest()
    local = os.path.join(os.path.expanduser("~"), ".blobfile", cache_key, model_path.split("/")[-1])
    if not os.path.isdir(local) or not os.path.exists(os.path.join(local, "config.json")):
        os.makedirs(local, exist_ok=True)
        subprocess.run(["bbb", "sync", "--concurrency", "64", model_path + "/", local + "/"], check=True)
    return local


def _write_parquet(df: pd.DataFrame, dest_path: str):
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


def _flush_results(results: list[dict], output_path: str, part: int):
    path = os.path.join(output_path, f"part_{part:05d}.parquet")
    _write_parquet(pd.DataFrame(results), path)
    print(f"[Consumer] Wrote {path} ({len(results)} rows)")


_TXT_RE = re.compile(r"<TXT>(.*?)</TXT>", re.DOTALL)


def _ntb_get(ntb, key, default=None):
    """Read first element of a non_tensor_batch entry if present."""
    if key in ntb:
        try:
            return ntb[key][0]
        except (IndexError, TypeError, KeyError):
            return default
    return default


def _clean_response(raw_resp: str, tokenizer, response_ids) -> str:
    """Extract the transcription text from a raw ASR response.

    Prefers the content inside <TXT>...</TXT>; falls back to a
    special-token-stripped decode.
    """
    matches = _TXT_RE.findall(raw_resp)
    if matches:
        return matches[-1].strip()
    if response_ids is not None:
        return tokenizer.decode(
            [t for t in response_ids[0].tolist() if t != 0], skip_special_tokens=True
        ).strip()
    return re.sub(r"<\|[^|]*\|>", "", raw_resp).strip()


def _audio_from_raw_prompt(raw_prompt) -> str:
    """Extract an audio path/url from chat-style raw_prompt messages."""
    messages = raw_prompt
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    if not isinstance(messages, (list, tuple)):
        return ""
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, (list, tuple)):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "audio":
                return str(item.get("audio") or item.get("audio_url") or "")
    return ""


def _decode_rollout_sample(sample_bytes, tokenizer) -> dict:
    rollout_sample = ray.cloudpickle.loads(sample_bytes)
    ret = rollout_sample.full_batch

    response_ids = ret.batch.get("responses")
    raw_resp = tokenizer.decode(
        [t for t in response_ids[0].tolist() if t != 0], skip_special_tokens=False
    ) if response_ids is not None else ""
    clean_resp = _clean_response(raw_resp, tokenizer, response_ids)

    ntb = ret.non_tensor_batch

    reward_detail = _ntb_get(ntb, "reward_extra_info")
    if not isinstance(reward_detail, dict):
        reward_detail = {}

    extra_info = _ntb_get(ntb, "extra_info")
    extra_info = extra_info if isinstance(extra_info, dict) else {}

    reward_model_data = _ntb_get(ntb, "reward_model")
    gt = (reward_model_data.get("ground_truth", reward_model_data.get("gt_output", ""))
          if isinstance(reward_model_data, dict)
          else (_ntb_get(ntb, "ground_truth") or _ntb_get(ntb, "text")
                or extra_info.get("ground_truth") or extra_info.get("text") or ""))

    audio_path = (_ntb_get(ntb, "audio_path") or _ntb_get(ntb, "audio")
                  or _ntb_get(ntb, "wav_path")
                  or extra_info.get("audio_path") or extra_info.get("audio")
                  or _audio_from_raw_prompt(_ntb_get(ntb, "raw_prompt")) or "")

    row = {
        "sample_id": rollout_sample.sample_id,
        "text": str(gt) if gt else "",
        "raw_response": raw_resp,
        "response": clean_resp,
        "data_source": _ntb_get(ntb, "data_source", "") or extra_info.get("data_source", ""),
        "audio_path": audio_path,
    }
    row.update(reward_detail)
    return row


async def _consume_queue(mq_client: MessageQueueClient, tokenizer, output_path: str,
                         rollouter, save_freq: int = 100):
    """Consume from MessageQueue, decode, write a parquet part every save_freq samples."""
    all_results, examples_done, t0 = [], 0, time.time()
    n_err, n_ref = 0.0, 0.0

    while True:
        result = await mq_client.get_sample()
        if result is None:
            break
        sample_bytes, _ = result
        if sample_bytes is None:
            break

        row = _decode_rollout_sample(sample_bytes, tokenizer)
        all_results.append(row)
        examples_done += 1
        if row.get("n_ref") is not None:
            n_err += float(row.get("n_err") or 0.0)
            n_ref += float(row.get("n_ref") or 0.0)

        if len(all_results) >= save_freq:
            global_step = ray.get(rollouter.get_global_steps.remote())
            _flush_results(all_results, output_path, global_step)
            all_results = []
            ray.get(rollouter.save_checkpoint.remote(output_path))
            rate = examples_done / (time.time() - t0)
            print(f"[Consumer] {examples_done} samples | {rate:.1f} samples/s")

    if all_results:
        global_step = ray.get(rollouter.get_global_steps.remote())
        _flush_results(all_results, output_path, global_step)
        ray.get(rollouter.save_checkpoint.remote(output_path))

    overall_wer = (n_err / n_ref) if n_ref > 0 else float("nan")
    summary = {
        "n_egs": examples_done,
        "n_err": n_err,
        "n_ref": n_ref,
        "wer": overall_wer,
    }
    summary_path = os.path.join(output_path, "summary.json")
    with bf.BlobFile(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    rate = examples_done / max(time.time() - t0, 1)
    print(f"\nDone: {examples_done} samples | {rate:.1f} samples/s | output: {output_path}")
    print(f"[Consumer] Overall weighted WER: {overall_wer:.4f} "
          f"(n_err={n_err:.0f} / n_ref={n_ref:.0f}) | summary: {summary_path}")


def prepare_model(config):
    """Resolve the (possibly remote) model path locally and build the tokenizer.

    Mutates ``config.actor_rollout_ref.model.path`` in place to point at the
    resolved local directory and returns ``(local_model_path, tokenizer)``.
    """
    OmegaConf.resolve(config)
    _normalize_config(config)

    local_model_path = _resolve_model_path(config.actor_rollout_ref.model.path.rstrip("/"))
    OmegaConf.update(config, "actor_rollout_ref.model.path", local_model_path)
    tokenizer = hf_tokenizer(local_model_path,
                             trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False))
    return local_model_path, tokenizer


async def run_rollout_engine(config, tokenizer, local_model_path, consume_fn, *, tag="ASRRollout"):
    """Set up MessageQueue + FullyAsyncRollouter, run ``consume_fn``, then tear down.

    ``consume_fn`` is an async callable invoked as ``consume_fn(mq_client, rollouter)``
    while the rollouter is generating; it owns decoding/scoring/writing of results.
    """
    mq_actor = MessageQueue.remote(config=config,
                                   max_queue_size=config.async_training.get("max_queue_size", None))
    mq_client = MessageQueueClient(mq_actor)

    rollouter = FullyAsyncRollouter.remote(config=config, tokenizer=tokenizer,
                                           processor=None, local_model_path=local_model_path)
    await rollouter.init_workers.remote()
    await rollouter.set_message_queue_client.remote(mq_client)
    await rollouter.set_max_required_samples.remote()
    ray.get(rollouter.load_checkpoint.remote())

    rollouter_future = rollouter.fit.remote()
    await consume_fn(mq_client, rollouter)

    try:
        await asyncio.wrap_future(rollouter_future.future())
    except Exception as e:
        print(f"[{tag}] Rollouter: {e}")
    await mq_client.shutdown()


async def _run_asr_rollout(config):
    local_model_path, tokenizer = prepare_model(config)

    output_path = config.data.output_path
    makedirs(output_path, exist_ok=True)
    save_freq = config.data.get("save_freq", 100)

    async def consume(mq_client, rollouter):
        await _consume_queue(mq_client, tokenizer, output_path, rollouter, save_freq=save_freq)

    await run_rollout_engine(config, tokenizer, local_model_path, consume, tag="ASRRollout")


def init_ray(config):
    """Initialise Ray with the shared ASR runtime env (idempotent)."""
    if ray.is_initialized():
        return
    env_vars = EnvMgr().envs()
    ray_init_kwargs = OmegaConf.to_container(
        config.get("ray_kwargs", {}).get("ray_init", {}), resolve=True) or {}
    runtime_env = {
        "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN",
                     "VLLM_LOGGING_LEVEL": "WARN", "HF_HUB_OFFLINE": "1",
                     "VERL_USE_EXTERNAL_MODULES": "hf_qwen35_audio",
                     "VLLM_PLUGINS": "qwen35_audio",
                     "QWEN35_AUDIO_DISABLE_CUDNN": "1",
                     "PYTORCH_ALLOC_CONF": "expandable_segments:True", **env_vars},
        "excludes": [str(Path(__file__).parents[2] / ".git")],
        **ray_init_kwargs.pop("runtime_env", {}),
    }
    ray_init_kwargs["runtime_env"] = runtime_env
    ray.init(**ray_init_kwargs)


@hydra.main(config_path="config/rollout", config_name="rollout_oss_ls", version_base=None)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

    init_ray(config)
    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_asr_rollout(config))


if __name__ == "__main__":
    main()

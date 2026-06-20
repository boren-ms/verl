"""ASR generation using FullyAsyncRollouter + MessageQueue consumer.

Usage:
    python3 -m recipe.phimm.asr_rollout \
        --config-path=../../recipe/phimm/config/gen \
        --config-name=gen_oss_ls
"""

import asyncio
import hashlib
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
_REWARD_DETAIL_KEYS = (
    "wer", "n_ref", "n_err", "n_edge", "p_fmt", "p_lang",
    "p_bracket", "p_repeat", "p_kw_missing", "p_tail_hallu",
)
_DEBUG_KEYS_DUMPED = False


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
    global _DEBUG_KEYS_DUMPED
    rollout_sample = ray.cloudpickle.loads(sample_bytes)
    ret = rollout_sample.full_batch

    response_ids = ret.batch.get("responses")
    raw_resp = tokenizer.decode(
        [t for t in response_ids[0].tolist() if t != 0], skip_special_tokens=False
    ) if response_ids is not None else ""
    clean_resp = _clean_response(raw_resp, tokenizer, response_ids)

    ntb = ret.non_tensor_batch

    if not _DEBUG_KEYS_DUMPED:
        _DEBUG_KEYS_DUMPED = True
        print(f"[Decode] non_tensor_batch keys: {sorted(ntb.keys())}")
        print(f"[Decode] batch keys: {sorted(ret.batch.keys())}")
        print(f"[Decode] reward_extra_info[0]: {_ntb_get(ntb, 'reward_extra_info')}")

    reward_score = None
    rm_scores = ret.batch.get("rm_scores")
    if rm_scores is not None:
        scores_1d = rm_scores[0]
        nonzero = scores_1d.nonzero(as_tuple=True)[0]
        reward_score = scores_1d[nonzero[-1]].item() if len(nonzero) > 0 else scores_1d.sum().item()

    reward_detail = _ntb_get(ntb, "reward_extra_info")
    if not isinstance(reward_detail, dict):
        reward_detail = {}
    if reward_score is None and "score" in reward_detail:
        reward_score = reward_detail["score"]

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
        "reward": reward_score,
        "data_source": _ntb_get(ntb, "data_source", "") or extra_info.get("data_source", ""),
        "audio_path": audio_path,
    }
    for key in _REWARD_DETAIL_KEYS:
        if key in reward_detail:
            row[key] = reward_detail[key]
        else:
            val = _ntb_get(ntb, key)
            if val is not None:
                row[key] = val
    return row


async def _consume_queue(mq_client: MessageQueueClient, tokenizer, output_path: str,
                         rollouter, save_freq: int = 100):
    """Consume from MessageQueue, decode, write a parquet part every save_freq samples."""
    all_results, examples_done, part, t0 = [], 0, 0, time.time()

    while True:
        result = await mq_client.get_sample()
        if result is None:
            break
        sample_bytes, _ = result
        if sample_bytes is None:
            break

        all_results.append(_decode_rollout_sample(sample_bytes, tokenizer))
        examples_done += 1

        if len(all_results) >= save_freq:
            _flush_results(all_results, output_path, part)
            all_results = []
            part += 1
            ray.get(rollouter.save_checkpoint.remote(output_path))
            rate = examples_done / (time.time() - t0)
            print(f"[Consumer] {examples_done} samples | {rate:.1f} samples/s")

    if all_results:
        _flush_results(all_results, output_path, part)
        ray.get(rollouter.save_checkpoint.remote(output_path))

    rate = examples_done / max(time.time() - t0, 1)
    print(f"\nDone: {examples_done} samples | {rate:.1f} samples/s | output: {output_path}")


async def _run_asr_rollout(config):
    OmegaConf.resolve(config)
    _normalize_config(config)

    local_model_path = _resolve_model_path(config.actor_rollout_ref.model.path.rstrip("/"))
    OmegaConf.update(config, "actor_rollout_ref.model.path", local_model_path)
    tokenizer = hf_tokenizer(local_model_path,
                             trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False))

    output_path = config.data.output_path
    makedirs(output_path, exist_ok=True)
    save_freq = config.data.get("save_freq", 100)

    mq_actor = MessageQueue.remote(config=config,
                                   max_queue_size=config.async_training.get("max_queue_size", 1000))
    mq_client = MessageQueueClient(mq_actor)

    rollouter = FullyAsyncRollouter.remote(config=config, tokenizer=tokenizer,
                                           processor=None, local_model_path=local_model_path)
    await rollouter.init_workers.remote()
    await rollouter.set_message_queue_client.remote(mq_client)
    await rollouter.set_max_required_samples.remote()
    ray.get(rollouter.load_checkpoint.remote())

    rollouter_future = rollouter.fit.remote()
    await _consume_queue(mq_client, tokenizer, output_path, rollouter, save_freq=save_freq)

    try:
        await asyncio.wrap_future(rollouter_future.future())
    except Exception as e:
        print(f"[ASRRollout] Rollouter: {e}")
    await mq_client.shutdown()


@hydra.main(config_path="config/gen", config_name="gen_oss_ls", version_base=None)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))

    env_mgr = EnvMgr()
    env_vars = env_mgr.envs()

    if not ray.is_initialized():
        ray_init_kwargs = OmegaConf.to_container(
            config.get("ray_kwargs", {}).get("ray_init", {}), resolve=True) or {}
        runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN",
                         "VLLM_LOGGING_LEVEL": "WARN", "HF_HUB_OFFLINE": "1",
                         "PYTORCH_ALLOC_CONF": "expandable_segments:True", **env_vars},
            "excludes": [str(Path(__file__).parents[2] / ".git")],
            **ray_init_kwargs.pop("runtime_env", {}),
        }
        ray_init_kwargs["runtime_env"] = runtime_env
        ray.init(**ray_init_kwargs)

    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_asr_rollout(config))


if __name__ == "__main__":
    main()

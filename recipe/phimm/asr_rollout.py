"""ASR generation using FullyAsyncRollouter + MessageQueue consumer.

Usage:
    python3 -m recipe.phimm.asr_rollout \
        --config-path=../../recipe/phimm/config/gen \
        --config-name=gen_oss_ls
"""

import asyncio
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path
from pprint import pprint

import blobfile as bf
import hydra
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


def _flush_results(results: list[dict], output_path: str, step: int):
    path = os.path.join(output_path, f"step_{step}.parquet")
    _write_parquet(pd.DataFrame(results), path)
    print(f"[Consumer] Wrote {path} ({len(results)} rows)")


def _decode_rollout_sample(sample_bytes, tokenizer) -> dict:
    rollout_sample = ray.cloudpickle.loads(sample_bytes)
    ret = rollout_sample.full_batch

    response_ids = ret.batch.get("responses")
    raw_resp = tokenizer.decode(
        [t for t in response_ids[0].tolist() if t != 0], skip_special_tokens=False
    ) if response_ids is not None else ""

    reward_score = None
    rm_scores = ret.batch.get("rm_scores")
    if rm_scores is not None:
        scores_1d = rm_scores[0]
        nonzero = scores_1d.nonzero(as_tuple=True)[0]
        reward_score = scores_1d[nonzero[-1]].item() if len(nonzero) > 0 else scores_1d.sum().item()

    ntb = ret.non_tensor_batch
    reward_model_data = ntb.get("reward_model", [None])[0] if "reward_model" in ntb else None
    gt = (reward_model_data.get("ground_truth", reward_model_data.get("gt_output", ""))
          if isinstance(reward_model_data, dict)
          else (ntb.get("text", [""])[0] if "text" in ntb else ""))

    row = {
        "text": str(gt) if gt else "",
        "response": raw_resp,
        "reward": reward_score,
        "data_source": ntb.get("data_source", [""])[0] if "data_source" in ntb else "",
        "audio_path": ntb.get("audio_path", [""])[0] if "audio_path" in ntb else "",
        "sample_id": rollout_sample.sample_id,
    }
    extra_info = ntb.get("extra_info", [None])[0] if "extra_info" in ntb else None
    if isinstance(extra_info, dict):
        row.update({k: v for k, v in extra_info.items() if k not in row})
    return row


async def _consume_queue(mq_client: MessageQueueClient, tokenizer, output_path: str,
                         rollouter, save_freq: int = 100):
    """Consume from MessageQueue, decode, write parquet every save_freq steps."""
    all_results, examples_done, t0 = [], 0, time.time()

    while True:
        result = await mq_client.get_sample()
        if result is None:
            break
        sample_bytes, _ = result
        if sample_bytes is None:
            break

        all_results.append(_decode_rollout_sample(sample_bytes, tokenizer))
        examples_done += 1

        global_steps = ray.get(rollouter.__ray_call__.remote(lambda self: self.global_steps))
        if global_steps % save_freq == 0 and all_results:
            _flush_results(all_results, output_path, global_steps)
            all_results = []
            ray.get(rollouter.save_checkpoint.remote(output_path))
            rate = examples_done / (time.time() - t0)
            print(f"[Consumer] step {global_steps} | {examples_done} samples | {rate:.1f} s/s")

    if all_results:
        global_steps = ray.get(rollouter.__ray_call__.remote(lambda self: self.global_steps))
        _flush_results(all_results, output_path, global_steps)
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

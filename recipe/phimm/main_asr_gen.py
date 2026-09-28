# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate ASR responses and configurable measurements as resumable JSONL splits.
"""

import json
import logging
import math
import os
import re

import hydra
import numpy as np
import ray
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'
import uuid
from pprint import pprint
from datasets import Dataset, Sequence, Value
from omegaconf import OmegaConf
from torch.utils.data import Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_processor, hf_tokenizer
from recipe.phimm.data.rl_dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.utils.fs import copy_to_local
import blobfile as bf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from pathlib import Path
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.reward.asr_eval import openasr_eval
from recipe.phimm.reward.asr_response import get_hyp_text


def _load_generation_scoring(config):
    reward_config = config.get("custom_reward_function") or {}
    reward_kwargs = reward_config.get("reward_kwargs", {})
    if OmegaConf.is_config(reward_kwargs):
        reward_kwargs = OmegaConf.to_container(reward_kwargs, resolve=True)
    score_fn = get_custom_reward_fn(config) or openasr_eval
    return score_fn, reward_kwargs


def _part_index(path: str) -> int | None:
    match = re.fullmatch(r"part-(\d+)\.jsonl", os.path.basename(path))
    return int(match.group(1)) if match else None


def _jsonl_num_rows(path: str) -> int:
    count = 0
    with bf.BlobFile(path, "r") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            count += 1
    if count == 0:
        raise ValueError(f"Cannot resume from an empty output split: {path}")
    return count


def _resume_state_from_output(output_dir: str, total_egs: int, batch_size: int, enabled: bool) -> tuple[int, int]:
    if any(bf.glob(f"{output_dir}/part-*.parquet")):
        raise ValueError(
            f"Only JSONL output is supported; {output_dir} contains Parquet splits. Use a new output_path."
        )
    if not enabled:
        return 0, 0

    parts = {}
    for path in bf.glob(f"{output_dir}/part-*.jsonl"):
        idx = _part_index(path)
        if idx is not None:
            if idx in parts:
                raise ValueError(f"Duplicate output split index {idx}: {parts[idx]} and {path}")
            parts[idx] = path
    if not parts:
        return 0, 0

    existing_egs = 0
    for expected_idx, (idx, path) in enumerate(sorted(parts.items())):
        if idx != expected_idx:
            raise ValueError(
                f"Missing output split {expected_idx} before {path}; use a complete prefix or a new output_path"
            )
        existing_egs += _jsonl_num_rows(path)
    if existing_egs > total_egs:
        raise ValueError(f"Saved output has {existing_egs} examples, exceeding the current dataset size {total_egs}")
    if existing_egs < total_egs and existing_egs % batch_size != 0:
        raise ValueError(
            f"Cannot resume from {existing_egs} saved examples because it is not aligned to batch_size={batch_size}"
        )
    print(f"Resuming generation from {existing_egs}/{total_egs} saved examples across {len(parts)} parts.")
    return existing_egs, len(parts)


def get_env_vars():
    env_vars = EnvMgr().envs()
    required_envs = ["DATA_PATH"]
    assert all(k in env_vars for k in required_envs), (
        f"Missing env vars: {[k for k in required_envs if k not in env_vars]}"
    )
    return env_vars


def cwd():
    return Path(__file__).parents[2]


@hydra.main(config_path="config/gen", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    env_vars = get_env_vars()
    # Register the custom Qwen3.5-Audio HF model in every Ray process via
    # ``import verl`` so tokenizer/processor/config load with
    # ``trust_remote_code=False`` (no dependency on per-checkpoint remote *.py files).
    env_vars.setdefault("VERL_USE_EXTERNAL_MODULES", "hf_qwen35_audio")
    print(f"Cluster Env: {env_vars}")
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_PLUGINS": "qwen35_audio",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                **env_vars,
            },
            "excludes": [str(cwd() / ".git")],
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


def log_examples(ds, num_examine=1):
    if "wer" not in ds.column_names and "n_err" in ds.column_names and "n_ref" in ds.column_names:
        ds = ds.map(lambda r: {"wer": r["n_err"] / max(r["n_ref"], 1)})
    sort_ds = ds.sort("wer", reverse=True)
    audio_key = "audio_chunk" if "audio_chunk" in ds.column_names else "audio_path"

    for i in range(min(num_examine, len(sort_ds))):
        print(f"--- Example {i + 1} ---")
        print(f"WER: {sort_ds[i]['wer']:.2%}")
        print("Ref:", sort_ds[i]["text"])
        print("Hyp:", sort_ds[i]["raw_response"])
        if audio_key in sort_ds.column_names:
            print("Audio:", sort_ds[i][audio_key])


@ray.remote(num_cpus=1)
def main_task(config):
    # Register the `eval` resolver inside the remote worker process so
    # interpolations like `${eval:...}` resolve when Ray spins up a fresh worker.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    score_fn, reward_kwargs = _load_generation_scoring(config)
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.model.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    print(f"{trust_remote_code=}")
    assert tokenizer is not None, "Please specify a valid tokenizer"
    assert processor is not None, "Please specify a valid processor"
    num_examine = config.data.get("eval_num_examine", 1)
    ds_conf = config.data.get("gen_data", config.data.get("train_data", config.data.get("val_data", None)))
    assert ds_conf is not None, "Please specify data.gen_data or data.train_data or data.val_data in the config"
    dataset = RLHFDataset(ds_conf, tokenizer, config.data, processor, is_train=False)
    print(f"Loaded RLHFDataset with {len(dataset)} samples.")

    output_dir = config.data.get("output_path", None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")
    split_size = config.data.get("output_split_size", 1000)
    total_egs = len(dataset)
    batch_size = config.data.batch_size
    resume_from_output = config.data.get("resume_from_output", True)
    val_shuffle = config.data.get("validation_shuffle", False)
    if resume_from_output and val_shuffle:
        raise ValueError("data.resume_from_output requires data.validation_shuffle=False")
    left_egs, split_idx = _resume_state_from_output(output_dir, total_egs, batch_size, resume_from_output)
    start_batch_idx = left_egs // batch_size
    total_batches = math.ceil(total_egs / batch_size)
    if left_egs >= total_egs:
        print(f"All {total_egs} samples already exist in {output_dir}; nothing to do.")
        print("All Done")
        return
    if left_egs > 0:
        dataset = Subset(dataset, range(left_egs, total_egs))

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.get("num_workers", 0),
        prefetch_factor=config.data.get("prefetch_factor", 2) if config.data.get("num_workers", 0) > 0 else None,
        pin_memory=config.data.get("pin_memory", False),
        persistent_workers=config.data.get("persistent_workers", False) and config.data.get("num_workers", 0) > 0,
        shuffle=val_shuffle,
        drop_last=False,
        collate_fn=default_collate_fn,
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    # need this to create fused worker group
    worker_dict_cls = create_colocated_worker_cls(class_dict={"rollout": ray_cls_with_init})
    process_on_nodes = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, max_colocate_count=1)
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=config.trainer.device,
    )
    wg = wg_dict.spawn(prefix_set=["rollout"])["rollout"]
    wg.init_model()

    tn_err = 0
    tn_ref = 0
    processed_egs = 0

    def write_data(batches, idx):
        num_rows = sum(len(batch) for batch in batches)
        if not num_rows:
            return 0
        bf.makedirs(output_dir)
        split_path = f"{output_dir}/part-{idx:03d}.jsonl"
        with bf.BlobFile(split_path, "w") as f:
            # Preserve nullable integers and float precision without pandas coercion.
            for batch in batches:
                for record in batch:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return num_rows

    import threading
    import queue as _queue

    prefetch_depth = int(config.data.get("prefetch_depth", 2))
    prep_queue: _queue.Queue = _queue.Queue(maxsize=max(1, prefetch_depth))
    post_queue: _queue.Queue = _queue.Queue(maxsize=max(1, prefetch_depth))
    _SENTINEL = object()
    stopped = threading.Event()
    errors = _queue.SimpleQueue()

    def put_item(target_queue, item):
        while not stopped.is_set():
            try:
                target_queue.put(item, timeout=0.1)
                return True
            except _queue.Full:
                continue
        return False

    def get_item(source_queue):
        while not stopped.is_set():
            try:
                return source_queue.get(timeout=0.1)
            except _queue.Empty:
                continue
        return _SENTINEL

    def run_stage(stage):
        try:
            stage()
        except BaseException as exc:
            errors.put(exc)
            stopped.set()
            logging.getLogger(__name__).exception("Generation pipeline stage %s failed", stage.__name__)

    def producer():
        for batch_idx, batch_dict in enumerate(dataloader):
            if stopped.is_set():
                return
            global_batch_idx = start_batch_idx + batch_idx
            prompts = [msg[0]["content"] for msg in batch_dict["prompt"]]
            texts = [x["ground_truth"] for x in batch_dict["reward_model"]]
            n_egs = len(texts)
            audio_paths = batch_dict.get("audio_path", [None] * n_egs)
            audio_chunks = batch_dict.get("audio_chunk", [None] * n_egs)
            extras = batch_dict.get("extra_info", [{}] * n_egs)
            results = []
            for i in range(n_egs):
                r = {"prompt": prompts[i], "text": texts[i],
                     "audio_path": audio_paths[i], "audio_chunk": audio_chunks[i]}
                if extras[i]:
                    r.update(extras[i])
                results.append(r)
            data = DataProto.from_single_dict(batch_dict)
            if "uid" not in data.non_tensor_batch:
                data.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(data.batch))], dtype=object
                )
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)
            if not put_item(prep_queue, (global_batch_idx, n_egs, data_padded, pad_size, results, extras)):
                return
        put_item(prep_queue, _SENTINEL)

    def consumer():
        nonlocal tn_err, tn_ref, processed_egs, left_egs, split_idx
        local_batches = []
        buffered_rows = 0
        while not stopped.is_set():
            item = get_item(post_queue)
            if item is _SENTINEL:
                break
            output, results, extras = item
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                sample_reward_kwargs = {**reward_kwargs, "extra_info": extras[i] or {}}
                score = score_fn(response_str, results[i]["text"], **sample_reward_kwargs)
                if not isinstance(score, dict) or not {"n_err", "n_ref"} <= score.keys():
                    raise ValueError("Generation reward function must return a dict containing n_err and n_ref.")
                score = {key: value for key, value in score.items() if key not in {"n_edge", "edge_wer"}}
                score["response"] = get_hyp_text(response_str, version=reward_kwargs.get("version"))
                score["raw_response"] = response_str
                results[i].update(score)
            tn_err += sum(r["n_err"] for r in results)
            tn_ref += sum(r["n_ref"] for r in results)
            processed_egs += len(results)
            b_ds = Dataset.from_list(results)
            if "keywords" in b_ds.features:
                features = b_ds.features
                features["keywords"] = Sequence(Value("string"))
                if features != b_ds.features:
                    b_ds = b_ds.cast(features)
            log_examples(b_ds, num_examine=num_examine)
            local_batches.append(b_ds)
            buffered_rows += len(b_ds)
            if stopped.is_set():
                return
            if buffered_rows >= split_size:
                left_egs += write_data(local_batches, split_idx)
                split_idx += 1
                local_batches = []
                buffered_rows = 0
        if not stopped.is_set() and local_batches:
            left_egs += write_data(local_batches, split_idx)
            split_idx += 1

    def generate():
        while not stopped.is_set():
            item = get_item(prep_queue)
            if item is _SENTINEL:
                break
            global_batch_idx, n_egs, data_padded, pad_size, results, extras = item
            print(f"\n(Batch {global_batch_idx + 1}/{total_batches}) Generating {n_egs} samples")
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)
            if not put_item(post_queue, (output, results, extras)):
                return
            pbar.update(1)
        put_item(post_queue, _SENTINEL)

    threads = [
        threading.Thread(target=run_stage, args=(producer,), name="asr-generation-producer", daemon=True),
        threading.Thread(target=run_stage, args=(consumer,), name="asr-generation-consumer", daemon=True),
    ]
    pbar = tqdm(total=total_batches, initial=start_batch_idx)
    try:
        for thread in threads:
            thread.start()
        run_stage(generate)
    except BaseException:
        stopped.set()
        raise
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        pbar.close()
    if not errors.empty():
        raise errors.get()

    print(f"Overall wer: {tn_err / max(tn_ref, 1):.2%} [{tn_err}/{tn_ref}] on {processed_egs} generated samples")
    print(f"Saved {left_egs}/{total_egs} [{left_egs / total_egs:.2%}] samples.")
    print(f"Saved {split_idx} splits to {output_dir}")
    print("All Done")


if __name__ == "__main__":
    main()

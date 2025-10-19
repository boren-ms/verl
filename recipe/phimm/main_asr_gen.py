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
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'
import uuid
from pprint import pprint
from collections import defaultdict
from datasets import Dataset, concatenate_datasets
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_processor, hf_tokenizer
from recipe.phimm.data.rl_dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.utils.fs import copy_to_local, copy_to_remote
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from pathlib import Path
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.reward.asr_bias import compute_wers, WordError, sum_wers


def get_env_vars():
    env_vars = EnvMgr().envs()
    required_envs = ["DATA_PATH"]
    assert all(k in env_vars for k in required_envs), (
        f"Missing env vars: {[k for k in required_envs if k not in env_vars]}"
    )
    return env_vars


def cwd():
    return Path(__file__).parents[2]


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    env_vars = get_env_vars()
    print(f"Cluster Env: {env_vars}")
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
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


def filter_ds(ds, **kwargs):
    wer_range = kwargs.get("wer_range", None)
    err_range = kwargs.get("err_range", None)

    def filter_fn(x):
        if wer_range is not None and not (wer_range[0] <= x["wer"] <= wer_range[1]):
            return False
        if err_range is not None and not (err_range[0] <= x["n_err"] <= err_range[1]):
            return False
        return True

    return ds.filter(filter_fn)


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    # breakpoint()
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
    dataset = RLHFDataset(ds_conf, tokenizer, config.data, processor)
    print(f"Loaded RLHFDataset with {len(dataset)} samples.")
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.get("num_workers", 0),
        shuffle=config.data.get("validation_shuffle", False),
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

    remote_output_dir = config.data.get("remote_output_path", None)
    local_output_dir = Path(config.data.local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    split_size = config.data.get("output_split_size", 1000)
    total_egs = len(dataset)
    total_splits = (total_egs + split_size - 1) // split_size

    split_idx = 0
    total_batches = len(dataloader)
    wer_kwargs = config.data.get("wer_kwargs", {})

    ds_list = []
    total_wer = WordError()
    left_egs = 0

    err_range = config.data.get("err_range", None)
    wer_range = config.data.get("wer_range", None)
    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
        results = defaultdict(list)
        results["prompt"].extend([msg[0]["content"] for msg in batch_dict["prompt"]])  # get user prompt
        results["text"].extend([x["ground_truth"] for x in batch_dict["reward_model"]])

        for value in batch_dict.get("extra_info", []):
            for k, v in value.items():
                results[k].append(v)

        n_egs = len(results["text"])
        results["audio_path"].extend(batch_dict.get("audio_path", [None] * n_egs))
        results["audio_chunk"].extend(batch_dict.get("audio_chunk", [None] * n_egs))
        data = DataProto.from_single_dict(batch_dict)
        if "uid" not in data.non_tensor_batch:
            data.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(data.batch))], dtype=object)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)
        print(f"\n(Batch {batch_idx + 1}/{total_batches}) Generating {n_egs} samples")
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)
        responses = []
        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses.append(response_str)

        results["response"].extend(responses)
        wers = compute_wers(results["text"], results["response"], **wer_kwargs)
        results["wer"] = [x.wer for x in wers]
        results["n_err"] = [x.n_err for x in wers]
        batch_wer = sum_wers(wers)
        print(f"Batch WER: {batch_wer.wer:.2%} [{batch_wer.n_err}/{batch_wer.n_ref}] on {n_egs} samples")

        total_wer += batch_wer
        batch_ds = Dataset.from_dict(results)
        sort_ds = batch_ds.sort("wer", reverse=True)
        audio_key = "audio_chunk" if "audio_chunk" in batch_dict else "audio_path"
        for i in range(num_examine):
            print(f"--- Example {i + 1} ---")
            print(f"WER: {sort_ds[i]['wer']:.2%}")
            print("Ref:", sort_ds[i]["text"])
            print("Hyp:", sort_ds[i]["response"])
            print("Audio:", sort_ds[i][audio_key])

        batch_ds = filter_ds(batch_ds, wer_range=wer_range, err_range=err_range)
        print(f"Batch Filtering: {n_egs} => {len(batch_ds)} samples.")

        if len(batch_ds) > 0:
            ds_list.append(batch_ds)
            left_egs += len(batch_ds)

        if sum(len(ds) for ds in ds_list) >= split_size or (batch_idx + 1) == total_batches:  # save a split
            if len(ds_list) == 0:
                print(f"No samples to save for split {split_idx}, skip.")
                split_idx += 1
                continue
            split_path = local_output_dir / f"part-{split_idx:03d}-{total_splits:03d}.parquet"
            print(f"Writting results to {split_path}")
            split_ds = concatenate_datasets(ds_list)
            split_ds.to_parquet(str(split_path))
            if remote_output_dir is not None:
                print(f"Copying {split_path.name} to remote: {remote_output_dir}")
                copy_to_remote(split_path, remote_output_dir)
            split_idx += 1
            ds_list = []

    print(f"Overall WER: {total_wer.wer:.2%} [{total_wer.n_err}/{total_wer.n_ref}] on {total_egs} samples")
    print(f"Filtering with: {wer_range=}, {err_range=}")
    print(f"Keep {left_egs}/{total_egs} [{left_egs / total_egs:.2%}] samples.")
    print(f"Saved {split_idx}/{total_splits} splits to {local_output_dir}")
    if remote_output_dir is not None:
        print(f"Saved {split_idx}/{total_splits} splits to {remote_output_dir}")
    print("All Done")


if __name__ == "__main__":
    main()

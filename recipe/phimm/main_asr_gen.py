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
from verl.utils.fs import copy_to_local
import blobfile as bf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from pathlib import Path
from recipe.phimm.utils.env import EnvMgr
from recipe.phimm.reward.asr_edge import eval_score
from recipe.phimm.utils.shared import parse_asr_response


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
    edge_wer_range = kwargs.get("edge_wer_range", None)
    bad_fmt = kwargs.get("bad_fmt", False)
    bad_lang = kwargs.get("bad_lang", False)
    

    def filter_fn(x):
        if wer_range is not None and not (wer_range[0] <= x["wer"] <= wer_range[1]):
            return False
        if edge_wer_range is not None and not (edge_wer_range[0] <= x["edge_wer"] <= edge_wer_range[1]):
            return False
        if bad_fmt and x.get("p_fmt", 0.0):
            return False
        if bad_lang and x.get("p_lang", 0.0):
            return False
        return True

    return ds.filter(filter_fn)


def log_examples(ds, num_examine=1):
    sort_ds = ds.sort("wer", reverse=True)
    audio_key = "audio_chunk" if "audio_chunk" in ds.column_names else "audio_path"
    
    for i in range(min(num_examine, len(sort_ds))):
        print(f"--- Example {i + 1} ---")
        print(f"WER: {sort_ds[i]['wer']:.2%}  edge_wer: {sort_ds[i]['edge_wer']:.2%}")
        print("Ref:", sort_ds[i]["text"])
        print("Hyp:", sort_ds[i]["raw_response"])
        if audio_key in sort_ds.column_names:
            print("Audio:", sort_ds[i][audio_key])


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

    output_dir = config.data.get("output_path", None)
    assert output_dir is not None, "Please specify data.output_path"
    output_dir = output_dir.rstrip("/")
    split_size = config.data.get("output_split_size", 1000)
    total_egs = len(dataset)
    left_egs = 0
    
    total_batches = len(dataloader)
    wer_kwargs = OmegaConf.to_container(config.data.get("wer_kwargs", {}), resolve=True) or {}

    batches = []
    tn_err = 0
    tn_ref = 0
    tn_edge = 0
    split_idx = 0
    
    def write_data(batches, idx):
        batches = [b for b in batches if len(b) > 0]
        if not batches:
            return 0
        split_ds = concatenate_datasets(batches)
        bf.makedirs(output_dir)
        split_path = f"{output_dir}/part-{idx:03d}.parquet"
        with bf.BlobFile(split_path, "wb") as f:
            split_ds.to_parquet(f)
        return len(split_ds)


    wer_range = config.data.get("wer_range", None)
    edge_wer_range = config.data.get("edge_wer_range", None)
    bad_format = config.data.get("bad_format", False)
    for batch_idx, batch_dict in tqdm(enumerate(dataloader)):
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
            data.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(data.batch))], dtype=object)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)
        print(f"\n(Batch {batch_idx + 1}/{total_batches}) Generating {n_egs} samples")
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)
        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            score = eval_score(response_str, results[i]["text"], **wer_kwargs)
            score["response"] = parse_asr_response(response_str).get("text") # the parsed ASR text from the response
            score["raw_response"] = response_str
            results[i].update(score)
        bn_err = sum(r["n_err"] for r in results)
        bn_ref = sum(r["n_ref"] for r in results)
        bn_edge = sum(r["n_edge"] for r in results)
        
        print(
            f"\n(Batch {batch_idx + 1}/{total_batches}) "
            f"Batch wer: {bn_err / max(bn_ref, 1):.2%} [{bn_err}/{bn_ref}] "
            f"Batch edge_wer={bn_edge / max(bn_ref, 1):.2%} "
        )

        tn_err += bn_err
        tn_ref += bn_ref
        tn_edge += bn_edge
        b_ds = Dataset.from_list(results)
        log_examples(b_ds, num_examine=num_examine)

        b_ds = filter_ds(b_ds, wer_range=wer_range, edge_wer_range=edge_wer_range, bad_format=bad_format)
        print(f"Filtering with {wer_range=}, {edge_wer_range=}, {bad_format=}")
        print(f"Batch Filtering: {n_egs} => {len(b_ds)} samples.")
        batches.append(b_ds)
        
        if sum(len(ds) for ds in batches) >= split_size:
            left_egs += write_data(batches, split_idx)
            split_idx += 1
            batches = []

    left_egs += write_data(batches, split_idx)

    print(
        f"Overall wer: {tn_err / max(tn_ref, 1):.2%} [{tn_err}/{tn_ref}] "
        f"edge_wer={tn_edge / max(tn_ref, 1):.2%} on {total_egs} samples"
    )
    print(f"Filtering with: {wer_range=}, {edge_wer_range=}, {bad_format=}")
    print(f"Keep {left_egs}/{total_egs} [{left_egs / total_egs:.2%}] samples.")
    print(f"Saved {split_idx} splits to {output_dir}")
    print("All Done")


if __name__ == "__main__":
    main()

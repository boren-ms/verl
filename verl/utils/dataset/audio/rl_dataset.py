import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.audio.dataset import create_audio_dataset, get_num_proc
from verl.utils.dataset.audio.audio_utils import load_audio

logger = logging.getLogger(__name__)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return x


def remove_empty_tensors(batch: dict) -> dict:
    keys_to_remove = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.numel() == 0:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        batch.pop(key, None)
    return batch


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_confs: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_confs, Sequence):
            data_confs = [data_confs]
        self.data_confs = data_confs
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_audio_dur = config.get("max_audio_dur", 40)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.num_proc = get_num_proc(config.get("num_proc", "auto"))
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.ds = self.load_datasets()

    def load_datasets(self):
        data_sets = [create_audio_dataset(**data_conf) for data_conf in self.data_confs]
        ds = datasets.concatenate_datasets(data_sets)
        return ds

    def resume_dataset_state(self):
        pass

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.ds[i]
        messages = row_dict[self.prompt_key]

        raw_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )

        audios = [load_audio(row_dict, self.max_audio_dur)]

        row_dict["multi_modal_data"] = {"audio": [(to_numpy(audio), fs) for (audio, fs) in audios]}
        model_inputs = self.processor(text=[raw_prompt], audios=audios, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if self.return_multi_modal_inputs:
            inputs_dict = dict(model_inputs)
            inputs_dict = remove_empty_tensors(inputs_dict)
            if "input_audio_embeds" in inputs_dict:
                inputs_dict["input_audio_embeds"] = inputs_dict["input_audio_embeds"].squeeze(0)
            if "input_audio_embeds" in inputs_dict and inputs_dict.get("audio_attention_mask", None) is None:
                inputs_dict["audio_attention_mask"] = torch.ones(
                    inputs_dict["input_audio_embeds"].shape[:-1],
                    dtype=torch.long,
                )
            row_dict["multi_modal_inputs"] = inputs_dict

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        return self.__dict__.copy()


def main(config_path, tokenizer_path, data_files=None):
    """
    Example main function to instantiate RLHFDataset from CLI.
    """
    from omegaconf import OmegaConf

    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=True)
    processor = hf_processor(tokenizer_path, trust_remote_code=True)
    config = (
        OmegaConf.load(config_path)["data"]
        if config_path
        else {
            "audio_key": "audio_path",
            "asr_dataset": True,
            # "filter_overlong_prompts": False,
            "max_prompt_length": 1024,
            "max_response_length": 2048,
            "train_files": data_files,
        }
    )  # must have for phi4mm
    data_files = config.get("train_files", None)
    data_conf = config.get("train_data", None)
    # data_conf = config.get("val_data", None)
    dataset = RLHFDataset(data_conf, tokenizer, config, processor)
    from torchdata.stateful_dataloader import StatefulDataLoader

    loader = StatefulDataLoader(dataset=dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
    for batch in loader:
        print(batch)
        break


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    # data_files = "/home/boren/data/parquet/ls_sc1k_fn1_h100.parquet"
    # data_conf = "/home/boren/data/parquet/data_conf.yaml"
    tokenizer_path = "/home/boren/data/ckp/hf_models/Phi-4-multimodal-instruct"
    tokenizer_path = "/home/boren/data/ckp/hf_models/phi4_mm_bias_merged"
    data_yaml = "/home/boren/verl/recipe/dapo/config/data/test_audio.yaml"
    main(data_yaml, tokenizer_path)

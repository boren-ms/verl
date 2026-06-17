import logging
from collections.abc import Mapping, Sequence
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from recipe.phimm.data.dataset import create_audio_dataset, get_num_proc
from recipe.phimm.utils.audio import load_audio
from verl.utils.model import compute_position_id_with_mask

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


def _to_data_confs(data_files, config, is_train=True):
    """Convert data_files (new API) to data_confs list for create_audio_dataset."""
    # Check if config has explicit train_data / val_data
    key = "train_data" if is_train else "val_data"
    confs = config.get(key, None)
    if confs is not None:
        if isinstance(confs, (DictConfig, ListConfig)):
            confs = OmegaConf.to_container(confs, resolve=True)
        if isinstance(confs, list) and len(confs) > 0 and isinstance(confs[0], Mapping):
            return confs
    # If data_files is already a list of dicts, use directly
    if data_files and isinstance(data_files, (list, ListConfig)):
        if isinstance(data_files, ListConfig):
            data_files = OmegaConf.to_container(data_files, resolve=True)
        if len(data_files) > 0 and isinstance(data_files[0], Mapping):
            return data_files
    # Convert file paths to data_confs
    if data_files is None:
        return []
    if isinstance(data_files, str):
        data_files = [data_files]
    if isinstance(data_files, ListConfig):
        data_files = OmegaConf.to_container(data_files, resolve=True)
    confs = []
    for fp in data_files:
        if isinstance(fp, str):
            ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else "jsonl"
            if ext in ("parquet", "pq"):
                confs.append({"dataset_name": "parquet", "path": fp})
            else:
                confs.append({"dataset_name": "jsonl", "jsonl_paths": fp})
        elif isinstance(fp, Mapping):
            confs.append(dict(fp))
    return confs


class RLHFDataset(Dataset):
    """
    Audio RLHF dataset for phimm ASR training.
    Compatible with both old (data_confs) and new (data_files) verl APIs.
    """

    def __init__(
        self,
        data_files=None,
        tokenizer: PreTrainedTokenizer = None,
        config: DictConfig = None,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        # Legacy kwargs
        data_confs=None,
        is_training: bool = False,
    ):
        if data_confs is not None and data_files is None:
            data_files = data_confs
        self.data_confs = _to_data_confs(data_files, config, is_train=is_training)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.is_training = is_training
        self.max_samples = max_samples
        self.interleave_ds = config.get("interleave_ds", {})
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_audio_dur = config.get("max_audio_dur", 40)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "right2")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.num_proc = get_num_proc(config.get("num_proc", "auto"))
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.ds = self.load_datasets()

    def load_datasets(self):
        data_sets = [create_audio_dataset(**data_conf) for data_conf in self.data_confs]
        if self.is_training and len(data_sets) > 1:
            ds = datasets.interleave_datasets(data_sets, **self.interleave_ds)
        else:
            ds = datasets.concatenate_datasets(data_sets)
        if self.max_samples > 0 and len(ds) > self.max_samples:
            ds = ds.select(range(self.max_samples))
        return ds
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

        # Use processor.apply_chat_template if available; fall back to tokenizer
        _chat_obj = self.processor if getattr(self.processor, "chat_template", None) else self.tokenizer
        raw_prompt = _chat_obj.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        # print(f"raw_prompt after chat template [{i}]: {raw_prompt}  with messages: {messages}")
        # Remove empty thinking block injected by Qwen3.5 chat template
        raw_prompt = raw_prompt.replace("<think>\n\n</think>\n\n", "")
        # print(f"raw_prompt before prefix [{i}]: {raw_prompt}")
        extra_info = row_dict.get("extra_info") or {}
        prefix = extra_info.get("prefix", "") or ""
        raw_prompt = f"{raw_prompt}{prefix}"
        # print(f"raw_prompt after prefix [{i}]: {raw_prompt}")
        # print(f"raw_prompt[{i}]: {raw_prompt}", i, raw_prompt)

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
            if "audio_attention_mask" in inputs_dict and inputs_dict["audio_attention_mask"] is not None:
                inputs_dict["audio_attention_mask"] = inputs_dict["audio_attention_mask"].squeeze(0)
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
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        row_dict["extra_info"] = extra_info
        index = extra_info.get("index", 0)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        need_tools_kwargs = extra_info.get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        return self.__dict__.copy()

    @classmethod
    def _extract_audio_info(cls, messages: list) -> list | None:
        """Extract audio URLs/paths from messages for the agent loop."""
        audios = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                if "audio" in item:
                    audios.append(item["audio"])
                elif "audio_url" in item:
                    audios.append(item["audio_url"])
                else:
                    audios.append({k: v for k, v in item.items() if k != "type"})
        return audios or None

    @classmethod
    async def process_multi_modal_info(cls, messages: list, image_patch_size=14, config=None):
        """Extract and load audio data from messages for the agent loop."""
        audio_refs = cls._extract_audio_info(messages)
        if not audio_refs:
            return None, None, None

        loaded_audios = []
        for ref in audio_refs:
            if isinstance(ref, str):
                try:
                    audio, sr = load_audio({"audio_path": ref}, max_dur=40)
                    wav = audio.numpy() if hasattr(audio, 'numpy') else audio
                    loaded_audios.append((wav, sr))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to load audio {ref}: {e}")
            elif isinstance(ref, tuple):
                loaded_audios.append(ref)
            else:
                loaded_audios.append(ref)

        return None, None, loaded_audios or None


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
            "max_prompt_length": 30,
            "max_response_length": 2048,
            "train_files": data_files,
        }
    )  # must have for phi4mm
    data_files = config.get("train_files", None)
    data_conf = config.get("train_data", None)
    # data_conf = config.get("val_data", None)
    dataset = RLHFDataset(data_conf, tokenizer, config, processor, True)
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

    loader = StatefulDataLoader(dataset=dataset, batch_size=2, num_workers=0, collate_fn=default_collate_fn)
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
    # data_yaml = "/home/boren/verl/recipe/phimm/config/data_local.yaml"
    # data_yaml = "/home/boren/verl/recipe/phimm/config/data_local_parquet.yaml"
    data_yaml = "/home/boren/verl/recipe/phimm/config/data_local_2.yaml"

    main(data_yaml, tokenizer_path)

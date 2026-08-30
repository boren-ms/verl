import logging
from collections.abc import Mapping
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from recipe.phimm.data.dataset import create_audio_dataset, get_num_proc
from recipe.phimm.utils.audio import load_audio, set_chunk_load_mode
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

try:
    from hf_qwen35_audio.processing_qwen3_5_audio import AUDIO_PAD_TOKEN_ID
except Exception:  # pragma: no cover - fallback if plugin import path differs
    AUDIO_PAD_TOKEN_ID = 248076


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
    """Convert the trainer dataset API into audio dataset configurations."""
    key = "train_data" if is_train else "val_data"
    confs = config.get(key, None)
    if confs is not None:
        if isinstance(confs, (DictConfig, ListConfig)):
            confs = OmegaConf.to_container(confs, resolve=True)
        if isinstance(confs, list) and confs and isinstance(confs[0], Mapping):
            return confs

    if isinstance(data_files, ListConfig):
        data_files = OmegaConf.to_container(data_files, resolve=True)
    if data_files and isinstance(data_files, list) and isinstance(data_files[0], Mapping):
        return data_files

    if data_files is None:
        return []
    if isinstance(data_files, str):
        data_files = [data_files]

    confs = []
    for data_file in data_files:
        if isinstance(data_file, str):
            extension = data_file.rsplit(".", 1)[-1].lower() if "." in data_file else "jsonl"
            if extension in ("parquet", "pq"):
                confs.append({"dataset_name": "parquet", "path": data_file})
            else:
                confs.append({"dataset_name": "jsonl", "jsonl_paths": data_file})
        elif isinstance(data_file, Mapping):
            confs.append(dict(data_file))
    return confs


def _promote_null_feature(feat):
    """Promote ``null``-typed leaves to ``string`` while preserving structure.

    HuggingFace infers ``Value('null')`` / ``List(Value('null'))`` for columns
    (e.g. ``extra_info.keywords``) that are entirely empty/``None`` in a given
    data source. Such a source cannot be concatenated/interleaved with another
    source where the same column carries real strings. This promotes only the
    ``null`` leaves (bare ``null`` values and lists whose element is ``null``),
    recursing into struct/``Features`` dicts to reach nested fields, and leaves
    list-of-struct fields such as the chat ``prompt`` untouched (never turning a
    ``list<struct>`` into a struct-of-lists).
    """
    from datasets import Features, Sequence, Value

    if isinstance(feat, Features):
        return Features({k: _promote_null_feature(v) for k, v in feat.items()})
    if isinstance(feat, dict):
        return {k: _promote_null_feature(v) for k, v in feat.items()}
    if isinstance(feat, Value) and feat.dtype == "null":
        return Value("string")
    if isinstance(feat, Sequence) and isinstance(feat.feature, Value) and feat.feature.dtype == "null":
        return Sequence(Value("string"), length=feat.length)
    if hasattr(datasets, "List") and isinstance(feat, datasets.List) and isinstance(feat.feature, Value) and feat.feature.dtype == "null":
        return datasets.List(Value("string"))
    if isinstance(feat, list) and len(feat) == 1 and isinstance(feat[0], Value) and feat[0].dtype == "null":
        return [Value("string")]
    return feat


def _align_null_features(data_sets):
    """Cast every dataset so ``null``-typed columns become ``string``-typed.

    Ensures sources with all-empty optional fields (e.g. ``keywords``) share a
    schema with sources that populate them, so ``concatenate_datasets`` /
    ``interleave_datasets`` can align features.
    """
    aligned = []
    for ds in data_sets:
        promoted = _promote_null_feature(ds.features)
        if promoted != ds.features:
            ds = ds.cast(promoted)
        aligned.append(ds)
    return aligned


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
        data_files=None,
        tokenizer: PreTrainedTokenizer = None,
        config: DictConfig = None,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        is_train: bool = True,
        data_confs=None,
    ):
        if data_confs is not None and data_files is None:
            data_files = data_confs
        self.data_confs = _to_data_confs(data_files, config, is_train=is_train)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.is_training = is_train
        self.max_samples = max_samples
        self.use_interleave = config.get("use_interleave", True)
        self.interleave_ds = config.get("interleave_ds", {})
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_audio_dur = config.get("max_audio_dur", 40)
        if chunk_load_mode := config.get("chunk_load_mode", None):
            set_chunk_load_mode(chunk_load_mode)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.truncation = config.get("truncation", "right2")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.num_proc = get_num_proc(config.get("num_proc", "auto"))
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.ds = self.load_datasets()

    def load_datasets(self):
        data_sets = [create_audio_dataset(**data_conf) for data_conf in self.data_confs]
        data_sets = _align_null_features(data_sets)
        if self.is_training and self.use_interleave and len(data_sets) > 1:
            logger.info(
                "Interleaving %s datasets with parameters: %s",
                len(data_sets),
                self.interleave_ds,
            )
            ds = datasets.interleave_datasets(data_sets, **self.interleave_ds)
        else:
            logger.info("Concatenating %s datasets", len(data_sets))
            ds = datasets.concatenate_datasets(data_sets)
        if self.max_samples > 0 and len(ds) > self.max_samples:
            ds = ds.select(range(self.max_samples))
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
        extra_info = row_dict.get("extra_info") or {}
        prefix = extra_info.get("prefix", "") or ""
        raw_prompt = f"{raw_prompt}{prefix}"

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

        # Prompt truncation (e.g. "right2") can drop the tail of the audio-placeholder
        # block for overlong prompts, leaving fewer AUDIO_PAD tokens in input_ids than
        # the stored audio_embed_sizes describe. That mismatch trips the assertion
        # `audio_embed_sizes.sum() == len(positions)` in the model's audio embedding
        # forward. Reconcile audio_embed_sizes with the surviving placeholder tokens so
        # each sample stays internally consistent; the encoder still produces the full
        # audio_set_tensor and only its aligned prefix frames are consumed.
        if self.return_multi_modal_inputs:
            mmi = row_dict.get("multi_modal_inputs")
            if mmi is not None and "audio_embed_sizes" in mmi:
                audio_embed_sizes = mmi["audio_embed_sizes"]
                n_pad = int((row_dict["input_ids"] == AUDIO_PAD_TOKEN_ID).sum().item())
                if int(audio_embed_sizes.sum().item()) != n_pad:
                    remaining = n_pad
                    new_sizes = []
                    for sz in audio_embed_sizes.tolist():
                        take = min(int(sz), remaining)
                        new_sizes.append(take)
                        remaining -= take
                    logger.warning(
                        "Reconciled audio_embed_sizes %s -> %s for sample %s (data source: %s) "
                        "after prompt truncation dropped %s audio placeholder token(s).",
                        audio_embed_sizes.tolist(),
                        new_sizes,
                        i,
                        row_dict.get("data_source"),
                        int(audio_embed_sizes.sum().item()) - n_pad,
                    )
                    mmi["audio_embed_sizes"] = torch.tensor(new_sizes, dtype=audio_embed_sizes.dtype)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # The async agent loop consumes the structured messages and passes the
        # rendered text to vLLM so model-specific audio placeholders are retained.
        row_dict["raw_prompt"] = messages
        row_dict["full_prompt_text"] = raw_prompt

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

"""Turn the Base3a JSONL audio mix into teacher-forced Qwen audio examples."""

import torch
from recipe.phimm.agentic.dataset import (
    _align_source_features,
    _combine_datasets,
    _prepare_data_confs,
)
from recipe.phimm.data.dataset import create_audio_dataset
from recipe.phimm.reward.asr_response import parse_jsonl_response
from recipe.phimm.utils.audio import load_audio, set_chunk_load_mode
from torch.utils.data import Dataset


class ASRSFTDataset(Dataset):
    """Keep source transforms and train budgets; supervise formatted references only.

    Validation uses individual source rows rather than RL parent grouping. A list
    of audio paths within a row is loaded as one waveform with one full target.
    """

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if processor is None:
            raise ValueError("ASR SFT requires an audio processor; set model.external_lib=hf_qwen35_audio")
        if config.pad_mode != "no_padding" or config.truncation != "error":
            raise ValueError("ASR SFT requires data.pad_mode=no_padding and data.truncation=error")
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_train = parquet_files == config.train_files
        set_chunk_load_mode(config.get("chunk_load_mode") or "indexed")
        sources = _prepare_data_confs(config, is_train=self.is_train)
        data_sets = [create_audio_dataset(**conf) for conf in sources]
        # RL-only diagnostics have different nested schemas across sources.
        # SFT consumes the prompt, audio, source name, and formatted reference.
        data_sets = [ds.remove_columns("extra_info") if "extra_info" in ds.column_names else ds for ds in data_sets]
        self.ds = _combine_datasets(_align_source_features(data_sets), config, is_train=self.is_train)
        if max_samples > 0:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = dict(self.ds[index])
        target = row["reward_model"].get("gt_output")
        if not isinstance(target, str) or any(segment["language"] is None for segment in parse_jsonl_response(target)):
            raise ValueError(f"ASR SFT row {index} requires valid JSONL in reward_model.gt_output")
        # Never crop a waveform while retaining the complete supervised target.
        row["asr_full_reference"] = True
        paths = row.get("audio_path")
        if isinstance(paths, list):
            if not paths or any(not isinstance(path, str) or not path for path in paths):
                raise ValueError(f"ASR SFT row {index} has an invalid audio_path list")
            row["audio_path"] = ";".join(paths)
            row.pop("audio", None)
        max_dur = self.config.max_audio_dur if self.is_train else self.config.val_max_audio_dur
        audio = load_audio(row, max_dur=max_dur)
        chat = self.processor if getattr(self.processor, "chat_template", None) else self.tokenizer
        kwargs = dict(self.config.get("apply_chat_template_kwargs", {}))
        prompt = row[self.config.prompt_key]
        prefix = chat.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False, **kwargs)
        full = chat.apply_chat_template(
            [*prompt, {"role": "assistant", "content": target}],
            add_generation_prompt=False,
            tokenize=False,
            **kwargs,
        )
        if not full.startswith(prefix):
            raise ValueError("Chat template changes the prompt when adding the assistant target")
        # Process audio once. Append the assistant turn using the same tokenizer,
        # retaining chat-template end-of-turn tokens in the supervised suffix.
        inputs = dict(self.processor(text=[prefix], audios=[audio], return_tensors="pt"))
        prompt_ids = inputs.pop("input_ids")[0]
        inputs.pop("attention_mask", None)
        response_ids = self.tokenizer.encode(full[len(prefix) :], add_special_tokens=False)
        if not response_ids:
            raise ValueError(f"ASR SFT row {index} has no assistant tokens")
        response_ids = torch.tensor(response_ids, dtype=prompt_ids.dtype)
        input_ids = torch.cat((prompt_ids, response_ids))
        if input_ids.numel() > self.config.max_length:
            raise ValueError(
                f"ASR SFT row {index} has {input_ids.numel()} tokens; "
                f"increase data.max_length={self.config.max_length} and data.max_token_len_per_gpu"
            )
        loss_mask = torch.cat((torch.zeros_like(prompt_ids), torch.ones_like(response_ids)))
        return {
            "input_ids": input_ids,
            "position_ids": torch.arange(input_ids.numel(), dtype=torch.long),
            "loss_mask": loss_mask,
            "multi_modal_inputs": inputs,
        }

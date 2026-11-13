"""Expose the unchanged ASR data mix through main's raw-prompt dataset contract."""

from collections import defaultdict

from recipe.phimm.data.rl_dataset import RLHFDataset


def group_validation_rows(rows):
    """Match LongAudioGroupedRewardManager's source/parent and segment ordering."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        info = row.get("extra_info") or {}
        parent = info.get("parent_audio_path") or info.get("audio_path")
        parent = str(parent).split("#", 1)[0] if parent else f"__row_{index}__"
        groups[(row["data_source"], parent)].append(index)

    def order(index):
        start = (rows[index].get("extra_info") or {}).get("seg_start")
        try:
            return (float(start) if start is not None else 0.0, index)
        except (TypeError, ValueError):
            return (0.0, index)

    return [sorted(indices, key=order) for indices in groups.values()]


class ASRDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        is_train = config.get("train_data") is not None and data_files == config.train_files
        data_confs = config.train_data if is_train else config.val_data
        super().__init__(data_confs, tokenizer, config, processor, is_train=is_train)
        if max_samples > 0:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        self.groups = None if is_train else group_validation_rows(self.ds)

    def __len__(self):
        return len(self.ds) if self.groups is None else len(self.groups)

    def __getitem__(self, index):
        indices = [index] if self.groups is None else self.groups[index]
        rows = [self.ds[i] for i in indices]
        row = dict(rows[0])
        row["raw_prompt"] = row[self.prompt_key]
        row["index"] = index
        row["asr_segments"] = rows
        row["asr_validate"] = self.groups is not None
        row["agent_name"] = "asr"
        return row

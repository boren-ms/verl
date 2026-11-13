"""Expose the unchanged ASR data mix through main's raw-prompt dataset contract."""

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import datasets
from torch.utils.data import Dataset

from recipe.phimm.data.dataset import create_audio_dataset
from recipe.phimm.utils.audio import set_chunk_load_mode

logger = logging.getLogger(__name__)


def _flatten_data_confs(data_confs):
    if isinstance(data_confs, str | bytes) or not isinstance(data_confs, Sequence):
        return [data_confs]

    flattened = []
    for data_conf in data_confs:
        if isinstance(data_conf, Sequence) and not isinstance(data_conf, str | bytes):
            flattened.extend(_flatten_data_confs(data_conf))
        else:
            flattened.append(data_conf)
    return flattened


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
    if (
        hasattr(datasets, "List")
        and isinstance(feat, datasets.List)
        and isinstance(feat.feature, Value)
        and feat.feature.dtype == "null"
    ):
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


def _collect_datasources(data_sets: Sequence[datasets.Dataset]) -> dict[str, datasets.Dataset]:
    """Return one combined dataset per datasource in first-seen source order."""
    grouped = {}
    for index, ds in enumerate(data_sets):
        if not len(ds):
            logger.warning("Skipping empty training dataset %s; its data_source cannot be inferred.", index)
            continue
        source = ds[0].get("data_source", "default")
        grouped.setdefault(source, []).append(ds)
    return {
        name: datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0] for name, parts in grouped.items()
    }


def mix_datasets(
    data_sets: Sequence[datasets.Dataset],
    mix: Sequence[Mapping],
    train_batch_size: int | None = None,
    seed: int = 42,
) -> datasets.Dataset:
    """Concatenate ordered groups with independent source budgets and local shuffling."""
    if isinstance(mix, str | bytes) or not isinstance(mix, Sequence) or not mix:
        raise ValueError("data.mix must be a non-empty list of groups.")
    sources = _collect_datasources(data_sets)
    groups = []
    for group_index, group in enumerate(mix):
        context = f"data.mix[{group_index}]"
        if not isinstance(group, Mapping) or set(group) - {"datasource", "shuffle"}:
            raise ValueError(f"{context} must contain datasource and optional shuffle only.")
        entries = group.get("datasource")
        if isinstance(entries, str | bytes) or not isinstance(entries, Sequence) or not entries:
            raise ValueError(f"{context}.datasource must be a non-empty list.")
        if not isinstance(group.get("shuffle", False), bool):
            raise ValueError(f"{context}.shuffle must be a boolean.")
        parts = []
        for source_index, options in enumerate(entries):
            source_context = f"{context}.datasource[{source_index}]"
            if not isinstance(options, Mapping) or set(options) - {
                "name",
                "sample",
                "batch",
                "epoch",
                "shuffle",
                "range",
                "shuffle_seed",
            }:
                raise ValueError(
                    f"{source_context} accepts name, sample, batch, epoch, shuffle, range, and shuffle_seed only."
                )
            name = options.get("name")
            if not isinstance(name, str) or name not in sources:
                raise ValueError(f"{source_context}.name has no non-empty training datasource: {name!r}")
            if not isinstance(options.get("shuffle", False), bool):
                raise ValueError(f"{source_context}.shuffle must be a boolean.")
            budgets = [key for key in ("sample", "batch", "epoch") if options.get(key) is not None]
            if len(budgets) > 1:
                raise ValueError(f"{source_context} must use only one of sample, batch, or epoch.")
            ds = sources[name]
            size = len(ds)
            source_range = options.get("range")
            if source_range is not None:
                if (
                    isinstance(source_range, str | bytes)
                    or not isinstance(source_range, Sequence)
                    or len(source_range) != 2
                    or any(isinstance(bound, bool) or not isinstance(bound, int) for bound in source_range)
                ):
                    raise ValueError(f"{source_context}.range must be [start, end] with integer bounds.")
                start, end = source_range
                if not 0 <= start <= end <= size:
                    raise ValueError(
                        f"{source_context}.range must satisfy 0 <= start <= end <= datasource size ({size})."
                    )
                ds = ds.select(range(start, end) if start < end else [])
                size = len(ds)
            source_seed = options.get("shuffle_seed", seed + group_index + source_index)
            if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
                raise ValueError(f"{source_context}.shuffle_seed must be a non-negative integer.")
            count = size
            if budgets:
                key = budgets[0]
                value = options[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{source_context}.{key} must be a finite non-negative number.")
                if key == "batch":
                    if (
                        isinstance(train_batch_size, bool)
                        or not isinstance(train_batch_size, int)
                        or train_batch_size <= 0
                    ):
                        raise ValueError("data.train_batch_size must be a positive integer for mix batch budgets.")
                    count = value * train_batch_size
                elif key == "epoch":
                    count = size * value
                else:
                    count = value
                if not math.isfinite(count):
                    raise ValueError(f"{source_context}.{key} produces a non-finite row count.")
                count = round(count)
            logger.info("Mix group %s datasource %s: %s => %s rows", group_index, name, size, count)
            if not count:
                continue
            if not size:
                raise ValueError(f"{source_context}.range leaves no rows for a positive sample budget.")
            if options.get("shuffle", False):
                ds = ds.shuffle(seed=source_seed)
            full_passes, remainder = divmod(count, size)
            repeated = [ds] * full_passes
            if remainder:
                repeated.append(ds.select(range(remainder)))
            parts.append(datasets.concatenate_datasets(repeated) if len(repeated) > 1 else repeated[0])
        if not parts:
            logger.warning("Skipping mix group %s: all datasource budgets select zero rows.", group_index)
            continue
        ds = datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if group.get("shuffle", False):
            ds = ds.shuffle(seed=seed + group_index)
        groups.append(ds)
    if not groups:
        raise ValueError("No samples remain after applying data.mix budgets.")
    return datasets.concatenate_datasets(groups) if len(groups) > 1 else groups[0]


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


class ASRDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        is_train = config.get("train_data") is not None and data_files == config.train_files
        data_confs = config.train_data if is_train else config.val_data
        self.prompt_key = config.get("prompt_key", "prompt")
        set_chunk_load_mode(config.get("chunk_load_mode") or "indexed")
        data_sets = _align_null_features(
            [create_audio_dataset(**{**conf, "version": config.version}) for conf in _flatten_data_confs(data_confs)]
        )
        if is_train and config.get("mix") is not None:
            if config.get("shuffle", False):
                raise ValueError("Set data.shuffle=false when using data.mix to preserve ordered mix groups.")
            # Preserve the source's effective seed: data.seed=null (or 0) uses 42.
            self.ds = mix_datasets(data_sets, config.mix, config.train_batch_size, seed=config.get("seed") or 42)
        else:
            self.ds = datasets.concatenate_datasets(data_sets)
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

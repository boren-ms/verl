"""Bounded-buffer packing helpers used by offline ASR preparation."""

import math
import random
from collections.abc import Sequence as SequenceABC
from numbers import Real

from datasets import Dataset, Sequence, Value
from recipe.phimm.utils.audio import get_audio_duration


def _merge_rows(rows, indices, durations, text_separator):
    result = dict(rows[0])
    # Some raw manifests retain a producer's formatted reference. Task
    # formatting must regenerate it from all merged text/components.
    result.pop("gt_output", None)
    paths = [row.get("audio_path") or row.get("audio_file") or row.get("audio_chunk") for row in rows]
    source = ";".join(paths)
    for field in ("audio_path", "audio_file", "audio_chunk"):
        if field == "audio_path" or field in result:
            result[field] = source
    result["text"] = text_separator.join(row["text"] for row in rows)
    result["audio_length"] = math.fsum(durations)
    if "audio_duration_seconds" in result:
        result["audio_duration_seconds"] = result["audio_length"]
    if "id" in result:
        result["id"] = ";".join(
            str(row["id"] if row.get("id") is not None else index) for row, index in zip(rows, indices, strict=True)
        )
    if "keywords" in result:
        result["keywords"] = list(dict.fromkeys(word for row in rows for word in (row.get("keywords") or [])))
    if "components" in result:
        result["components"] = [
            component
            for row in rows
            for component in (row.get("components") or [{"language": row.get("language"), "text": row["text"]}])
        ]
    result["random_merge_count"] = len(rows)
    result["random_merge_source_indices"] = indices
    return result


def _iter_random_merge(
    ds, *, max_audio_length, buffer_size, seed, text_separator, group_by, merge_fn=_merge_rows, merge_count=None
):
    """Consume each row once, retaining at most buffer_size candidate rows.

    Pick a random seed utterance, then the longest compatible candidate that
    still fits (random ties). Refill after each selection. This is a greedy
    bounded-lookahead packing heuristic, not a global subset-sum optimum.
    """
    rng = random.Random(seed)
    stream = iter(enumerate(ds))
    buffer = []

    def refill():
        while len(buffer) < buffer_size:
            try:
                index, row = next(stream)
            except StopIteration:
                break
            source = row.get("audio_path") or row.get("audio_file") or row.get("audio_chunk")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"random_merge row {index} requires a non-empty audio source string")
            if ";" in source or source.count("#") > 1:
                raise ValueError("random_merge requires individual sources before merging or nested segmentation")
            if row.get("audio") is not None:
                raise ValueError("random_merge requires path-based audio, not inline waveforms")
            if "reward_model" in row:
                raise ValueError("Offline packing requires raw rows before add_task_info and verl_format")
            if not isinstance(row.get("text"), str):
                raise ValueError(f"random_merge row {index} requires text after rename_fields")
            duration = row.get("audio_length")
            if duration is None:
                duration = get_audio_duration(source)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, Real)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(f"random_merge row {index} audio_length must be positive finite seconds")
            if duration >= max_audio_length:
                raise ValueError(
                    f"random_merge row {index} duration {duration} must be below {max_audio_length} seconds"
                )
            buffer.append((index, row, float(duration)))

    refill()
    while buffer:
        first = buffer.pop(rng.randrange(len(buffer)))
        selected = [first]
        target_count = rng.randint(*merge_count) if merge_count is not None else None
        key = tuple(first[1].get(field) for field in group_by)
        refill()
        while target_count is None or len(selected) < target_count:
            candidates = [
                i
                for i, (_, row, duration) in enumerate(buffer)
                if tuple(row.get(field) for field in group_by) == key
                and math.fsum([*(item[2] for item in selected), duration]) < max_audio_length
            ]
            if not candidates:
                break
            longest = max(buffer[i][2] for i in candidates)
            index = rng.choice([i for i in candidates if buffer[i][2] == longest])
            selected.append(buffer.pop(index))
            refill()
        indices, rows, durations = zip(*selected, strict=True)
        yield merge_fn(rows, list(indices), durations, text_separator)


def _validate_options(max_audio_length, buffer_size, seed, text_separator, group_by):
    if (
        isinstance(max_audio_length, bool)
        or not isinstance(max_audio_length, Real)
        or not math.isfinite(max_audio_length)
        or max_audio_length <= 0
    ):
        raise ValueError("random_merge max_audio_length must be positive finite seconds")
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ValueError("random_merge buffer_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("random_merge seed must be an integer")
    if not isinstance(text_separator, str):
        raise ValueError("random_merge text_separator must be a string")
    if (
        isinstance(group_by, str)
        or not isinstance(group_by, SequenceABC)
        or any(not isinstance(k, str) for k in group_by)
    ):
        raise ValueError("random_merge group_by must be a list of field names")


def random_merge(
    ds,
    *,
    max_audio_length=600,
    buffer_size=100,
    seed=42,
    text_separator=" ",
    group_by=("data_source", "language", "prompt"),
    enabled=True,
):
    """Pack whole utterances below a configurable duration without decoding.

    Audio references use ``;``; text uses ``text_separator`` in the same order.
    This standalone utility expects raw rows after path/text normalization.
    Training datasets load prepared offline manifests. Output is written incrementally to
    an Arrow dataset; only candidate rows are kept in the bounded buffer.
    Leftovers are emitted, never dropped. An individual over-limit utterance
    raises instead of silently truncating audio against its full reference.
    Other metadata comes from the first constituent; source indices identify
    all constituents within this input dataset.
    """
    if not enabled:
        return ds
    _validate_options(max_audio_length, buffer_size, seed, text_separator, group_by)
    features = ds.features.copy()
    features.pop("gt_output", None)
    features.update(
        audio_path=Value("string"),
        audio_length=Value("float64"),
        random_merge_count=Value("int64"),
        random_merge_source_indices=Sequence(Value("int64")),
    )
    # Raw manifests may keep unused aliases as all-null columns. Merging
    # populates every present alias, so its output schema must accept strings.
    for field in ("audio_file", "audio_chunk"):
        if field in features:
            features[field] = Value("string")
    if "audio_duration_seconds" in features:
        features["audio_duration_seconds"] = Value("float64")
    if "id" in features:
        features["id"] = Value("string")
    if "keywords" in features:
        features["keywords"] = Sequence(Value("string"))
    if not len(ds):
        return Dataset.from_dict({key: [] for key in features}, features=features)
    return Dataset.from_generator(
        _iter_random_merge,
        features=features,
        gen_kwargs={
            "ds": ds,
            "max_audio_length": max_audio_length,
            "buffer_size": buffer_size,
            "seed": seed,
            "text_separator": text_separator,
            "group_by": tuple(group_by),
        },
    )

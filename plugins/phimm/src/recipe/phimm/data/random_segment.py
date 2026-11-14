"""Reproducible, text-preserving random partitions of complete ASR recordings."""

import math
import random
from collections.abc import Sequence
from numbers import Real

from recipe.phimm.utils.audio import get_audio_duration


def _validate_range(value, name, *, integers=False):
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, int if integers else Real)
            or not math.isfinite(item)
            or item <= 0
            for item in value
        )
        or value[0] > value[1]
    ):
        kind = "integers" if integers else "durations in seconds"
        raise ValueError(f"random_segment {name} requires [minimum, maximum] with positive finite {kind}")
    return value


def _segment_fractions(rng, count, minimum):
    weights = [rng.expovariate(1.0) for _ in range(count)]
    total = sum(weights)
    return [minimum + (1 - count * minimum) * weight / total for weight in weights]


def _length_fractions(rng, duration, lower, upper):
    fractions = []
    remaining = duration
    while remaining > upper:
        length = rng.uniform(lower, upper)
        fractions.append(length / duration)
        remaining -= length
    fractions.append(remaining / duration)
    return fractions


def random_segment(
    ds,
    *,
    segment_count=None,
    segment_len=None,
    min_segment_fraction=None,
    seed=42,
    enabled=True,
):
    """Replace audio_path with ordered percentage ranges without decoding audio.

    Each input row remains one sample with its original full reference. Random
    partitions are fixed by seed and row index when the dataset is prepared.
    ``segment_len`` defaults to None; if neither range is provided, use
    ``segment_count=(1, 2)``. Configure at most one non-null inclusive range:
    ``segment_count`` for counts, or ``segment_len`` for seconds. Only duration mode inspects
    ``audio_length`` or falls back to header IO. Its final remainder may be
    shorter than the lower duration bound, preserving the complete recording.
    ``min_segment_fraction`` defaults to 0.1 in count mode and is not applied
    in duration mode.
    """
    if not enabled:
        return ds
    if segment_count is None and segment_len is None:
        segment_count = (1, 2)
    elif segment_count is not None and segment_len is not None:
        raise ValueError("random_segment requires exactly one non-null range: segment_count or segment_len")
    if segment_count is not None:
        lower, upper = _validate_range(segment_count, "segment_count", integers=True)
        if min_segment_fraction is None:
            min_segment_fraction = 0.1
        if (
            isinstance(min_segment_fraction, bool)
            or not isinstance(min_segment_fraction, Real)
            or not math.isfinite(min_segment_fraction)
            or not 0 <= min_segment_fraction <= 1 / upper
        ):
            raise ValueError("random_segment min_segment_fraction must be between 0 and 1/segment_count[1]")
    else:
        lower, upper = _validate_range(segment_len, "segment_len")
        if min_segment_fraction not in (None, 0):
            raise ValueError("random_segment min_segment_fraction applies only to segment_count mode")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("random_segment seed must be an integer")

    def partition(row, index):
        source = row.get("audio_path")
        if not isinstance(source, str) or not source:
            raise ValueError("random_segment requires one non-empty audio_path string per input row")
        rng = random.Random(seed + index)
        if segment_count is not None:
            count = rng.randint(lower, upper)
            fractions = _segment_fractions(rng, count, min_segment_fraction)
        else:
            duration = row.get("audio_length")
            if duration is None:
                duration = get_audio_duration(source)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, Real)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError("random_segment audio_length must be a positive finite duration in seconds")
            fractions = _length_fractions(rng, duration, lower, upper)
        bounds = [0.0]
        for fraction in fractions[:-1]:
            bounds.append(bounds[-1] + fraction * 100)
        bounds.append(100.0)
        # A full inner selector makes the outer range apply to the entire
        # concatenation, including when its last file has no selector yet.
        segment_source = f"{source}#0%:100%" if ";" in source else source
        return {
            "audio_path": [
                f"{segment_source}#{start:.12g}%:{end:.12g}%" for start, end in zip(bounds, bounds[1:], strict=False)
            ]
        }

    return ds.map(partition, with_indices=True, desc="Random audio segmentation")

"""Randomly pack complete utterances into ordered audio lists for ASR context."""

import json
import math
from collections.abc import Sequence as SequenceABC
from functools import partial

from datasets import Dataset, Features, Sequence, Value
from recipe.phimm.data.prompts import resolve_language
from recipe.phimm.data.random_merge import _iter_random_merge, _validate_options

AUDIO_FIELDS = ("audio_path", "audio_file", "audio_chunk")
TEXT_FIELDS = ("text", "display", "display_ne_tagged", "lexical", "ref", "hyp")
DURATION_FIELDS = ("audio_length", "audio_duration_seconds", "duration")
DROP_FIELDS = ("gt_output", "audio", "sr")
COMPONENT = {"language": Value("string"), "target_language": Value("string"), "text": Value("string")}


def _languages(value):
    return resolve_language(value or "English").split()


def _merge_rows(rows, indices, durations, text_separator, *, default_language):
    paths = [row.get("audio_path") or row.get("audio_file") or row.get("audio_chunk") for row in rows]
    total = math.fsum(durations)
    merged = {}
    for key in set().union(*(row.keys() for row in rows)):
        values = [row.get(key) for row in rows]
        if key in DROP_FIELDS:
            continue
        if key in AUDIO_FIELDS:
            merged[key] = paths
        elif key in TEXT_FIELDS:
            if any(value is not None and not isinstance(value, str) for value in values):
                raise ValueError(f"random_merge requires string {key}")
            merged[key] = text_separator.join(value or "" for value in values)
        elif key in DURATION_FIELDS:
            merged[key] = total
        elif key == "id":
            merged[key] = "|".join(
                str(value if value is not None else i) for value, i in zip(values, indices, strict=True)
            )
        elif key in ("data_source", "prompt"):
            if any(value != values[0] for value in values):
                raise ValueError(f"random_merge cannot combine incompatible {key}")
            merged[key] = values[0]
        elif key not in ("keywords", "language", "components"):
            # Arbitrary producer metadata can have heterogeneous nested schemas.
            # An aligned JSON list preserves every value without Arrow coercion.
            merged[key] = [json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values]

    keywords, components, source_languages = [], [], []
    for row in rows:
        words = row.get("keywords") or []
        if not isinstance(words, list) or any(not isinstance(word, str) for word in words):
            raise ValueError("random_merge keywords must be a list of strings")
        keywords.extend(words)
        supplied_parts = row.get("components") or []
        component_languages = " ".join(part.get("language") or "" for part in supplied_parts if isinstance(part, dict))
        language = " ".join(_languages(row.get("language") or component_languages or default_language))
        parts = supplied_parts or [{"language": language, "text": row["text"]}]
        row_languages = []
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise ValueError("random_merge components require language-labelled string text")
            source = " ".join(_languages(part.get("language") or language))
            target = " ".join(_languages(part.get("target_language") or source))
            row_languages.extend(source.split())
            components.append({"language": source, "target_language": target, "text": part["text"]})
        source_languages.append(" ".join(dict.fromkeys([*language.split(), *row_languages])))
    merged.update(
        audio_path=paths,
        audio_length=total,
        text=text_separator.join(row["text"] for row in rows),
        keywords=list(dict.fromkeys(keywords)),
        language=" ".join(dict.fromkeys(lang for value in source_languages for lang in value.split())),
        components=components,
        random_merge_count=len(rows),
        random_merge_source_indices=indices,
        source_audio_lengths=list(durations),
        source_languages=source_languages,
        source_metadata_json=[json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows],
    )
    return merged


def _features(ds):
    features = {}
    for key, feature in ds.features.items():
        if key in DROP_FIELDS:
            continue
        if key in AUDIO_FIELDS or key == "keywords":
            features[key] = Sequence(Value("string"))
        elif key in TEXT_FIELDS or key in ("id", "language"):
            features[key] = Value("string")
        elif key in DURATION_FIELDS:
            features[key] = Value("float64")
        elif key == "components":
            continue
        elif key in ("data_source", "prompt"):
            features[key] = feature
        else:
            features[key] = Sequence(Value("string"))
    features.update(
        audio_path=Sequence(Value("string")),
        audio_length=Value("float64"),
        text=Value("string"),
        keywords=Sequence(Value("string")),
        language=Value("string"),
        components=[COMPONENT],
        random_merge_count=Value("int64"),
        random_merge_source_indices=Sequence(Value("int64")),
        source_audio_lengths=Sequence(Value("float64")),
        source_languages=Sequence(Value("string")),
        source_metadata_json=Sequence(Value("string")),
    )
    return Features(features)


def _generate(ds, **kwargs):
    default_language = kwargs.pop("default_language")
    yield from _iter_random_merge(ds, merge_fn=partial(_merge_rows, default_language=default_language), **kwargs)


def random_merge(
    ds,
    *,
    max_audio_length=180,
    buffer_size=100,
    seed=42,
    text_separator=" ",
    group_by=("data_source", "prompt"),
    default_language=None,
    merge_count=None,
    enabled=True,
):
    """Pack raw rows into audio lists before task/reference formatting.

    A random seed row starts each pack, then compatible whole utterances fill
    the strict duration cap using bounded lookahead. Each input is used once;
    leftovers stay as singleton lists. No waveform is concatenated or decoded.
    Languages and keywords are stable unions; components and metadata retain
    audio order. Other producer fields become lists of JSON-encoded values.
    """
    if not enabled:
        return ds
    _validate_options(max_audio_length, buffer_size, seed, text_separator, group_by)
    if merge_count is not None:
        if (
            not isinstance(merge_count, SequenceABC)
            or isinstance(merge_count, str)
            or len(merge_count) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) for n in merge_count)
            or not 1 <= merge_count[0] <= merge_count[1]
        ):
            raise ValueError("random_merge merge_count must be [min_count, max_count] with positive ordered integers")
        merge_count = tuple(merge_count)
    if "reward_model" in ds.column_names:
        raise ValueError("Random Merge must run in pre_process before add_task_info and verl_format")
    features = _features(ds)
    if not len(ds):
        return Dataset.from_dict({key: [] for key in features}, features=features)
    return Dataset.from_generator(
        _generate,
        features=features,
        gen_kwargs={
            "ds": ds,
            "max_audio_length": max_audio_length,
            "buffer_size": buffer_size,
            "seed": seed,
            "text_separator": text_separator,
            "group_by": tuple(group_by),
            "default_language": default_language,
            "merge_count": merge_count,
        },
    )

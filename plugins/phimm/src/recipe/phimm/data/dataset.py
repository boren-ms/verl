"""JSONL loading and transforms required by Base3a training and validation."""

import gzip
import os
import re
import string
from pathlib import Path

import blobfile as bf
import pyarrow as pa
import pyarrow.json as pjson
from datasets import Dataset, Sequence, Value, concatenate_datasets
from recipe.phimm.data.prompts import get_task_output, get_task_prompt, resolve_language
from recipe.phimm.utils.languages import LANGUAGES
from recipe.phimm.utils.shared import (
    all_rank_print,
    dist_state,
    get_value,
    rank_print,
)

from verl.audio_cache import submit_audio_cache_dataset


def jsonl_dataset(jsonl_paths, **kwargs):
    """Load a JSONL dataset from the specified paths.

    Supports both plain ``.jsonl`` files and gzip-compressed ``.jsonl.gz`` files.
    """

    parse_options = pjson.ParseOptions(
        explicit_schema=pa.schema([pa.field("keywords", pa.list_(pa.string()))]),
        unexpected_field_behavior="infer",
    )

    def load_jsonl(file_path):
        with bf.BlobFile(file_path, "rb") as file_obj:
            if file_path.endswith(".gz"):
                with gzip.GzipFile(fileobj=file_obj, mode="rb") as gz:
                    return Dataset(pjson.read_json(gz, parse_options=parse_options))
            return Dataset(pjson.read_json(file_obj, parse_options=parse_options))

    return _load_expanded_datasets(jsonl_paths, ext=("jsonl", "jsonl.gz"), load_fn=load_jsonl)


def _has_glob_pattern(file_path):
    return any(char in file_path for char in "*?[")


def _expand_paths(file_paths, ext="parquet"):
    exts = (ext,) if isinstance(ext, str) else tuple(ext)
    expanded_files = []
    for file_path in file_paths:
        file_path = os.path.expanduser(str(file_path))
        if _has_glob_pattern(file_path):
            matches = sorted(bf.glob(file_path))
        elif bf.isdir(file_path):
            matches = []
            for e in exts:
                matches.extend(bf.glob(bf.join(file_path, f"*.{e}")))
            matches = sorted(matches)
        elif bf.exists(file_path):
            matches = [file_path]
        else:
            matches = []
        if not matches:
            raise FileNotFoundError(f"No {'/'.join(exts)} files matched: {file_path}")
        expanded_files.extend(matches)
    return expanded_files


def _load_expanded_datasets(file_paths, ext, load_fn):
    data_files = [file_paths] if isinstance(file_paths, str) else file_paths
    datasets = [load_fn(file_path) for file_path in _expand_paths(data_files, ext=ext)]
    if len(datasets) <= 1:
        return datasets[0]
    try:
        return concatenate_datasets(datasets)
    except ValueError as e:
        rank_print(f"[WARN] concatenate_datasets failed ({e}); retrying with pyarrow permissive promotion")
        tables = [ds.data.table if hasattr(ds.data, "table") else ds._data.table for ds in datasets]
        merged = pa.concat_tables(tables, promote_options="permissive")
        return Dataset(merged)


def get_num_proc(num_proc):
    if num_proc == "auto":
        n_cpu = os.cpu_count()
        n_proc = max(dist_state().num_processes, 2)
        num_proc = int(n_cpu / n_proc)
    return num_proc


def pop_map_kwargs(kwargs):
    output = {}
    streaming = kwargs.get("streaming", False)
    if (num_proc := kwargs.pop("num_proc", None)) and not streaming:
        output["num_proc"] = get_num_proc(num_proc)
    if remove_columns := kwargs.pop("remove_columns", None):
        output["remove_columns"] = remove_columns
    if batch_size := kwargs.pop("batch_size", None):
        output["batch_size"] = batch_size
    return output


def clean_tagged_text(ds, **kwargs):
    """Strip angle- or square-bracket tags and move paired contents to keywords."""
    src_field = kwargs.get("src_field", "text")
    tgt_field = kwargs.get("tgt_field", "keywords")
    paired_tag_pattern = re.compile(
        r"<([A-Za-z][\w:.-]*)\b[^>]*>"
        r"((?:(?!<\1\b[^>]*>).)*?)</\1\s*>"
        r"|\[([A-Za-z][\w:.-]*)\]"
        r"((?:(?!\[\3\]).)*?)\[/\3\s*\]",
        re.DOTALL | re.IGNORECASE,
    )
    inline_tag_pattern = re.compile(
        r"\[([A-Za-z][\w:.-]*)(?:\s+|&lt;)"
        r"((?:(?!\[[A-Za-z][\w:.-]*(?:\s+|&lt;)).)*?)\s*\]",
        re.DOTALL | re.IGNORECASE,
    )
    tag_pattern = re.compile(
        r"<[^>]*>"
        r"|\[/?[A-Za-z][\w:.-]*\s*(?:\]|>)"
        r"|\[/?[A-Za-z][\w:.-]*(?:&lt;)?(?=\s)",
        re.IGNORECASE,
    )

    def remove_tags(value):
        value = tag_pattern.sub(" ", value)
        return re.sub(r"\s+", " ", value).strip()

    def clean_tagged_value(value):
        return remove_tags(value or "").strip(string.punctuation + string.whitespace)

    def replace_paired_tag(match):
        value = match.group(2) or match.group(4) or ""
        text = clean_tagged_value(value)
        # Transcribers sometimes put the word separator inside the entity tag,
        # e.g. "from<NE> Morgan Stanley</NE>". Keep it when removing the tag.
        if value and value[0].isspace():
            text = " " + text
        if value and value[-1].isspace():
            text += " "
        return text

    def replace_inline_tag(match):
        return clean_tagged_value(match.group(2))

    def clean_text(egs):
        text = get_value(egs, src_field, "") or ""
        keywords = (
            list(dict.fromkeys(egs.get(tgt_field) or [])) if kwargs.get("preserve_existing_keywords", False) else []
        )
        tagged_values = [
            (match.start(), match.group(2) or match.group(4)) for match in paired_tag_pattern.finditer(text)
        ]
        tagged_values.extend((match.start(), match.group(2)) for match in inline_tag_pattern.finditer(text))
        for _, tagged_value in sorted(tagged_values):
            keyword = clean_tagged_value(tagged_value)
            if keyword and keyword not in keywords:
                keywords.append(keyword)

        text = paired_tag_pattern.sub(replace_paired_tag, text)
        text = inline_tag_pattern.sub(replace_inline_tag, text)
        text = remove_tags(text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return {src_field: text, tgt_field: keywords}

    features = ds.features.copy()
    features[tgt_field] = Sequence(Value("string"))
    return ds.map(clean_text, features=features, **pop_map_kwargs(kwargs))


def path_map(ds, **kwargs):
    """Map the dataset paths."""
    field = kwargs.get("field", "audio_path")
    src_part = kwargs.get("src_part", None)
    dst_part = kwargs.get("dst_part", None)

    def map_fn(x):
        x[field] = x[field].replace(src_part, dst_part)
        return x

    if src_part and dst_part:
        ds = ds.map(map_fn, **pop_map_kwargs(kwargs))
    return ds


def cache_audio(ds, **kwargs):
    """Submit remote audio fields for background caching without changing them."""
    fields = kwargs.get("fields", ["audio_path", "audio_chunk"])
    max_workers = int(kwargs.pop("max_workers", 16))
    submit_audio_cache_dataset(ds, fields, max_workers=max_workers)
    return ds


def rename_fields(ds, **kwargs):
    """Map the dataset fields."""
    mappings = kwargs.get("mappings", {})

    def rename_fn(egs):
        output = {}
        for src, dst in mappings.items():
            output[src] = get_value(egs, dst, None)
        return output

    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs["remove_columns"] = list(set(mappings.values()))
    ds = ds.map(rename_fn, **map_kwargs, desc="Renaming fields")
    return ds


def merge_kwargs(*args):
    """Merge two dictionaries, with overrides taking precedence."""
    merged = {}
    for d in args:
        merged.update(d)
    return merged


def limit_ds(ds, egs_limit=None):
    """Limit the dataset to a maximum number of examples."""
    if egs_limit is not None and len(ds) > egs_limit:
        all_rank_print(f"Limiting dataset from {len(ds)} to {egs_limit} examples.")
        ds = ds.take(egs_limit)
    return ds


def to_user_msg(prompt):
    if not isinstance(prompt, str):
        return prompt

    return [{"role": "user", "content": prompt}]


def _extra_info_value(egs, key):
    """Get extra_info value with defaults for required keys."""
    value = egs.get(key)
    if key == "id" and not value:
        audio_ref = egs.get("audio_path") or egs.get("audio_chunk") or egs.get("audio_file") or ""
        value = Path(str(audio_ref)).stem if audio_ref else "unknown_id"
    if key == "language" and not value:
        value = "English"
    if key == "keywords" and value is None:
        value = [""]
    return value


def verl_format_ds(ds, **kwargs):
    """Format the dataset for verl training."""
    prompt_key = kwargs.get("prompt_key", "prompt")
    default_extra_keys = ["id", "language", "keywords"]
    configured_extra_keys = kwargs.get("extra_keys", []) or []
    extra_keys = list(dict.fromkeys(default_extra_keys + list(configured_extra_keys)))

    def map_fn(egs):
        text = egs.get("text", "")
        result = {
            prompt_key: to_user_msg(egs[prompt_key]),
            "reward_model": {"ground_truth": text, "gt_output": egs.get("gt_output", text)},
            "extra_info": {key: _extra_info_value(egs, key) for key in extra_keys},
            "data_source": egs.get("data_source", "asr"),
        }
        return result

    col_names = [x for x in ds.column_names if not x.startswith("audio")]
    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs["remove_columns"] = list(set(map_kwargs.get("remove_columns", []) + col_names))
    ds = ds.map(map_fn, **map_kwargs, desc="RL formatting")
    return ds


def add_field_ds(ds, **kwargs):
    """Add or overwrite constant fields in the dataset."""
    fields = kwargs.get("fields", {})

    def map_fn(egs):
        return fields

    return ds.map(map_fn, **pop_map_kwargs(kwargs))


def process_ds(ds, **kwargs):
    """Post process the dataset."""
    map_kwargs = pop_map_kwargs(kwargs)
    unsupported = set(kwargs) - {
        "input_egs_limit",
        "path_map",
        "rename_fields",
        "clean_tagged_text",
        "cache_audio",
        "add_field",
        "verl_format",
    }
    if unsupported:
        raise ValueError(f"Unsupported Base3a data transforms: {sorted(unsupported)}")
    if input_egs_limit := kwargs.get("input_egs_limit", None):
        ds = limit_ds(ds, egs_limit=input_egs_limit)
    if path_map_kwargs := kwargs.get("path_map", {}):
        ds = path_map(ds, **merge_kwargs(map_kwargs, path_map_kwargs))
    if rename_fields_kwargs := kwargs.get("rename_fields", {}):
        ds = rename_fields(ds, **merge_kwargs(map_kwargs, rename_fields_kwargs))
    if "clean_tagged_text" in kwargs:
        clean_tagged_text_kwargs = dict(kwargs.get("clean_tagged_text") or {})
        ds = clean_tagged_text(ds, **merge_kwargs(map_kwargs, clean_tagged_text_kwargs))
    if "cache_audio" in kwargs:
        cache_audio_kwargs = kwargs.get("cache_audio") or {}
        ds = cache_audio(ds, **merge_kwargs(map_kwargs, cache_audio_kwargs))
    if add_field_kwargs := kwargs.get("add_field", {}):
        ds = add_field_ds(ds, **merge_kwargs(map_kwargs, add_field_kwargs))
    if verl_format_kwargs := kwargs.get("verl_format", {}):
        ds = verl_format_ds(ds, **merge_kwargs(map_kwargs, verl_format_kwargs))
    return ds


def mask_foreign_language(ds, *, task_info, **kwargs):
    """Mask foreign speech for verbatim language-hint examples.

    Mixed-language rows require ordered, language-labelled components covering
    the original text; the block never guesses language boundaries.
    """
    if task_info.get("style", "verb") != "verb" or not task_info.get("lang_hint", False):
        return ds
    hint = task_info.get("language")
    if not hint:
        raise ValueError("mask_foreign_language requires an explicit hint language")
    hint = resolve_language(hint)
    if hint.casefold() not in LANGUAGES:
        raise ValueError("mask_foreign_language requires one known hint language")
    marker = kwargs.get("marker", "<foreign_language>")
    if not isinstance(marker, str) or not marker:
        raise ValueError("mask_foreign_language marker must be a nonempty string")

    def single_language(value):
        language = resolve_language(value)
        if language.casefold() not in LANGUAGES:
            raise ValueError("mask_foreign_language requires one known component language")
        return language

    def transform(egs):
        source = egs.get("language")
        if not source or not isinstance(egs.get("text"), str):
            raise ValueError("mask_foreign_language requires source language and transcription")
        components = egs.get("components")
        if components:
            if any(not c.get("language") or not isinstance(c.get("text"), str) for c in components):
                raise ValueError("each language component must have a language and string text")
            languages = [single_language(c["language"]) for c in components]
            original = egs["text"]
            pieces = []
            offset = 0
            for component, language in zip(components, languages, strict=True):
                text = component["text"].strip()
                if not text:
                    continue
                start = original.find(text, offset)
                if start < 0 or original[offset:start].strip():
                    raise ValueError("component text must cover the original transcription in order")
                pieces.append(original[offset:start])
                pieces.append(text if language.casefold() == hint.casefold() else marker)
                offset = start + len(text)
            if original[offset:].strip():
                raise ValueError("component text must cover the original transcription in order")
            pieces.append(original[offset:])
            # A wholly foreign utterance has one target marker even when it has
            # several components. Mixed utterances retain their source spacing.
            if (
                len({lang.casefold() for lang in languages}) == 1
                and languages[0].casefold() != hint.casefold()
                and original.strip()
            ):
                text = marker
            else:
                text = "".join(pieces)
            return {"text": text, "language": hint, "components": None}
        if len(resolve_language(source).split()) != 1:
            raise ValueError("mixed source language requires language-labelled components")
        language = single_language(source)
        return {
            "text": egs["text"] if not egs["text"].strip() or language.casefold() == hint.casefold() else marker,
            "language": hint,
        }

    return ds.map(transform, **pop_map_kwargs(kwargs), desc="Masking foreign-language speech")


def add_task_info(ds, *, style="verb", language=None, lang_hint=False, **kwargs):
    """Add the prompt, formatted output, and resolved language without changing text."""

    def format_jsonl_example(row):
        # A merged list can contain multiple spoken languages. The configured
        # source hint filled missing languages before merging; retain the union
        # here when constructing the language hint and formatted reference.
        merged_list = row.get("random_merge_count") is not None and isinstance(row.get("audio_path"), list)
        hint = row.get("language") if merged_list else language
        lang = resolve_language(hint or row.get("language") or "English")
        return {
            "prompt": get_task_prompt(style=style, language=lang, lang_hint=lang_hint),
            "gt_output": get_task_output(
                style=style,
                language=lang,
                text=row.get("text", ""),
                components=row.get("components"),
            ),
            "language": lang,
        }

    return ds.map(format_jsonl_example, **pop_map_kwargs(kwargs))


def augment(ds, **kwargs):
    """Augment the dataset with additional information."""
    map_kwargs = pop_map_kwargs(kwargs)
    if pre_process_kwargs := kwargs.get("pre_process", {}):
        ds = process_ds(ds, **merge_kwargs(map_kwargs, pre_process_kwargs))
    if "mask_foreign_language" in kwargs:
        mask_kwargs = kwargs.get("mask_foreign_language") or {}
        ds = mask_foreign_language(
            ds, task_info=kwargs.get("add_task_info") or {}, **merge_kwargs(map_kwargs, mask_kwargs)
        )
    if add_task_info_kwargs := kwargs.get("add_task_info", {}):
        ds = add_task_info(ds, **merge_kwargs(map_kwargs, add_task_info_kwargs))
    if post_process_kwargs := kwargs.get("post_process", {}):
        ds = process_ds(ds, **merge_kwargs(map_kwargs, post_process_kwargs))
    return ds


def create_audio_dataset(**kwargs):
    """Load the JSONL dependency closure used by Base3a and its validation suites."""
    if kwargs.get("dataset_name") != "jsonl":
        raise ValueError("The Base3a recipe supports dataset_name=jsonl")
    with dist_state().local_main_process_first():
        ds = augment(jsonl_dataset(**kwargs), **kwargs)
    return ds

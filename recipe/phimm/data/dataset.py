# %%
import os
import re
import uuid
import math
import json
import gzip
from collections import defaultdict
import ast
import random
import socket
import blobfile as bf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as pjson
import pyarrow.csv as pcsv
import string
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3]))
from datasets import load_dataset, concatenate_datasets, Dataset
from bs4 import BeautifulSoup
from recipe.phimm.data.error_simu import ErrorSimulator
from recipe.phimm.data.biasing import PieceSampler, tag_pieces, text_norm as biasing_text_norm
from recipe.phimm.data.prompts import resolve_task_language, get_task_prompt, get_task_prefix, get_task_output
from recipe.phimm.utils.tn import text_norm
from recipe.phimm.data.chunk import get_chunk_manager, create_chunk_datasets
from recipe.phimm.data.audio_augment import AudioAugmenter, safe_audio_stem
from recipe.phimm.utils.shared import (
    hash_id,
    get_value,
    rank_print,
    dist_state,
    all_rank_print,
    to_list,
    in_range,
    is_list,
    to_int,
    to_float,
    unbatch,
    has_brackets as has_brackets_fn,
    parse_asr_response,
    has_repeat_error,
    has_missing_keyword,
    has_tail_hallucination,
)
from recipe.phimm.utils.audio import sf_read, sf_write, load_raw_audio
from recipe.phimm.utils.languages import get_language_name
from recipe.phimm.utils.storage import get_path_with_options

prompt_format = "<audio>\n{}"


def format_asr_prompt(prompt, model_version=None):
    if model_version == 2607:
        return f"{prompt}<audio>"
    return prompt_format.format(prompt)


def read_words(file_path, num=None, tn_name=None):
    """Read the top N lines from a file."""
    if file_path is None:
        return []
    words = []
    tn_name = tn_name or "identity"
    with bf.BlobFile(file_path, "r") as f:
        for i, line in enumerate(f):
            if num is not None and i >= num:
                break
            word = line.split()[0]
            words.append(text_norm(word, tn_name))
    return words


def read_word_count(file_path, num=None, tn_name=None):
    """Read the top N lines from a file."""
    if file_path is None:
        return []
    wd_cnt = defaultdict(int)
    tn_name = tn_name or "identity"
    with bf.BlobFile(file_path, "r") as f:
        for i, line in enumerate(f):
            if num is not None and i >= num:
                break
            parts = line.split()
            word = text_norm(parts[0], tn_name)
            cnt = to_int(parts[1], 0) if len(parts) > 1 else 0
            wd_cnt[word] += cnt
    return wd_cnt


def prefix_match(str1, str2, nd=2):
    n = min(len(str1), len(str2))
    m = max(len(str1), len(str2))
    if m - n > nd:
        return False
    return str1[:n] == str2[:n]


def has_digit(s):
    return any(c.isdigit() for c in s)


def find_rare(srcs, tgts, nd=2):
    srcs = set(srcs) - set(tgts)
    lefts = []
    for src in srcs:
        if any(prefix_match(src, tgt, nd=nd) for tgt in tgts):
            continue
        if has_digit(src):
            continue
        lefts.append(src)
    return lefts


def extract_entities(text):
    """extract named entities from text that are surrounded by <NE> </NE> or <NE:type> </NE:type> tags."""

    bs = BeautifulSoup(text, "html.parser")
    entities = [tag.get_text().strip() for tag in bs.find_all() if tag.name.startswith("ne")]
    return set(entities)


def jsonl_dataset(jsonl_paths, **kwargs):
    """Load a JSONL dataset from the specified paths.

    Supports both plain ``.jsonl`` files and gzip-compressed ``.jsonl.gz`` files.
    """

    def load_jsonl(file_path):
        with bf.BlobFile(file_path, "rb") as file_obj:
            if file_path.endswith(".gz"):
                with gzip.GzipFile(fileobj=file_obj, mode="rb") as gz:
                    return Dataset(pjson.read_json(gz))
            return Dataset(pjson.read_json(file_obj))

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


def parquet_dataset(parquet_paths, **kwargs):
    """Load a Parquet dataset from the specified paths."""

    def load_parquet(file_path):
        with bf.BlobFile(file_path, "rb") as file_obj:
            return Dataset(pq.read_table(file_obj))

    return _load_expanded_datasets(parquet_paths, ext="parquet", load_fn=load_parquet)


def update_dir(data_path, src_dir=None, dst_dir=None):
    if not src_dir or not dst_dir:
        return data_path
    data_path = str(data_path)
    src_dir = src_dir.rstrip("/") + "/"  # Ensure src_dir is a clean path
    dst_dir = dst_dir.rstrip("/") + "/"  # Ensure dst_dir is a clean path
    return data_path.replace(src_dir, dst_dir) if data_path.startswith(src_dir) else data_path


def get_num_proc(num_proc):
    if num_proc == "auto":
        n_cpu = os.cpu_count()
        n_proc = max(dist_state().num_processes, 2)
        num_proc = int(n_cpu / n_proc)
    return num_proc


def pop_filter_kwargs(kwargs):
    output = {}
    streaming = kwargs.get("streaming", False)
    if (num_proc := kwargs.pop("num_proc", None)) and not streaming:
        output["num_proc"] = get_num_proc(num_proc)
    return output


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


def ls_bias_dataset(jsonl_path, bias_key=None, with_gt=False, min_word_len=None, bias_sort=False, tag="*", data_dir=None, **kwargs):
    """Create a dataset from the given split."""
    ds = jsonl_dataset(jsonl_path, **kwargs)

    def load_sample(example):
        """Load audio from a file."""
        bias_words = example.get(bias_key, [])
        gt_words = example.get("ground_truth", [])
        if with_gt:
            bias_words = list(set(bias_words) | set(gt_words))
        if min_word_len is not None:
            bias_words = [w for w in bias_words if len(w) >= min_word_len]
        if bias_sort:
            bias_words = sorted(bias_words)
        bias_str = ", ".join(tag_pieces(bias_words, tag=tag))
        prompt = get_task_prompt(task="biasing" if bias_str else "asr")
        audio_path = update_dir(example["audio_path"], src_dir="/root/data", dst_dir=data_dir)
        words = example.get("text", "").strip().split()
        words = tag_pieces(words, tag=tag, specified=gt_words, norm=biasing_text_norm)
        return {
            "prompt": prompt_format.format(f"{prompt} {bias_str}"),
            "audio_path": audio_path,
            "text": " ".join(words),
            "keywords": gt_words,
            "id": example.get("id", Path(audio_path).stem),
        }

    ds = ds.map(load_sample, **pop_map_kwargs(kwargs))
    return ds


def chunk_dataset(specs, max_cached_chunk=None, **kwargs):
    """Iterate over the chunk dataset based on the specification files."""
    if max_cached_chunk is not None:
        get_chunk_manager(max_cached_chunk)  # Initialize the chunk manager with a maximum size. and reuse later.
    map_kwargs = pop_map_kwargs(kwargs)
    np = map_kwargs.get("num_proc", None)
    streaming = kwargs.get("streaming", False)
    print(f"Creating chunk {'streaming' if streaming else 'non-streaming'} dataset (NP={np}), please be patient.")
    ds = create_chunk_datasets(specs, **map_kwargs, **kwargs)
    if isinstance(ds, Dataset):
        print(f"Loaded {len(ds)} examples from chunk dataset.")
    if "transcription" in ds.column_names:
        ds = ds.rename_column("transcription", "text")
    return ds


def entity_dataset(
    jsonl_path,
    max_bias=0,
    entity_file=None,
    distractor_file=None,
    word_bias=False,
    tag="*",
    src_dir=None,
    data_dir=None,
    **kwargs,
):
    ds = jsonl_dataset(jsonl_path, **kwargs)
    distractors = read_words(distractor_file)
    shared_entities = read_words(entity_file)

    def load_sample(example):
        """Load audio from a file."""
        nonlocal src_dir  # not a local variable

        trans = example.get("Transcription", "").strip()
        src_dir = src_dir or "/datablob1/users/ruchaofan"
        audio_path = update_dir(example["WavPath"], src_dir=src_dir, dst_dir=data_dir)
        bs = BeautifulSoup(trans, "html.parser")

        entities = [tag.get_text().strip() for tag in bs.find_all() if tag.name.startswith("ne")]
        entities = list(set(entities + shared_entities))  # Combine with shared entities

        utt_id = example.get("UUID", Path(audio_path).stem)

        if max_bias > 0 and max_bias < len(entities):
            print(f"Groundtruth words [{len(entities)}] exceed max_bias [{max_bias}], truncating.")
        bias_entities = entities.copy()[:max_bias]
        if word_bias:
            bias_words = set([word for entity in bias_entities for word in entity.split()])  # split entities into words
        else:
            bias_words = set(bias_entities)
        bias_words.update(distractors[: max(0, max_bias - len(bias_words))])

        bias_str = ", ".join(tag_pieces(bias_words, tag=tag))
        prompt = get_task_prompt(task="biasing" if bias_str else "asr")

        return {
            "prompt": prompt_format.format(f"{prompt} {bias_str}"),
            "audio_path": audio_path,
            "text": bs.get_text().strip(),
            "keywords": entities,
            "id": utt_id,
        }

    return ds.map(load_sample, **pop_map_kwargs(kwargs))


def audio_dir_dataset(data_dir, **kwargs):
    """Load audio files from a directory."""
    examples = []
    for audio_file in bf.glob(f"{data_dir}/*.wav"):
        text_file = audio_file.replace(".wav", ".txt")
        if not bf.exists(text_file):
            continue
        with bf.BlobFile(text_file, "r") as f:
            trans = f.read().strip()
        examples.append(
            {
                "audio_path": audio_file,
                "text": trans,
            }
        )
    return Dataset.from_list(examples)


def tsv_dataset(tsv_paths, **kwargs):
    """Load a TSV dataset from the specified paths."""

    def load_tsv(file_path):
        with bf.BlobFile(file_path, "rb") as file_obj:
            table = pcsv.read_csv(
                file_obj,
                parse_options=pcsv.ParseOptions(delimiter="\t"),
                read_options=pcsv.ReadOptions(column_names=["id", "paths", "msgs"]),
            )
        ds = Dataset(table)
        tsv_dir = file_path.rsplit("/", 1)[0]  # get the directory of the tsv file, do not use os.path
        return ds.add_column("dir", [tsv_dir] * len(ds))

    ds = _load_expanded_datasets(tsv_paths, ext="tsv", load_fn=load_tsv)
    # ds = stream_shuffle(ds, **kwargs) # need stream or shuffle

    def load_sample(egs):
        """Process a single sample."""
        audio_path = ast.literal_eval(egs["paths"])[0]
        if egs["dir"]:
            audio_path = audio_path.replace("/root/data/LibriSpeech", egs["dir"])
        messages = ast.literal_eval(egs["msgs"])[0]["messages"]
        x = {
            "prompt": prompt_format.format("Transcribe the audio clip into text."),
            "audio_path": audio_path,
            "text": messages[-1]["content"],
            "id": egs["id"],
        }
        return x

    ds = ds.map(load_sample, **pop_map_kwargs(kwargs))
    return ds


def openasr_dataset(**kwargs):
    """Create a dataset from the given split."""
    name = kwargs.get("name", "librispeech")
    split = kwargs.get("split", "test.clean")
    ds = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        name,
        split=split,
    )
    ds = stream_shuffle(ds, **kwargs)
    return ds


def bias_sampling(ds, **kwargs):
    """Apply bias sampling to the dataset."""
    rand_prompt = kwargs.pop("rand_prompt", False)
    map_kwargs = pop_map_kwargs(kwargs)
    kwargs = kwargs or {
        "bias_prob": 0.9,
        "hit_prob": 0.9,
        "max_piece_len": 1,
    }
    bias_sampler = PieceSampler(**kwargs)

    def proc_sample(sample):
        """Process a sample from the dataset."""
        context, text, keywords = bias_sampler.sample(sample["text"])
        if context:
            prompt = get_task_prompt(task="biasing", rand=rand_prompt)
            prompt = f"{prompt} {context}"
        else:
            prompt = get_task_prompt(task="asr", rand=rand_prompt)
        return {
            "prompt": prompt_format.format(prompt),
            "text": text,  # text is updated
            "keywords": keywords,
            "context": context,
        }

    ds = ds.map(proc_sample, **map_kwargs)
    return ds


def to_chat(text, chat=True, role="assistant"):
    """Convert text to conversation format."""
    if not chat:
        return text
    assert role in ["assistant", "user"], "Role must be either 'assistant' or 'user'."
    return [
        {
            "role": role,
            "content": text,
        }
    ]


def format_preference(ds, **kwargs):
    """Format the preference for the dataset."""
    chosen_key = kwargs.get("chosen_key", "chosen")
    rejected_key = kwargs.get("rejected_key", "rejected")
    prompt_key = kwargs.get("prompt_key", "prompt")

    def format_sample(sample):
        """Format a single sample."""
        return {
            "prompt": sample.get(prompt_key, None),
            "chosen": sample.get(chosen_key, None),
            "rejected": sample.get(rejected_key, None),
        }

    return ds.map(format_sample, **pop_map_kwargs(kwargs))


def simulate_preference(ds, **kwargs):
    """simulate the preference  to the dataset."""
    error_range = kwargs.pop("error_range", (0.1, 0.25))
    num_rejections = kwargs.pop("num_rejections", 1)
    chat = kwargs.get("chat", False)
    if not is_list(error_range):
        error_range = [float(error_range), float(error_range)]
    simulator = ErrorSimulator(**kwargs)

    def add_preference(sample, error_range):
        """Process a sample from the dataset."""
        text = sample["text"]
        rejections = [simulator.random_error(text, random.uniform(*error_range)) for _ in range(num_rejections)]
        return {
            "chosen": to_chat(text, chat),
            "rejected": [to_chat(x, chat) for x in rejections],
        }

    return ds.map(add_preference, fn_kwargs={"error_range": error_range}, **pop_map_kwargs(kwargs))


def load_audio(ds, **kwargs):
    """Post process the dataset."""

    def read_audio(sample):
        """Read audio from the file."""
        audio, sr = sf_read(sample["audio_path"])
        return {"audio": audio, "sr": sr}

    ds = ds.map(read_audio, **pop_map_kwargs(kwargs))
    return ds


def filter_ds(ds, **kwargs):
    """Filter the dataset."""
    wer_file = kwargs.get("wer_file", None)
    if wer_file and bf.exists(wer_file):
        with bf.BlobFile(wer_file, "r") as f:
            df = pd.read_json(f, lines=True)
        if wer_range := kwargs.get("wer_range", None):
            df = df[(df["WER"] >= wer_range[0]) & (df["WER"] <= wer_range[1])]
        ids = df["id"].tolist()
        n_egs = len(ds)
        ds = ds.filter(lambda x: x["id"] in ids, **pop_filter_kwargs(kwargs))
        print(f"Filter dataset: {n_egs} to {len(ds)}")
    return ds


def trim_silence(ds, **kwargs):
    """Trim head/tail silence from audio files using Silero VAD."""
    from recipe.phimm.utils.trim_silence import SilenceTrimmer

    output_dir = kwargs.get("output_dir", None)
    rand_cut_ms = kwargs.get("rand_cut_ms", 0)
    min_dur_ms = kwargs.get("min_dur_ms", 300)

    SilenceTrimmer._ensure_model()

    def trim_batch(batch):
        trimmer = SilenceTrimmer(threshold=kwargs.get("threshold", 0.5))
        out_paths = []
        for example in unbatch(batch):
            try:
                audio, sr = load_raw_audio(example)
                hc = random.randint(0, rand_cut_ms) if rand_cut_ms > 0 else 0
                tc = random.randint(0, rand_cut_ms) if rand_cut_ms > 0 else 0
                trimmed = trimmer.trim(audio, sr, head_cut_ms=hc, tail_cut_ms=tc)
                if trimmed is None or len(trimmed) / sr * 1000 < min_dur_ms:
                    out_paths.append(example.get("audio_path", ""))
                    continue
                out_path = f"{output_dir.rstrip('/')}/{uuid.uuid4().hex}.wav"
                sf_write(out_path, trimmed, sr)
                out_paths.append(out_path)
            except Exception as e:
                print(f"[WARN] trim_silence: {e}")
                out_paths.append(example.get("audio_path", ""))
        return {"audio_path": out_paths}

    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs.setdefault("batch_size", 16)
    ds = ds.map(trim_batch, batched=True, **map_kwargs, desc="Trimming silence")
    n_before = len(ds)
    ds = ds.filter(lambda x: bool(x.get("audio_path", "")))
    n_after = len(ds)
    if n_before != n_after:
        print(f"Filtered empty audio_path after trim: {n_before} => {n_after}")
    return ds


def load_timestamps(timestamps):
    """Parse timestamps from list or serialized string."""
    if isinstance(timestamps, str):
        try:
            timestamps = json.loads(timestamps)
        except json.JSONDecodeError:
            try:
                timestamps = ast.literal_eval(timestamps)
            except (ValueError, SyntaxError):
                return []
    return timestamps if isinstance(timestamps, list) else []


def trim_tailing(ds, **kwargs):
    """Randomly remove trailing words and cut audio based on word timestamps."""

    output_dir = kwargs.get("output_dir", None)
    cut_rate = kwargs.get("max_cut_rate", 0.1)
    ts_field = kwargs.get("ts_field", "word_timestamps")
    min_dur_ms = kwargs.get("min_dur_ms", 300)

    def trim_batch(batch):
        out_paths = []
        out_texts = []
        for example in unbatch(batch):
            try:
                words = load_timestamps(example.get(ts_field, None))
                last_n = math.ceil((1 - random.random() * cut_rate) * len(words))
                words = words[:last_n]
                if not words:
                    out_paths.append(example.get("audio_path", ""))
                    out_texts.append(example.get("text", ""))
                    continue
                text = " ".join([w.get("text", "") for w in words]).strip()
                end = to_float(words[-1].get("end", 0.0))
                if end <= 0:
                    out_paths.append(example.get("audio_path", ""))
                    out_texts.append(example.get("text", ""))
                    continue
                audio, sr = load_raw_audio(example)
                if end * 1000 < min_dur_ms:
                    out_paths.append(example.get("audio_path", ""))
                    out_texts.append(example.get("text", ""))
                    continue
                out_path = f"{output_dir.rstrip('/')}/{uuid.uuid4().hex}.wav"
                sf_write(out_path, audio[: int(end * sr)], sr)
                out_paths.append(out_path)
                out_texts.append(text)
            except Exception as e:
                print(f"[WARN] trim_tailing: {e}")
                out_paths.append(example.get("audio_path", ""))
                out_texts.append(example.get("text", ""))
        return {
            "audio_path": out_paths,
            "text": out_texts,
        }

    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs.setdefault("batch_size", 16)
    ds = ds.map(trim_batch, batched=True, **map_kwargs, desc="Trimming trailing words")
    ds = ds.filter(lambda x: bool(x.get("audio_path", "")))
    return ds


def filter_short_audio(ds, **kwargs):
    """Filter out audio samples shorter than min_dur_ms milliseconds."""
    min_dur_ms = kwargs.get("min_dur_ms", 300)
    min_dur = min_dur_ms / 1000.0
    batch_size = kwargs.get("batch_size", 64)

    def is_long_enough_batch(batch):
        examples = unbatch(batch)
        results = []
        for example in examples:
            try:
                audio, sr = load_raw_audio(example)
                results.append(len(audio) / sr >= min_dur)
            except Exception as e:
                print(f"[WARN] filter_short_audio: {e}")
                results.append(False)
        return results

    n_egs = len(ds)
    ds = ds.filter(is_long_enough_batch, batched=True, batch_size=batch_size, num_proc=1, desc="Filtering short audio")
    print(f"Filtered short audio (<{min_dur}s): {n_egs} => {len(ds)} [{len(ds) / n_egs:.2%}]")
    return ds


def augment_audio(ds, **kwargs):
    """Write randomly speed/noise augmented audio and update audio_path."""
    output_dir = kwargs.get("output_dir", None)
    assert output_dir is not None, "augment_audio.output_dir must be set"
    audio_output_dir = f"{output_dir.rstrip('/')}/audio"
    if not audio_output_dir.startswith("az://"):
        Path(audio_output_dir).mkdir(parents=True, exist_ok=True)

    seed = int(kwargs.get("seed", 0))
    augmenter = AudioAugmenter(
        speed_prob=kwargs.get("speed_prob", 1.0),
        speed_range=to_list(kwargs.get("speed_range", [0.9, 1.1])),
        noise_prob=kwargs.get("noise_prob", 1.0),
        snr_range=to_list(kwargs.get("snr_range", [10.0, 12.0])),
        noise_path=kwargs.get("noise_path", None),
        peak=kwargs.get("peak", 0.99),
        seed=seed,
    )

    def augment_batch(batch, indices):
        output_paths = []
        source_paths = []
        speed_factors = []
        snr_dbs = []
        chosen_noise_paths = []

        for example, idx in zip(unbatch(batch), indices, strict=True):
            source_path = example.get("audio_path") or example.get("audio_chunk") or example.get("audio_file", "")
            try:
                audio, sr = load_raw_audio(example)
                augmented, aug_info = augmenter.augment(audio, sr, idx)

                out_name = f"{idx:08d}_{safe_audio_stem(Path(source_path).stem, idx)}.wav"
                out_path = f"{audio_output_dir.rstrip('/')}/{out_name}"
                sf_write(out_path, augmented, sr)

                output_paths.append(out_path)
                source_paths.append(source_path)
                speed_factors.append(aug_info["speed_factor"])
                snr_dbs.append(aug_info["snr_db"])
                chosen_noise_paths.append(aug_info["noise_path"])
            except Exception as e:
                print(f"[WARN] augment_audio failed for index {idx}: {e}")
                output_paths.append("")
                source_paths.append(source_path)
                speed_factors.append(1.0)
                snr_dbs.append(-1)
                chosen_noise_paths.append("")

        return {
            "audio_path": output_paths,
            "source_audio_path": source_paths,
            "speed_factor": speed_factors,
            "snr_db": snr_dbs,
            "noise_path": chosen_noise_paths,
        }

    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs.setdefault("batch_size", 16)
    ds = ds.map(augment_batch, batched=True, with_indices=True, **map_kwargs, desc="Augmenting audio")
    n_before = len(ds)
    ds = ds.filter(lambda x: bool(x.get("audio_path", "")))
    n_after = len(ds)
    if n_before != n_after:
        print(f"Filtered failed audio augmentations: {n_before} => {n_after}")
    return ds


def audio_chunk_to_path(ds, **kwargs):
    """Convert ``audio_chunk`` specs to ``audio_path`` strings.

    Conversion rule: replace ``:`` with ``,`` and append ``.wav``.
    Existing destination paths are preserved.
    """
    dst_field = kwargs.get("dst_field", "audio_path")
    skip_existing = kwargs.get("skip_existing", True)

    def map_batch(batch, indices):
        out_paths = []
        for example, idx in zip(unbatch(batch), indices, strict=True):
            existing_path = str(example.get(dst_field, "") or "")
            if skip_existing and existing_path:
                out_paths.append(existing_path)
                continue
            chunk_spec = str(example.get("audio_chunk", "") or "")
            if not chunk_spec:
                out_paths.append(existing_path)
                continue
            converted = chunk_spec.replace(":", ",")
            if not converted.lower().endswith(".wav"):
                converted = f"{converted}.wav"
            try:
                if not bf.exists(converted):
                    if not converted.startswith("az://"):
                        Path(converted).parent.mkdir(parents=True, exist_ok=True)
                    audio, sr = load_raw_audio(example)
                    sf_write(converted, audio, sr)
            except Exception as e:
                print(f"[WARN] audio_chunk_to_path failed for index {idx}: {e}")
                out_paths.append(existing_path)
                continue
            out_paths.append(converted)
        return {dst_field: out_paths}

    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs.setdefault("batch_size", 16)
    ds = ds.map(map_batch, batched=True, with_indices=True, **map_kwargs, desc="Converting chunk specs to audio paths")
    return ds


def svad_explode(ds, **kwargs):
    """Run Smart VAD on each row's audio_path and explode into per-segment rows.

    For every row whose ``audio_path`` is a plain wav URI (no ``#start:end``
    chunk spec already), load the audio, run :class:`SVadChunker`
    (``max_len_sec=max_len_sec``), and emit one row per detected segment
    with ``audio_path`` rewritten to ``wav#start:end``. All other columns are
    preserved on every child row. ``seg_index`` / ``n_segments`` / ``seg_start``
    / ``seg_end`` / ``parent_audio_path`` are added so downstream aggregation
    can group hyps by parent.

    Rows already carrying a time-range chunk spec are passed through unchanged
    (with ``seg_index=0`` / ``n_segments=1``). Short wavs (<= ``max_len_sec``)
    also pass through as a single row.
    """
    from recipe.phimm.utils.audio import _is_time_chunk_spec, resample_audio
    from recipe.phimm.utils.svad.svad import SVadChunker
    import soundfile as sf

    max_len_sec = float(kwargs.get("max_len_sec", 30.0))
    min_seg_sec = float(kwargs.get("min_seg_sec", 0.1))
    audio_key = kwargs.get("audio_key", "audio_path")
    target_sr = int(kwargs.get("target_sr", 16000))
    skip_existing_chunks = kwargs.get("skip_existing_chunks", True)
    # Optional path prefix rewrite, e.g. {"/datablob1/": "az://orngwus2cresco/data/speech/"}
    path_replace = dict(kwargs.get("path_replace", {}) or {})

    def _translate(p: str) -> str:
        for src, dst in path_replace.items():
            if p.startswith(src):
                return dst + p[len(src):]
        return p

    chunker = SVadChunker(max_len_sec=max_len_sec, verbose=False)

    def _load_mono_16k(path: str):
        with bf.BlobFile(path, "rb") as f:
            data, sr = sf.read(f)
        if data.ndim == 2:
            data = data.mean(axis=1)
        import numpy as np
        data = np.asarray(data, dtype=np.float32)
        if sr != target_sr:
            data, sr = resample_audio(data, sr, target_sr)
        return data, sr

    def _explode_one(row: dict) -> list[dict]:
        path = str(row.get(audio_key, "") or "")
        if not path:
            return [row]
        path = _translate(path)
        row[audio_key] = path
        if skip_existing_chunks and _is_time_chunk_spec(path):
            out = dict(row)
            out.setdefault("seg_index", 0)
            out.setdefault("n_segments", 1)
            out.setdefault("parent_audio_path", path.rsplit("#", 1)[0])
            return [out]
        try:
            audio, sr = _load_mono_16k(path)
        except Exception as exc:
            print(f"[svad_explode] FAILED to load {path}: {exc}")
            return [row]
        dur = len(audio) / sr
        if dur <= max_len_sec:
            out = dict(row)
            out["seg_index"] = 0
            out["n_segments"] = 1
            out["seg_start"] = 0.0
            out["seg_end"] = round(dur, 3)
            out["parent_audio_path"] = path
            return [out]
        spans = chunker.chunk(audio, sr)
        kept = [(s, e) for (s, e) in spans if (e - s) >= min_seg_sec]
        if not kept:
            kept = [
                (index * max_len_sec, min((index + 1) * max_len_sec, dur))
                for index in range(math.ceil(dur / max_len_sec))
            ]
        rows_out = []
        for idx, (s, e) in enumerate(kept):
            child = dict(row)
            child[audio_key] = f"{path}#{round(float(s), 3)}:{round(float(e), 3)}"
            child["seg_index"] = idx
            child["n_segments"] = len(kept)
            child["seg_start"] = round(float(s), 3)
            child["seg_end"] = round(float(e), 3)
            child["parent_audio_path"] = path
            rows_out.append(child)
        print(f"[svad_explode] {path}: {dur:.1f}s -> {len(kept)} segments")
        return rows_out

    exploded: list[dict] = []
    for row in ds:
        exploded.extend(_explode_one(dict(row)))
    return Dataset.from_list(exploded)


def add_rare_keywords(ds, **kwargs):
    tn_name = kwargs.get("tn_name", "english")
    rare_ratio = kwargs.get("rare_ratio", None)
    rare_num = kwargs.get("rare_num", None)
    common_file = kwargs.get("common_file", None)
    common_num = kwargs.get("common_num", 10000)
    assert common_file is not None, "common_file must be set"
    wd_cnt = read_word_count(common_file, num=common_num, tn_name=tn_name)

    def rare_words(egs):
        text = text_norm(egs["text"], tn_name)
        words = set(text.split())
        words = [w for w in words if len(w) > 1]  # filter single character words
        n_rare = int(len(words) * rare_ratio) if rare_ratio else rare_num
        if not n_rare:
            keywords = [w for w in words if w not in wd_cnt]
        else:
            sorted_wds = sorted(words, key=lambda w: wd_cnt.get(w, 0))
            keywords = sorted_wds[:n_rare]

        return {"keywords": list(keywords)}

    ds = ds.map(rare_words, **pop_map_kwargs(kwargs))
    return ds


def extract_tags(text):
    """Return list of {"tag": "values"} occurrences."""
    pattern = re.compile(r"\[(\w+)\](.*?)\[/\1\]", re.DOTALL)
    matches = pattern.findall(text)
    results = defaultdict(list)
    for tag, value in matches:
        results[tag].append(value)
    return results


def add_tag_keywords(ds, **kwargs):
    """Add keywords extracted from tags in the text to the dataset."""
    src_field = kwargs.get("src_field", "info.alternative_transcription.lexical_tned_human_caption_mixed_case_GPT4o_raw")
    tgt_field = kwargs.get("tgt_field", "keywords")

    def tag_keywords(egs):
        text = get_value(egs, src_field, "")
        tags = extract_tags(text)
        words = set([item for sublist in tags.values() for item in sublist])
        return {tgt_field: words}

    ds = ds.map(tag_keywords, **pop_map_kwargs(kwargs))
    return ds


def filter_by_keywords(ds, **kwargs):
    min_num = kwargs.get("min_num", None)
    field = kwargs.get("field", "keywords")
    min_ratio = kwargs.get("min_ratio", None)
    skip_none = kwargs.get("skip_none", True)
    assert (min_num is not None) or (min_ratio is not None), "Either min_num or min_ratio must be set"

    def is_enough_keywords(egs):
        keywords = egs.get(field, None)
        if keywords is None:
            return not skip_none
        n_keywords = len(keywords)
        if min_num is not None and n_keywords < min_num:
            return False
        elif min_ratio is not None:
            n_words = len(set(egs["text"].split()))
            ratio = len(keywords) / (n_words + 1e-6)
            if ratio < min_ratio:
                return False
        return True

    n_egs = len(ds)
    ds = ds.filter(is_enough_keywords, **pop_filter_kwargs(kwargs), desc="Filtering keywords")
    n_left = len(ds)
    print(f"Filtered dataset: {n_egs} => {n_left} [{n_left / n_egs:.2%}] left")
    return ds


def filter_by_wer(ds, **kwargs):
    """Filter the dataset."""
    wer_range = kwargs.get("wer_range", None)
    bwer_range = kwargs.get("bwer_range", None)
    uwer_range = kwargs.get("uwer_range", None)

    def get_number(x, key, default=None):
        for name in [key, key.lower(), key.upper()]:
            if name not in x:
                continue
            return x[name]
        return default

    def is_good(val, val_range=None):
        if val_range is None or val is None:
            return True
        val_range = to_list(val_range)
        return val_range[0] <= val <= val_range[-1]

    def wer_filter_fn(x):
        good = is_good(get_number(x, "WER"), wer_range) and is_good(get_number(x, "BWER"), bwer_range) and is_good(get_number(x, "UWER"), uwer_range)
        return good

    n_egs = len(ds)
    ds = ds.filter(wer_filter_fn, **pop_filter_kwargs(kwargs))
    all_rank_print(f"Filtered dataset: {n_egs} to {len(ds)}")
    return ds


_check_field_logged = set()


def _check_field(example, field, val_range):
    val = to_float(example.get(field), default=None)
    if val is None and val_range is not None and field not in _check_field_logged:
        _check_field_logged.add(field)
        rank_print(f"[WARN] keep_samples: '{field}' missing for example {example}")
    return in_range(val, val_range)


def _is_bad_fmt(example):
    parsed = parse_asr_response(example.get("raw_response", "") or {})
    return not parsed.get("formatted", True)


def _is_bad_lang(example):
    parsed = parse_asr_response(example.get("raw_response", "") or {})
    lang = example.get("language") or "English"
    lang = get_language_name(lang).lower()
    return (parsed.get("lang") or "").lower() != lang


def _has_brackets(example):
    text = example.get("response", "")
    return has_brackets_fn(text)


def find_wrong_pieces(ref, hyp):
    """Return hyp word segments that fall inside non-equal alignment opcodes.

    Aligns ``ref`` vs ``hyp`` at the word level and returns one list of hyp
    words per insertion/substitution opcode. Pure-deletion opcodes are skipped.
    """
    ref_words = (ref or "").split()
    hyp_words = (hyp or "").split()
    from difflib import SequenceMatcher

    sm = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    pieces = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        pieces.append(hyp_words[j1:j2])
    return pieces


def _has_repeat(example, opts):
    """Check whether the error part of example[hyp_field] contains a repeated n-gram.

    Aligns the hypothesis (``hyp_field``, default ``response``) against the
    reference (``ref_field``, default ``text``) at the word level and runs
    ``has_repeat`` only on the hypothesis segments that fall inside non-equal
    opcodes (insertions/substitutions). This avoids flagging legitimately
    repeated words in the reference itself. If no reference is available, falls
    back to scanning the full hypothesis.
    """
    hyp_field = opts.get("hyp_field", "response")
    ref_field = opts.get("ref_field", "text")
    min_reps = opts.get("min_reps", 4)
    max_ngram = opts.get("max_ngram", 5)
    tn_name = opts.get("text_norm", None)
    lang_field = opts.get("lang_field", "language")
    lang = opts.get("lang", example.get(lang_field, "english"))
    hyp = example.get(hyp_field, "")
    if not hyp:
        return False
    ref = example.get(ref_field, "")
    return has_repeat_error(hyp, ref, min_reps=min_reps, max_ngram=max_ngram, tn_name=tn_name, lang=lang)


# Matches uppercase single-letter abbreviations like "U. S.", "U. P. S.",
# "U.S.", "K., F., C.,", or "C. S. V" — i.e. >=2 single uppercase letters where
# each (except possibly the last) is followed by a period and optional comma,
# separated by zero-or-more whitespace. The trailing letter's period is
# optional but the whole match must end at a word boundary so we don't grab
# parts of real words. Lowercase starts like "a.b.", "a.C.", "a. b.", "i.e.",
# "e.g.", "a.m." are intentionally excluded.
_SPACED_ABBREV_RE = re.compile(r"\b[A-Z]\.,?\s*(?:[A-Z]\.,?\s*)*[A-Z](?:\.,?)?(?!\w)")


def _has_spaced_abbrev(example, opts):
    """Check whether the error part of example[hyp_field] contains a spaced abbreviation.

    Aligns hyp (``hyp_field``, default ``response``) vs ref (``ref_field``,
    default ``text``) and runs the regex only on hyp segments inside non-equal
    opcodes (insertions/substitutions). Falls back to scanning the full hyp
    when no reference is available.
    """
    hyp_field = opts.get("hyp_field", "response")
    ref_field = opts.get("ref_field", "text")
    hyp = example.get(hyp_field, "") or ""
    if not hyp:
        return False
    ref = example.get(ref_field, "") or ""
    if not ref:
        return bool(_SPACED_ABBREV_RE.search(hyp))
    for piece in find_wrong_pieces(ref, hyp):
        if _SPACED_ABBREV_RE.search(" ".join(piece)):
            return True
    return False


def _has_tail_hallucination(example, opts):
    """Adapter: read fields from ``example`` and call shared ``has_tail_hallucination``."""
    hyp_field = opts.get("hyp_field", "response")
    ref_field = opts.get("ref_field", "text")
    min_words = opts.get("min_words", 3)
    tn_name = opts.get("text_norm", None)
    lang_field = opts.get("lang_field", "language")
    lang = opts.get("lang", example.get(lang_field, "english"))
    return has_tail_hallucination(
        example.get(hyp_field, ""),
        example.get(ref_field, ""),
        min_words=min_words,
        tn_name=tn_name,
        lang=lang,
    )


def _keyword_missing(example, opts):
    keywords_field = opts.get("keywords_field", "keywords")
    response_field = opts.get("response_field", "response")
    norm_name = opts.get("text_norm", None)
    lang_field = opts.get("lang_field", "language")
    lang = opts.get("lang", example.get(lang_field, "english"))
    return has_missing_keyword(
        example.get(keywords_field),
        example.get(response_field, ""),
        norm_name=norm_name,
        lang=lang,
    )


def _flatten_ter_category_edits(ter_category_info):
    """Flatten a ``ter_category_info`` block into ``{name: number_of_edits}``.

    Merges the top-level ``ter_categories`` (punc/cap/itn/lexical/others) and
    every ``*_subgroups`` map (punc_subgroups/cap_subgroups/itn_subgroups, etc.)
    into a single flat dict. Subgroup names are already disjoint from the
    top-level category names, so no collisions occur.
    """
    edits = {}
    info = ter_category_info or {}
    for key, group in info.items():
        if not isinstance(group, dict):
            continue
        if key != "ter_categories" and not key.endswith("_subgroups"):
            continue
        for name, stats in group.items():
            if isinstance(stats, dict):
                edits[name] = int(stats.get("number_of_edits") or 0)
    return edits


def _parse_ter_category_thresholds(categories, default_min):
    """Normalize the ``categories`` option into ``{name: min_edits}``.

    Accepts a dict ``{name: min_edits}``, a list/tuple of names (each using
    ``default_min``), or a single string name.
    """
    if categories is None:
        return {}
    if isinstance(categories, str):
        return {categories: default_min}
    if isinstance(categories, dict):
        return {str(name): int(to_int(thr, default=default_min) or default_min) for name, thr in categories.items()}
    if is_list(categories):
        return {str(name): default_min for name in categories}
    return {}


def _has_ter_category(example, opts):
    """Keep if requested TER category subgroups have at least minimal edits.

    Computes DisfluencyTolerant TER between the reference (``ref_field``,
    default ``text``) and hypothesis (``hyp_field``, default ``response``),
    then inspects the ``ter_category_info`` edit breakdown. ``categories`` maps
    a category or subgroup name to the minimum ``number_of_edits`` required;
    names may be top-level categories (``punc``, ``cap``, ``itn``, ``lexical``,
    ``others``) or fine-grained subgroups (``punc_none_2_comma``,
    ``cap_lower_2_upper``, ``itn_num``, ...). Returns True if ANY requested
    category meets its threshold (OR logic).
    """
    from recipe.phimm.reward.asr_inhouse_measure import _compute_dter, ensure_pack_dir

    ref_field = opts.get("ref_field", "text")
    hyp_field = opts.get("hyp_field", "response")
    default_min = int(to_int(opts.get("min_edits", 1), default=1) or 1)
    thresholds = _parse_ter_category_thresholds(opts.get("categories"), default_min)
    if not thresholds:
        return False

    ref = example.get(ref_field, "") or ""
    hyp = example.get(hyp_field, "") or ""
    if not ref or not hyp:
        return False

    ensure_pack_dir(opts.get("pack_dir"))
    _n_err, _n_ref, _dter, detail = _compute_dter(ref, hyp)
    if not detail:
        return False

    edits = _flatten_ter_category_edits(detail.get("ter_category_info"))
    return any(edits.get(name, 0) >= min_edits for name, min_edits in thresholds.items())


def keep_samples(
    ds,
    has_bad_fmt=None,
    has_bad_lang=None,
    has_brackets=None,
    has_repeat=None,
    has_spaced_abbrev=None,
    has_tail_hallucination=None,
    has_ter_category=None,
    keyword_missing=None,
    wer_range=None,
    error_count_range=None,
    edge_wer_range=None,
    **kwargs,
):
    """Keep samples matching ANY enabled criterion (OR logic).

    Args:
        has_bad_fmt: truthy — bad format in ASR response
        has_bad_lang: truthy — wrong language
        has_brackets: truthy — bracketed/parenthesized text
        has_repeat: truthy or dict {field, min_reps, max_ngram} — repeated n-gram
        has_spaced_abbrev: truthy or dict {field} — spaced single-letter abbreviations
            like "U. S." (should be "US") or "U. P. S." (should be "UPS")
        has_tail_hallucination: truthy or dict {hyp_field, ref_field, min_words, text_norm}
            — hyp ends with >= min_words inserted words not present in ref tail
        has_ter_category: dict {ref_field, hyp_field, min_edits, categories} — keep
            if any requested TER category/subgroup has at least ``min_edits``
            edits. ``categories`` maps a category or subgroup name (e.g.
            ``punc_none_2_comma``, ``cap_lower_2_upper``, ``itn_num``, ``punc``,
            ``lexical``) to a minimum edit count, or is a list of names that all
            use the shared ``min_edits`` default (1).
        keyword_missing: truthy or dict — keep if any keyword phrase is missing in
            response after normalization. Dict options:
            {keywords_field, response_field, norm}
        wer_range: [lo, hi] — WER range
        error_count_range: [lo, hi] — error count range
        edge_wer_range: [lo, hi] — edge WER range
    """
    filter_kwargs = pop_filter_kwargs(kwargs)
    checks = []

    if has_bad_fmt:
        checks.append(("has_bad_fmt", lambda ex: _is_bad_fmt(ex)))

    if has_bad_lang:
        checks.append(("has_bad_lang", lambda ex: _is_bad_lang(ex)))

    if has_brackets:
        checks.append(("has_brackets", lambda ex: _has_brackets(ex)))

    if has_repeat:
        repeat_opts = dict(has_repeat) if isinstance(has_repeat, dict) else {}
        checks.append(("has_repeat", lambda ex, _o=repeat_opts: _has_repeat(ex, _o)))

    if has_spaced_abbrev:
        abbrev_opts = dict(has_spaced_abbrev) if isinstance(has_spaced_abbrev, dict) else {}
        checks.append(("has_spaced_abbrev", lambda ex, _o=abbrev_opts: _has_spaced_abbrev(ex, _o)))

    if has_tail_hallucination:
        tail_opts = dict(has_tail_hallucination) if isinstance(has_tail_hallucination, dict) else {}
        checks.append(("has_tail_hallucination", lambda ex, _o=tail_opts: _has_tail_hallucination(ex, _o)))

    if has_ter_category:
        ter_opts = dict(has_ter_category) if isinstance(has_ter_category, dict) else {}
        checks.append(("has_ter_category", lambda ex, _o=ter_opts: _has_ter_category(ex, _o)))

    if keyword_missing:
        missing_opts = dict(keyword_missing) if isinstance(keyword_missing, dict) else {}
        checks.append(("keyword_missing", lambda ex, _o=missing_opts: _keyword_missing(ex, _o)))

    if wr := to_list(wer_range):
        checks.append(("wer", lambda ex, _r=wr: _check_field(ex, "wer", _r)))
    if ecr := to_list(error_count_range):
        checks.append(("n_err", lambda ex, _r=ecr: _check_field(ex, "n_err", _r)))
    if ewr := to_list(edge_wer_range):
        checks.append(("edge_wer", lambda ex, _r=ewr: _check_field(ex, "edge_wer", _r)))

    names = [n for n, _ in checks]

    def keep_fn(example):
        return any(fn(example) for _, fn in checks)

    n_egs = len(ds)
    label = ", ".join(names)
    ds = ds.filter(keep_fn, **filter_kwargs, desc=f"Keeping samples ({label})")
    n_left = len(ds)
    all_rank_print(f"Kept samples ({label}): {n_egs} => {n_left} [{n_left / n_egs if n_egs else 0.0:.2%}] left")
    return ds


# ---------------------------------------------------------------------------
# add_measures — attach per-sample scorer outputs as new columns
# ---------------------------------------------------------------------------

# Built-in scorer shortcuts: name → (module_path, function_name)
_SCORER_SHORTCUTS = {
    "inhouse": ("recipe/phimm/reward/asr_inhouse_measure.py", "eval_score"),
    "openasr": ("recipe/phimm/reward/asr_edge.py", "openasr_eval"),
    "eval": ("recipe/phimm/reward/asr_edge.py", "eval_score"),
}


def _load_scorer_fn(spec):
    """Dynamically import a scorer function from *spec*.

    *spec* is a dict with ``path`` + ``name`` (module file path and function
    name) **or** a plain string that maps to a built-in shortcut.
    Returns ``(fn, kwargs, prefix)`` ready for calling.
    """
    import importlib.util as _ilu

    if isinstance(spec, str):
        spec = {"shortcut": spec}

    shortcut = spec.get("shortcut")
    if shortcut:
        if shortcut not in _SCORER_SHORTCUTS:
            raise ValueError(f"Unknown scorer shortcut {shortcut!r}. Known: {list(_SCORER_SHORTCUTS)}")
        mod_path, fn_name = _SCORER_SHORTCUTS[shortcut]
        prefix = spec.get("prefix", shortcut)
    else:
        mod_path = spec["path"]
        fn_name = spec["name"]
        prefix = spec.get("prefix", fn_name)

    kwargs = dict(spec.get("kwargs") or {})

    # resolve relative paths to repo root
    mod_file = Path(mod_path)
    if not mod_file.is_absolute():
        mod_file = Path(__file__).parents[3] / mod_file
    if not mod_file.exists():
        raise FileNotFoundError(f"Scorer module not found: {mod_file}")

    mod_name = f"_scorer_{prefix}_{fn_name}"
    cached = sys.modules.get(mod_name)
    if cached is None:
        _spec = _ilu.spec_from_file_location(mod_name, str(mod_file))
        cached = _ilu.module_from_spec(_spec)
        sys.modules[mod_name] = cached
        _spec.loader.exec_module(cached)

    fn = getattr(cached, fn_name)
    return fn, kwargs, prefix


def add_measures(ds, scorers=None, solution_field=None, ground_truth_field=None, **kwargs):
    """Map each sample through one or more scorer functions and add their outputs.

    Config example (inside ``post_process``)::

        add_measures:
          scorers:
            - shortcut: inhouse          # prefix defaults to "inhouse"
            - shortcut: openasr
              prefix: oa                  # custom prefix
            - path: recipe/phimm/reward/asr_edge.py
              name: eval_score
              prefix: edge
              kwargs:
                text_norm: english
          solution_field: raw_response    # field containing model output
          ground_truth_field: text        # field containing reference text

    Each scorer's returned dict is merged into the sample with keys prefixed
    by ``{prefix}_``. Dict/list values (e.g. ``dter_detail``) are JSON-
    serialised to avoid Arrow schema issues across rows.
    """
    if not scorers:
        rank_print("[add_measures] No scorers configured, skipping.")
        return ds

    if isinstance(scorers, (str, dict)):
        scorers = [scorers]

    loaded = []
    for spec in scorers:
        fn, fn_kwargs, prefix = _load_scorer_fn(spec)
        loaded.append((fn, fn_kwargs, prefix))
        rank_print(f"[add_measures] Loaded scorer: {prefix} → {fn.__module__}.{fn.__name__}")

    sol_field = solution_field or "raw_response"
    gt_field = ground_truth_field or "text"

    scores_field = kwargs.pop("scores_field", "scores")

    def _measure_map(example):
        solution_str = example.get(sol_field) or example.get("response") or ""
        ground_truth = example.get(gt_field) or ""
        extra_info = {"language": example.get("language", "english")}
        groups = {}
        for fn, fn_kwargs, prefix in loaded:
            try:
                result = fn(solution_str, ground_truth, extra_info=extra_info, **fn_kwargs)
            except Exception as e:
                rank_print(f"[add_measures] scorer {prefix} failed: {e}")
                result = {}
            groups[prefix] = result or {}
        return {scores_field: json.dumps(groups, ensure_ascii=False)}

    map_kwargs = pop_map_kwargs(kwargs)
    ds = ds.map(_measure_map, **map_kwargs, desc="Adding measures")
    rank_print(f"[add_measures] Done. {len(ds)} samples, added columns from {[p for _, _, p in loaded]}")
    return ds


def filter_text_with_numbers(ds, **kwargs):
    field = kwargs.get("field", "text")
    norm_name = kwargs.get("text_norm", kwargs.get("tn_name", "identity"))

    def has_number_after_norm(example):
        text = str(example.get(field, ""))
        return not has_digit(text_norm(text, norm_name))

    n_egs = len(ds)
    ds = ds.filter(has_number_after_norm, **pop_filter_kwargs(kwargs), desc="Filtering text with numbers")
    all_rank_print(f"Filtered text with numbers after {norm_name} norm: {n_egs} to {len(ds)}")
    return ds


def stream_shuffle(ds, **kwargs):
    """Process the dataset."""
    streaming = kwargs.get("streaming", False)
    if streaming:
        num_shards = kwargs.get("num_shards", dist_state().num_processes)  # this is shared with shard_ds
        ds = ds.to_iterable_dataset(num_shards=num_shards)
    num_egs = kwargs.get("num_egs", None)
    if num_egs is not None and num_egs > len(ds):
        ds = ds.take(num_egs)
    return ds


def shard_ds(ds, **kwargs):
    """Shard the dataset."""
    num_shards = kwargs.get("num_shards", dist_state().num_processes)
    shard_id = kwargs.get("shard_id", dist_state().process_index)
    all_rank_print(f"Sharding dataset into {num_shards} shards, picking {shard_id}")
    all_rank_print("Original dataset:", ds)
    if num_shards > 1:
        ds = ds.shard(
            num_shards=num_shards,
            index=shard_id,
            contiguous=kwargs.get("contiguous", False),  # keeps a contiguous block; set False if you prefer striding
        )
    all_rank_print("Sharded dataset:", ds)
    return ds


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


def filter_long_text(ds, **kwargs):
    max_length = kwargs.get("max_length", None)
    assert max_length is not None, "max_length must be set"
    tokenizer_path = kwargs.get("tokenizer_path", None)
    assert tokenizer_path is not None, "tokenizer_path must be set"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def filter_fn(example):
        prompt = example.get("prompt", "").replace("<audio>", "")
        ids = tokenizer(prompt, truncation=False)["input_ids"]
        return len(ids) <= max_length

    ds = ds.filter(filter_fn, **pop_filter_kwargs(kwargs), desc="Filtering long text prompts")
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
        value = []
    if key == "prefix" and value is None:
        value = ""
    return value


def verl_format_ds(ds, **kwargs):
    """Format the dataset for verl training."""
    prompt_key = kwargs.get("prompt_key", "prompt")
    extra_keys = kwargs.get("extra_keys", ["id", "language", "keywords", "prefix"])

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
    """Add a new field to the dataset."""
    fields = kwargs.get("fields", {})
    assert len(fields) == 1, "Only one field can be added at a time"

    def map_fn(egs):
        return fields

    return ds.map(map_fn, **pop_map_kwargs(kwargs))


def _strip_lone_surrogates(s):
    """Drop lone UTF-8 surrogate code points so pyarrow can serialize the string.

    Some inhouse JSONL rows contain lone surrogates (e.g. ``\\udc4d``) that
    survive the JSON parse but cannot be encoded as valid UTF-8 by pyarrow.
    """
    if not isinstance(s, str):
        return s
    # Roundtrip through utf-8 with surrogatepass then drop invalid bytes on decode.
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def extract_chat(ds, **kwargs):
    """Extract ASR fields from the ``metadata`` block of inhouse chat samples.

    Each input row is a Conversation dict with a metadata block (configurable
    via ``metadata_key``, default ``"metadata"``) carrying ``audio_chunk``
    and/or ``audio_file``, ``text``, ``desc`` (entity-tagged transcription
    with ``<NE:type>`` tags) and ``whisper_language``. Output columns:
    ``id``, ``audio_path`` (from ``audio_file``), ``audio_chunk``, ``text``,
    ``desc``, ``keywords``, ``language``. All other columns are dropped.
    """
    metadata_key = kwargs.pop("metadata_key", "metadata")

    def map_fn(example):
        meta = get_value(example, metadata_key, {}) or {}
        desc = _strip_lone_surrogates((meta.get("desc") or "").strip())
        text = _strip_lone_surrogates((meta.get("text") or "").strip())
        audio_chunk = _strip_lone_surrogates(meta.get("audio_chunk") or "")
        audio_path = _strip_lone_surrogates(meta.get("audio_file") or "")
        audio_key = "audio_chunk" if audio_chunk else "audio_path"
        audio_val = audio_chunk or audio_path
        return {
            "id": _strip_lone_surrogates(example.get("id") or Path(audio_val).stem),
            audio_key: audio_val,
            "text": text,
            "desc": desc,
            "keywords": [_strip_lone_surrogates(k) for k in sorted(extract_entities(desc))] if desc else [],
            "language": _strip_lone_surrogates(meta.get("whisper_language") or "english"),
        }

    map_kwargs = pop_map_kwargs(kwargs)
    map_kwargs["remove_columns"] = list(ds.column_names)
    return ds.map(map_fn, **map_kwargs, desc="Extracting chat metadata")


def process_ds(ds, **kwargs):
    """Post process the dataset."""
    map_kwargs = pop_map_kwargs(kwargs)
    if "extract_chat" in kwargs:
        chat_kwargs = kwargs.get("extract_chat") or {}
        ds = extract_chat(ds, **merge_kwargs(map_kwargs, chat_kwargs))
    if audio_chunk_to_path_kwargs := kwargs.get("audio_chunk_to_path", {}):
        ds = audio_chunk_to_path(ds, **merge_kwargs(map_kwargs, audio_chunk_to_path_kwargs))
    if svad_explode_kwargs := kwargs.get("svad_explode", {}):
        ds = svad_explode(ds, **merge_kwargs(map_kwargs, svad_explode_kwargs))
    if input_egs_limit := kwargs.get("input_egs_limit", None):
        ds = limit_ds(ds, egs_limit=input_egs_limit)
    if filter_by_keywords_kwargs := kwargs.get("filter_by_keywords", {}):
        ds = filter_by_keywords(ds, **merge_kwargs(map_kwargs, filter_by_keywords_kwargs))
    if filter_long_text_kwargs := kwargs.get("filter_long_text", {}):
        ds = filter_long_text(ds, **merge_kwargs(map_kwargs, filter_long_text_kwargs))
    if wer_filter_kwargs := kwargs.get("filter_by_wer", {}):
        ds = filter_by_wer(ds, **merge_kwargs(map_kwargs, wer_filter_kwargs))
    if keep_kw := kwargs.get("keep_samples", {}):
        ds = keep_samples(ds, **merge_kwargs(map_kwargs, keep_kw))
    if trim_silence_kwargs := kwargs.get("trim_silence", {}):
        ds = trim_silence(ds, **merge_kwargs(map_kwargs, trim_silence_kwargs))
    if trim_tailing_kwargs := kwargs.get("trim_tailing", {}):
        ds = trim_tailing(ds, **merge_kwargs(map_kwargs, trim_tailing_kwargs))
    if filter_short_audio_kwargs := kwargs.get("filter_short_audio", {}):
        ds = filter_short_audio(ds, **merge_kwargs(map_kwargs, filter_short_audio_kwargs))
    if augment_audio_kwargs := kwargs.get("augment_audio", {}):
        ds = augment_audio(ds, **merge_kwargs(map_kwargs, augment_audio_kwargs))
    if path_map_kwargs := kwargs.get("path_map", {}):
        ds = path_map(ds, **merge_kwargs(map_kwargs, path_map_kwargs))
    if rename_fields_kwargs := kwargs.get("rename_fields", {}):
        ds = rename_fields(ds, **merge_kwargs(map_kwargs, rename_fields_kwargs))
    if kwargs.get("load_audio", False):
        ds = load_audio(ds, **map_kwargs)
    if kwargs.get("do_shard", False):
        ds = shard_ds(ds, **map_kwargs)
    if filter_text_with_numbers_kwargs := kwargs.get("filter_text_with_numbers", {}):
        ds = filter_text_with_numbers(ds, **merge_kwargs(map_kwargs, filter_text_with_numbers_kwargs))
    if output_egs_limit := kwargs.get("output_egs_limit", None):
        ds = limit_ds(ds, egs_limit=output_egs_limit)
    if add_field_kwargs := kwargs.get("add_field", {}):
        ds = add_field_ds(ds, **merge_kwargs(map_kwargs, add_field_kwargs))
    if measures_kw := kwargs.get("add_measures", {}):
        ds = add_measures(ds, **merge_kwargs(map_kwargs, measures_kw))
    if verl_format_kwargs := kwargs.get("verl_format", {}):
        ds = verl_format_ds(ds, **merge_kwargs(map_kwargs, verl_format_kwargs))
    return ds


def trunc_left_at_punc(text: str) -> str:
    """
    Truncate the string from the left at the first punctuation mark.
    Keeps the part after the punctuation, discards what’s before and including it.
    If no punctuation is found, returns the original string.
    """
    words = text.split()
    for i, word in enumerate(words):
        if word[-1] in string.punctuation:
            return " ".join(words[i + 1 :]).strip()  # cut at punctuation and strip leading spaces
    return text


def overlap_prefix(ds, **kwargs):
    """Complete the transcription for the given examples."""
    prefix_ratio = to_list(kwargs.pop("prefix_ratio", (0, 1)))
    log_interval = kwargs.get("log_interval", 10000)

    def add_overlap_prefix(egs, idx):
        words = egs["text"].split()
        ratio = random.uniform(prefix_ratio[0], prefix_ratio[-1])
        n_pfx = int(len(words) * ratio)
        prefix = " ".join(words[:n_pfx])
        prompt = f"Transcribe the audio clip into text with the prefix [{prefix}]"
        text = " ".join(words[n_pfx:])

        if idx % log_interval == 0:
            print(f"[{idx}], Prompt: {prompt}")
            print(f"[{idx}], Text  : {text}")

        return {
            "text": text,
            "prompt": prompt_format.format(prompt),
        }

    ds = ds.map(add_overlap_prefix, with_indices=True, **pop_map_kwargs(kwargs))
    return ds


def add_task_info(ds, **kwargs):
    """Add a prompt to the dataset."""
    task = kwargs.get("task", "asr")
    rand = kwargs.get("rand", False)
    language = kwargs.get("language", "English")
    model_version = kwargs.get("model_version")
    prompt_suffix = kwargs.get("prompt_suffix", "")
    prefix_prob = float(kwargs.get("prefix_prob", 0.0))

    def add_task_info_fn(egs):
        lang = resolve_task_language(task, lang=egs.get("language") or language)
        prompt = get_task_prompt(task=task, rand=rand)
        prompt = f"{prompt}{prompt_suffix}"
        prefix = get_task_prefix(task, lang=lang, prob=prefix_prob)
        gt_output = get_task_output(
            task=task,
            lang=lang,
            text=egs.get("text", ""),
            components=egs.get("components"),
        )
        return {
            "prompt": format_asr_prompt(prompt, model_version=model_version),
            "prefix": prefix,
            "gt_output": gt_output,
            "language": lang,
        }

    ds = ds.map(add_task_info_fn, **pop_map_kwargs(kwargs))
    return ds


def context_prefix(ds, **kwargs):
    """Complete the transcription for the given examples."""
    prefix_key = kwargs.get("prefix_key", "info.preceding_original_transcription")
    prefix_range = to_list(kwargs.get("prefix_range", (0, 100)))
    log_interval = kwargs.get("log_interval", 10000)

    def add_context_prefix(egs, idx):
        pfx_words = get_value(egs, prefix_key, "").strip().split()
        n_pfx = random.randint(prefix_range[0], prefix_range[-1])
        prefix = trunc_left_at_punc(" ".join(pfx_words[-n_pfx:]))
        if prefix:
            prompt = f"Transcribe the audio clip into text with the prefix: \n{prefix}\n"
        else:
            prompt = "Transcribe the audio clip into text."

        if idx % log_interval == 0:
            print(f"[{idx}], Prompt: {prompt}")
            print(f"[{idx}], Text  : {egs['text']}")
        return {"prompt": prompt_format.format(prompt)}

    ds = ds.map(add_context_prefix, with_indices=True, **pop_map_kwargs(kwargs))
    return ds


def num_gpus():
    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1

    # def tag_entity(ds, **kwargs):
    #     """Tag named entities in the transcription."""
    #     src_field = kwargs.get("src_field", "text")
    #     tgt_field = kwargs.get("tgt_field", "keywords")
    #     model_path = kwargs.get("model_path", None)
    #     assert model_path is not None, "model_path must be set for NER model"
    #     num_actors = kwargs.get("num_proc", None) or num_gpus()
    #     bs = kwargs.get("batch_size", 1000)
    #     local_model_path = cache_dir(model_path)
    #     print(f"Using NER model: {model_path} [{local_model_path}] with {num_actors} actors, {bs} batch size")
    #     ds = ner_ds(
    #         ds=ds,
    #         model_id=local_model_path,
    #         src_field=src_field,
    #         tgt_field=tgt_field,
    #         bs=bs,
    #         n_actors=num_actors,
    #     )
    #     return ds


def augment(ds, **kwargs):
    """Augment the dataset with additional information."""
    map_kwargs = pop_map_kwargs(kwargs)
    if pre_process_kwargs := kwargs.get("pre_process", {}):
        ds = process_ds(ds, **merge_kwargs(map_kwargs, pre_process_kwargs))
    if overlap_prefix_kwargs := kwargs.get("overlap_prefix", {}):
        ds = overlap_prefix(ds, **merge_kwargs(map_kwargs, overlap_prefix_kwargs))
    if context_prefix_kwargs := kwargs.get("context_prefix", {}):
        ds = context_prefix(ds, **merge_kwargs(map_kwargs, context_prefix_kwargs))
    if biasing_kwargs := kwargs.get("biasing", {}):
        ds = bias_sampling(ds, **merge_kwargs(map_kwargs, biasing_kwargs))
    if pref_kwargs := kwargs.get("simu_preference", {}):
        ds = simulate_preference(ds, **merge_kwargs(map_kwargs, pref_kwargs))
    if fmt_pref_kwargs := kwargs.get("format_preference", {}):
        ds = format_preference(ds, **merge_kwargs(map_kwargs, fmt_pref_kwargs))
    if add_rare_keywords_kwargs := kwargs.get("add_rare_keywords", {}):
        ds = add_rare_keywords(ds, **merge_kwargs(map_kwargs, add_rare_keywords_kwargs))
    if add_tag_keywords_kwargs := kwargs.get("add_tag_keywords", {}):
        ds = add_tag_keywords(ds, **merge_kwargs(map_kwargs, add_tag_keywords_kwargs))
    # if tag_entity_kwargs := kwargs.get("tag_entity", {}):
    #     ds = tag_entity(ds, **merge_kwargs(map_kwargs, tag_entity_kwargs))
    if add_task_info_kwargs := kwargs.get("add_task_info", {}):
        ds = add_task_info(ds, **merge_kwargs(map_kwargs, {"model_version": kwargs.get("model_version")}, add_task_info_kwargs))
    if post_process_kwargs := kwargs.get("post_process", {}):
        ds = process_ds(ds, **merge_kwargs(map_kwargs, post_process_kwargs))
    return ds


def cache_ds(**kwargs):
    cache_name = kwargs.get("cache_name", None)
    if cache_name is None:
        return None, None
    if cache_name.startswith("auto"):
        cache_name = f"{socket.gethostname()}{cache_name[4:]}_{hash_id(kwargs)}"
    cache_dir = kwargs.get("cache_dir", Path().home() / "data/cache_datasets")
    cache_path = Path(cache_dir) / cache_name
    ds = load_cached_ds(cache_path)
    return ds, cache_path


def load_cached_ds(cache_path):
    if not cache_path:
        return None
    try:
        # local_path = cache_dir(str(cache_path))
        fs_path, options = get_path_with_options(str(cache_path))
        rank_print(f"Loading cached dataset from {cache_path}")
        return Dataset.load_from_disk(fs_path, storage_options=options)
    except Exception as e:
        rank_print(f"Cache not found or invalid at {cache_path}. Error: {e}")
        return None


def save_cached_ds(ds, cache_path):
    if not cache_path:
        return
    rank_print(f"Saving dataset to cache at {cache_path}")
    rel_path, options = get_path_with_options(str(cache_path))
    ds.save_to_disk(rel_path, storage_options=options)


def create_audio_dataset(**kwargs):
    """Create a dataset from the given split."""
    ds_name = kwargs.get("dataset_name", "unknown").lower()
    with dist_state().local_main_process_first():
        ds, cache_path = cache_ds(**kwargs)
        if ds is not None:
            return ds
        if ds_name == "ls_bias":
            ds = ls_bias_dataset(**kwargs)
        elif ds_name == "inhouse_entity":
            ds = entity_dataset(**kwargs)
        elif ds_name == "openasr":
            ds = openasr_dataset(**kwargs)
        elif ds_name == "tsv":
            ds = tsv_dataset(**kwargs)
        elif ds_name == "jsonl":
            ds = jsonl_dataset(**kwargs)
        elif ds_name == "parquet":
            ds = parquet_dataset(**kwargs)
        elif ds_name == "chunk":
            ds = chunk_dataset(**kwargs)
        elif ds_name == "audio_dir":
            ds = audio_dir_dataset(**kwargs)
        elif ds_name == "cached":
            ds = load_cached_ds(kwargs.get("cache_path", None))
            assert ds is not None, "Cached dataset not found."
        else:
            raise ValueError(f"Unknown dataset name: {ds_name}")
        ds = augment(ds, **kwargs)
        save_cached_ds(ds, cache_path)
    return ds


def create_datasets(config):
    """Create dataset."""
    if config is None:
        return None
    if is_list(config):
        datasets = {}
        for i, cfg in enumerate(config):
            nickname = cfg.pop("nickname", f"dataset_{i}")
            datasets[nickname] = create_audio_dataset(**cfg)
        return datasets
    elif isinstance(config, dict):
        return create_audio_dataset(**config)
    raise ValueError("Unsupported dataset config type. Expected dict or list of dicts.")


# %%
if __name__ == "__main__":
    # Example usage
    # dataset = create_dataset(name="openasr", head=2)
    # print(dataset)
    # print(dataset[0]["text"])
    # yaml_file = "recipe/phimm/config/data/train_data/local_debug.yaml"
    yaml_file = "recipe/phimm/config/data/train_data/local_debug_chunk.yaml"
    # yaml_file = "recipe/phimm/config/data/train_data/local_debug_parquet.yaml"
    # yaml_file = "recipe/phimm/config/data/train_data/local_debug_audio_dir.yaml"
    import yaml

    kwargs = yaml.safe_load(Path(yaml_file).read_text())
    ds = create_datasets(kwargs)
    print(ds)
    for egs in ds.take(5):
        print(egs)

# %%

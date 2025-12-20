# %%
import os
import re
from collections import defaultdict
import ast
import random
import socket
import blobfile as bf
import pandas as pd
import string
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3]))
from datasets import load_dataset, concatenate_datasets, Dataset
from bs4 import BeautifulSoup
from recipe.phimm.data.error_simu import ErrorSimulator
from recipe.phimm.data.biasing import PieceSampler, tag_pieces, text_norm as biasing_text_norm
from recipe.phimm.data.prompts import get_task_prompt
from recipe.phimm.utils.tn import text_norm
from recipe.phimm.data.chunk import get_chunk_manager, create_chunk_datasets
from recipe.phimm.utils.shared import (
    hash_id,
    get_value,
    rank_print,
    dist_state,
    all_rank_print,
    to_list,
    is_list,
    to_int,
)
from recipe.phimm.utils.audio import sf_read
from recipe.phimm.utils.storage import get_path_with_options

prompt_format = "<|user|><|audio_1|>{}<|end|><|assistant|>"


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
    """Load a JSONL dataset from the specified paths."""

    data_files = [jsonl_paths] if isinstance(jsonl_paths, str) else jsonl_paths
    data_files = [get_path_with_options(str(file_path)) for file_path in data_files]

    options = data_files[0][1]
    fs_files = [file[0] for file in data_files]
    ds = load_dataset("json", data_files=fs_files, split="train", storage_options=options)
    ds = stream_shuffle(ds, **kwargs)
    return ds


def parquet_dataset(parquet_paths, **kwargs):
    """Load a Parquet dataset from the specified paths."""

    data_files = [parquet_paths] if isinstance(parquet_paths, str) else parquet_paths
    data_files = [get_path_with_options(str(file_path)) for file_path in data_files]

    options = data_files[0][1]
    fs_files = [file[0] for file in data_files]
    ds = load_dataset("parquet", data_files=fs_files, split="train", storage_options=options)
    return ds


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
        # n_proc = max(dist_state().num_processes, 2)
        n_proc = 8
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


def load_tsv(tsv_file, **kwargs):
    """Load a TSV file into a dataset."""
    # breakpoint()

    fs_path, options = get_path_with_options(tsv_file)
    ds = load_dataset(
        "csv",
        data_files=fs_path,
        split="train",
        delimiter="\t",
        column_names=["id", "paths", "msgs"],
        storage_options=options,
    )
    return ds


def tsv_dataset(tsv_paths, **kwargs):
    """Create a dataset from the given split."""
    if is_list(tsv_paths):
        ds = concatenate_datasets([load_tsv(tsv_path, **kwargs) for tsv_path in tsv_paths])
    else:
        ds = load_tsv(tsv_paths, **kwargs)

    ds = stream_shuffle(ds, **kwargs)

    def load_sample(egs):
        """Process a single sample."""
        audio_path = ast.literal_eval(egs["paths"])[0]
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
        prompt = example.get("prompt", "").replace("<|audio_1|>", "")
        ids = tokenizer(prompt, truncation=False)["input_ids"]
        return len(ids) <= max_length

    ds = ds.filter(filter_fn, **pop_filter_kwargs(kwargs), desc="Filtering long text prompts")
    return ds


def to_user_msg(prompt):
    if not isinstance(prompt, str):
        return prompt

    for word in ["<|user|>", "<|end|>", "<|assistant|>"]:
        prompt = prompt.replace(word, "")
    return [{"role": "user", "content": prompt}]


def verl_format_ds(ds, **kwargs):
    """Format the dataset for verl training."""
    prompt_key = kwargs.get("prompt_key", "prompt")
    extra_keys = kwargs.get("extra_keys", ["id", "keywords"])

    def map_fn(egs):
        return {
            prompt_key: to_user_msg(egs[prompt_key]),
            "reward_model": {"ground_truth": egs.get("text", "")},
            "extra_info": {key: egs.get(key, None) for key in extra_keys},
            "data_source": egs.get("data_source", "asr"),
        }

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


def process_ds(ds, **kwargs):
    """Post process the dataset."""
    map_kwargs = pop_map_kwargs(kwargs)
    if input_egs_limit := kwargs.get("input_egs_limit", None):
        ds = limit_ds(ds, egs_limit=input_egs_limit)
    if filter_by_keywords_kwargs := kwargs.get("filter_by_keywords", {}):
        ds = filter_by_keywords(ds, **merge_kwargs(map_kwargs, filter_by_keywords_kwargs))
    if filter_long_text_kwargs := kwargs.get("filter_long_text", {}):
        ds = filter_long_text(ds, **merge_kwargs(map_kwargs, filter_long_text_kwargs))
    if wer_filter_kwargs := kwargs.get("filter_by_wer", {}):
        ds = filter_by_wer(ds, **merge_kwargs(map_kwargs, wer_filter_kwargs))
    if path_map_kwargs := kwargs.get("path_map", {}):
        ds = path_map(ds, **merge_kwargs(map_kwargs, path_map_kwargs))
    if rename_fields_kwargs := kwargs.get("rename_fields", {}):
        ds = rename_fields(ds, **merge_kwargs(map_kwargs, rename_fields_kwargs))
    if kwargs.get("load_audio", False):
        ds = load_audio(ds, **map_kwargs)
    if kwargs.get("do_shard", False):
        ds = shard_ds(ds, **map_kwargs)
    if output_egs_limit := kwargs.get("output_egs_limit", None):
        ds = limit_ds(ds, egs_limit=output_egs_limit)
    if add_field_kwargs := kwargs.get("add_field", {}):
        ds = add_field_ds(ds, **merge_kwargs(map_kwargs, add_field_kwargs))
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


def add_prompt(ds, **kwargs):
    """Add a prompt to the dataset."""
    task = kwargs.get("task", "asr")
    rand = kwargs.get("rand", False)
    forced = kwargs.get("forced", False)

    def add_prompt_fn(egs):
        prompt = egs.get("prompt", None)
        if forced or prompt is None:
            prompt_txt = get_task_prompt(task=task, rand=rand)
            prompt = prompt_format.format(prompt_txt)
        return {"prompt": prompt}

    ds = ds.map(add_prompt_fn, **pop_map_kwargs(kwargs))
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
    if add_prompt_kwargs := kwargs.get("add_prompt", {}):
        ds = add_prompt(ds, **merge_kwargs(map_kwargs, add_prompt_kwargs))
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

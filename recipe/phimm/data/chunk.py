# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import random
from pathlib import Path
from math import ceil
from functools import partial
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import soundfile as sf
from datasets import Dataset
from cachetools import FIFOCache, cached
import blobfile as bf
from recipe.phimm.utils.shared import rank_print, get_values, to_list
from verl.audio_cache import localize_audio_source, resolve_audio_source


def resolve_path(path, prefix=None):
    if not isinstance(path, str):
        return path
    import re
    az_prefix = prefix or "az://orngwus2cresco/data/speech/"
    path = re.sub(r"^/datablob1?/", az_prefix, path)
    path = re.sub(r"^az://(orngcresco|orngscuscresco|orngwus2cresco)/", "az://orngwus2cresco/", path)
    return path


def parse_data(data, data_type, **kwargs):
    if data_type.lower() == "audio":
        return sf.read(io.BytesIO(data))
    if data_type.lower() in ["info", "sft", "alignment"]:
        return json.loads(str(data, "utf-8"))
    if data_type.lower() == "feature":
        feat = np.frombuffer(data, dtype=np.float32)
        return feat.reshape(-1, kwargs.get("feat_dim", 80))
    return str(data, "utf-8")


class ChunkLoader:
    def __init__(self, chunk_path, chunk_type, count):
        self.chunk_path = chunk_path
        self.chunk_type = chunk_type
        self.count = count
        self._examples = {}  # Will be loaded on demand

    def __repr__(self):
        return f"<ChunkLoader({self.chunk_path}, {self.chunk_type}, {self.count})>"

    def __len__(self):
        """Return the number of examples in the chunk."""
        return self.count

    def get(self, i):
        """Get the example at the specified index."""
        if i < 0 or i >= self.count:
            raise IndexError(f"Index {i} out of bounds for chunk with count {self.count}.")
        if i not in self._examples:
            self._examples = self._load_examples()
        egs = self._examples.pop(i)
        return egs

    def _load_examples(self):
        """Load examples for the given index."""
        rank_print(f"Loading all examples for chunk {self.chunk_path}.")
        examples = load_data_from_chunk(self.chunk_path, self.chunk_type, self.count)
        return dict(enumerate(examples))


MAX_CACHED_LOADERS = 10


class ChunkManager:
    """Manager for loading and managing chunks of data."""

    def __init__(self, maxsize=None):
        """Initialize the ChunkManager with a maximum size for the cache."""
        maxsize = maxsize or MAX_CACHED_LOADERS
        rank_print(f"Initializing ChunkManager with max {maxsize} ChunkLoaders.")
        self.chunk_loaders = FIFOCache(maxsize=maxsize)

    def get(self, chunk_path, count, chunk_type=None):
        """Get the example at the specified index."""
        if chunk_path not in self.chunk_loaders:
            chunk_type = chunk_type or chunk_path.split(".")[-1]
            self.chunk_loaders[chunk_path] = ChunkLoader(chunk_path, chunk_type, count)
        return self.chunk_loaders[chunk_path]


def get_chunk_manager(maxsize=None):
    """Return the global singleton ChunkManager."""
    global _chunk_manager_instance
    try:
        return _chunk_manager_instance
    except NameError:
        _chunk_manager_instance = ChunkManager(maxsize)
        return _chunk_manager_instance


def get_chunk_type_path(chunk, chunk_type):
    if chunk_type in chunk:
        return chunk[chunk_type]
    chunk_type_path = f"{chunk_type}_path"
    if chunk_type_path in chunk:
        return chunk[chunk_type_path]
    type_mapping = {
        "transcription": "trans_path",
    }
    if chunk_type in type_mapping and type_mapping[chunk_type] in chunk:
        return chunk[type_mapping[chunk_type]]
    return chunk.get("chunk_path", None)


def to_records(d):
    """Convert a dict of lists to a list of dict."""
    return [dict(zip(d.keys(), to_list(vs))) for vs in zip(*d.values())]


def load_examples(chunk, fields):
    examples = {}
    count = chunk["count"]
    for field in fields:
        parts = field.split(".")
        chunk_type = parts[0]
        chunk_file = get_chunk_type_path(chunk, chunk_type).rstrip("/") + f"/{chunk['name']}.{chunk_type}"
        chunk_file = resolve_path(chunk_file)
        chunk_file = resolve_audio_source(chunk_file)
        if not bf.exists(chunk_file):
            rank_print(f"Skip [{chunk_file}] due to missing.")
            return {}
        if field in ("audio", "audios"):
            examples["audio_chunk"] = [f"{chunk_file}:{count}:{i}" for i in range(count)]
        else:
            data_list = load_data_from_chunk(chunk_file, chunk_type, count)
            if sub_field := ".".join(parts[1:]):
                data_list = get_values(data_list, sub_field)
            if any(v is None for v in data_list):
                rank_print(f"Skip [{chunk_file}] due to missing field {field}.")
                return {}
            examples[field] = data_list

    examples["language"] = [chunk.get("language", None)] * count

    return examples


def load_examples_from_chunks(chunks, types):
    # Determine expected keys from types so empty batches still return a valid schema
    expected_keys = []
    for field in types:
        if field in ("audio", "audios"):
            expected_keys.append("audio_chunk")
        else:
            expected_keys.append(field)
    expected_keys.append("language")

    all_examples = {k: [] for k in expected_keys}
    chunks = to_records(chunks)
    for chunk in chunks:
        examples = load_examples(chunk, types)
        for key, value in examples.items():
            all_examples[key].extend(value)
    return all_examples


def load_data_from_chunk(chunk_path: str, chunk_type: str, chunk_size: int):
    ENDIAN = "little"
    data_list = []
    chunk_path = localize_audio_source(chunk_path)
    with bf.BlobFile(chunk_path, "rb") as f:
        target_type = f.read(len(chunk_type.encode())).decode()
        if chunk_type.lower() != target_type.lower():
            raise ValueError(f"Target type is not expected in {chunk_path}, expected {chunk_type}, but got {target_type}")
        _ = int.from_bytes(f.read(4), byteorder=ENDIAN)
        for i in range(chunk_size):
            egs_i = int.from_bytes(f.read(4), byteorder=ENDIAN)
            if egs_i != i:
                raise ValueError(f"The example index is corrupted in {chunk_path}, expected {i}, but got {egs_i}")
            if target_type.lower() == "audios":
                parsed_data = []
                n_audios = int.from_bytes(f.read(4), byteorder=ENDIAN)
                for i in range(n_audios):
                    data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                    data = f.read(data_size)
                    parsed_data.append(parse_data(data, "audio"))
            else:
                data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if target_type.lower() == "label":
                    data_size = int.from_bytes(f.read(2), byteorder=ENDIAN)
                data = f.read(data_size)
                parsed_data = parse_data(data, chunk_type)
            data_list.append(parsed_data)
    return data_list


def load_chunk_sample(chunk_path):
    """Load a single example from a chunk file, seeking past preceding entries.

    Accepts the same ``"file:count:index"`` format as :func:`load_chunk_example`.
    """
    chunk_path = resolve_path(chunk_path)
    chunk_file, chunk_count, chunk_index = chunk_path.rsplit(":", 2)
    chunk_file = localize_audio_source(chunk_file)
    index = int(chunk_index)
    chunk_type = chunk_file.split(".")[-1]
    ENDIAN = "little"
    with bf.BlobFile(chunk_file, "rb") as f:
        target_type = f.read(len(chunk_type.encode())).decode()
        if chunk_type.lower() != target_type.lower():
            raise ValueError(f"Type mismatch in {chunk_file}: expected {chunk_type}, got {target_type}")
        f.read(4)  # skip version
        for i in range(index + 1):
            egs_i = int.from_bytes(f.read(4), byteorder=ENDIAN)
            if egs_i != i:
                raise ValueError(f"Corrupted index in {chunk_file}: expected {i}, got {egs_i}")
            if target_type.lower() == "audios":
                n_audios = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if i == index:
                    return [parse_data(f.read(int.from_bytes(f.read(4), byteorder=ENDIAN)), "audio") for _ in range(n_audios)]
                for _ in range(n_audios):
                    f.seek(int.from_bytes(f.read(4), byteorder=ENDIAN), 1)
            else:
                data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if target_type.lower() == "label":
                    data_size = int.from_bytes(f.read(2), byteorder=ENDIAN)
                if i == index:
                    return parse_data(f.read(data_size), chunk_type)
                f.seek(data_size, 1)


def load_chunk_info(manifest_file, **kwargs):
    assert bf.exists(manifest_file), f"Chunk info file {manifest_file} does not exist."
    with bf.BlobFile(manifest_file, "r") as f:
        chunk_info = json.load(f)
    return [{**chunk, **kwargs} for chunk in chunk_info["fileInfo"]]


def load_specs(spec_files):
    specs = []
    for spec_file in to_list(spec_files):
        with bf.BlobFile(spec_file, "r") as f:
            spec_dict = json.load(f)
        language = spec_dict.get("language")
        for ds in spec_dict["data_sources"]:
            ds = {key: resolve_path(value) for key, value in ds.items()}
            ds["language"] = ds.get("language", language)
            specs.append(ds)
    return specs


def load_chunks(specs, chunks_per_spec=None):
    if isinstance(specs[0], str):  # specs is list of text # assume spec files
        specs = load_specs(specs)
    chunks = []
    rank_print(f"Loading chunks from {len(specs)} specs.")
    for spec in tqdm(specs, desc="Loading Specs"):
        chunks += load_chunk_info(**spec)[:chunks_per_spec]
    rank_print(f"Loaded {len(chunks)} chunks.", f"Max chunks per spec: {chunks_per_spec}.")
    return chunks


def limit_chunks(chunks, max_egs=None, max_chunks=None):
    """Limit the number of chunks to max_chunks."""
    n_chunks = len(chunks)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        rank_print(f"Limiting chunks {n_chunks} -> {len(chunks)} by max_chunks={max_chunks}.")
    if max_egs is not None:
        new_chunks = []
        total_egs = 0
        for chunk in chunks:
            if total_egs >= max_egs:
                break
            new_chunks.append(chunk)
            total_egs += chunk["count"]
        rank_print(f"Limiting chunks {n_chunks} -> {len(new_chunks)} by max_egs={max_egs}.")
        chunks = new_chunks
    return chunks


def generate_examples(specs, chunk_types=None, chunk_shuffle=True, max_chunks=None, max_egs=None):
    """Generate examples from the chunk dataset based on the specification files."""
    chunks_per_spec = ceil(max_chunks / len(specs)) if max_chunks else None
    chunks = load_chunks(specs, chunks_per_spec)
    chunks = limit_chunks(chunks, max_egs, max_chunks)

    if chunk_shuffle:
        random.shuffle(chunks)
    types = to_list(chunk_types or ["audio", "transcription"])
    for chunk in tqdm(chunks, desc="Loading Chunks"):
        examples = load_examples(chunk, types)
        yield from to_records(examples)


def chunks2dataset(chunks, chunk_types=None, num_proc=None, streaming=False):
    """Convert a list of chunks to a Dataset object."""
    types = to_list(chunk_types or ["audio", "transcription"])
    ds = Dataset.from_list(chunks)
    map_kwargs = {
        "batched": True,
        "batch_size": 10,
        "num_proc": num_proc,
        "remove_columns": ds.column_names,
    }
    if streaming:
        ds = ds.to_iterable_dataset()
        map_kwargs.pop("num_proc", None)

    ds = ds.map(
        partial(load_examples_from_chunks, types=types),
        **map_kwargs,
    )
    return ds


def create_chunk_datasets(
    specs,
    chunk_types=None,
    chunk_shuffle=True,
    max_chunks=None,
    max_egs=None,
    streaming=False,
    num_proc=None,
    **kwargs,
):
    chunks_per_spec = ceil(max_chunks / len(specs)) if max_chunks else None
    chunks = load_chunks(specs, chunks_per_spec)
    chunks = limit_chunks(chunks, max_egs, max_chunks)
    if chunk_shuffle:
        random.shuffle(chunks)
    chunk_types = to_list(chunk_types or ["audio", "transcription"])
    return chunks2dataset(chunks, chunk_types, num_proc, streaming)


@cached(FIFOCache(maxsize=5))
def load_chunk_example(chunk_path):
    """Load a single example from the chunk file."""
    chunk_path = resolve_path(chunk_path)
    chunk_file, chunk_count, chunk_index = chunk_path.rsplit(":", 2)  # make sure rsplit.
    chunk_file = localize_audio_source(chunk_file)
    chunk_loader = get_chunk_manager().get(chunk_file, int(chunk_count))
    return chunk_loader.get(int(chunk_index))


# %%
if __name__ == "__main__":
    spec_file = [
        # "/datablob1/users/ruchaofan/DataSpecs/mlang_s2/asr_person_filtered/asr_chunk_inhouse_en.json",
        "/home/boren/data/inhouse/data_spec/local_entity_debug.json"
    ]
    # dataset = ChunkDataset(spec_file)
    # for i in range(0, 100, 10):  # Print every 10th sample, 50 samples in each chunk
    #     sample = dataset[i]
    #     rank_print(f"Sample {i}: {sample}")  # Output the sample data
    # pass
    output_dir = Path("/home/boren/data/inhouse/data_spec/local_entity_debug_output")
    example_jsonl_file = output_dir / "trans.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_types = ["audio", "transcription"]
    with open(example_jsonl_file, "w") as jf:
        for i, example in enumerate(generate_examples(spec_file, chunk_types=chunk_types, max_chunks=2)):
            rank_print(f"Example {i}: {example}")
            data, fs = load_chunk_sample(example["audio_chunk"])
            wav_name = f"egs_{i}.wav"
            wav_path = output_dir / wav_name
            with open(wav_path, "wb") as f:
                sf.write(f, data, fs)
            print("Wav:", wav_path)
            jf.write(f"{wav_name}\t{example['transcription']}\n")
            if i > 10:
                break

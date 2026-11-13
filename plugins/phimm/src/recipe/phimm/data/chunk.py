# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json

import blobfile as bf
import numpy as np
import soundfile as sf
from cachetools import FIFOCache, cached

from recipe.phimm.utils.shared import rank_print
from verl.audio_cache import localize_audio_source


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


def load_data_from_chunk(chunk_path: str, chunk_type: str, chunk_size: int):
    ENDIAN = "little"
    data_list = []
    chunk_path = localize_audio_source(chunk_path)
    with bf.BlobFile(chunk_path, "rb") as f:
        target_type = f.read(len(chunk_type.encode())).decode()
        if chunk_type.lower() != target_type.lower():
            raise ValueError(
                f"Target type is not expected in {chunk_path}, expected {chunk_type}, but got {target_type}"
            )
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
                    return [
                        parse_data(f.read(int.from_bytes(f.read(4), byteorder=ENDIAN)), "audio")
                        for _ in range(n_audios)
                    ]
                for _ in range(n_audios):
                    f.seek(int.from_bytes(f.read(4), byteorder=ENDIAN), 1)
            else:
                data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if target_type.lower() == "label":
                    data_size = int.from_bytes(f.read(2), byteorder=ENDIAN)
                if i == index:
                    return parse_data(f.read(data_size), chunk_type)
                f.seek(data_size, 1)


@cached(FIFOCache(maxsize=5))
def load_chunk_example(chunk_path):
    """Load a single example from the chunk file."""
    chunk_path = resolve_path(chunk_path)
    chunk_file, chunk_count, chunk_index = chunk_path.rsplit(":", 2)  # make sure rsplit.
    chunk_file = localize_audio_source(chunk_file)
    chunk_loader = get_chunk_manager().get(chunk_file, int(chunk_count))
    return chunk_loader.get(int(chunk_index))


# %%

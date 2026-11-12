"""Bounded offset indexes for sample reads; never cache decoded audio chunks."""

import os
from functools import lru_cache

import blobfile as bf

from recipe.phimm.data.chunk import parse_data, resolve_path
from verl.audio_cache import localize_audio_source


def _uint(stream, width=4):
    value = stream.read(width)
    if len(value) != width:
        raise ValueError("Truncated chunk header")
    return int.from_bytes(value, "little")


@lru_cache(maxsize=32)
def _offsets(path, count, kind, identity):
    offsets = []
    with bf.BlobFile(path, "rb") as stream:
        if stream.read(len(kind)).decode().lower() != kind.lower():
            raise ValueError(f"Chunk type mismatch: {path}")
        _uint(stream)  # version, as in the original sample reader
        for index in range(count):
            if _uint(stream) != index:
                raise ValueError(f"Corrupted chunk index {index}: {path}")
            entries = []
            if kind.lower() == "audios":
                for _ in range(_uint(stream)):
                    size = _uint(stream)
                    entries.append((stream.tell(), size))
                    stream.seek(size, 1)
            else:
                size = _uint(stream)
                if kind.lower() == "label":
                    size = _uint(stream, 2)
                entries.append((stream.tell(), size))
                stream.seek(size, 1)
            if stream.tell() > identity[2]:
                raise ValueError(f"Truncated chunk payload: {path}")
            offsets.append(tuple(entries))
    return tuple(offsets)


def load_chunk_sample_indexed(spec):
    """Return the same record as load_chunk_sample using a cached byte index."""
    path, count, index = resolve_path(spec).rsplit(":", 2)
    path = localize_audio_source(path)
    count, index = int(count), int(index)
    if not 0 <= index < count:
        raise ValueError(f"Chunk index out of range: {spec}")
    kind = path.rsplit(".", 1)[-1]
    if "://" in path:
        stat = bf.stat(path)
        identity = (stat.version, stat.mtime, stat.size, stat.md5)
    else:
        stat = os.stat(path)
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    entries = _offsets(path, count, kind, identity)[index]
    values = []
    with bf.BlobFile(path, "rb") as stream:
        for offset, size in entries:
            stream.seek(offset)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"Truncated chunk payload: {path}")
            values.append(parse_data(payload, "audio" if kind.lower() == "audios" else kind))
    return values if kind.lower() == "audios" else values[0]

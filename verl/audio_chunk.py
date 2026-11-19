"""Byte references to encoded audio payloads inside audio/audios chunks."""

import io
import re
from contextlib import contextmanager

_REFERENCE = re.compile(r"(.+\.audios?)!(\d+):([1-9]\d*)", re.IGNORECASE)


def split_chunk_audio_source(source: str) -> tuple[str, int, int] | None:
    """Parse ``chunk.audio!byte_offset:byte_length`` without time selectors."""
    match = _REFERENCE.fullmatch(source)
    if match:
        return match[1], int(match[2]), int(match[3])
    if re.search(r"\.audios?!", source, re.IGNORECASE):
        raise ValueError(f"Invalid chunk byte reference: {source!r}")
    return None


def read_chunk_audio(source: str) -> bytes:
    """Read an encoded payload with a bounded range request or local seek."""
    import blobfile as bf

    from verl.audio_cache import resolve_audio_source

    reference = split_chunk_audio_source(source)
    if reference is None:
        raise ValueError(f"Expected a chunk byte reference: {source!r}")
    path, offset, size = reference
    path = resolve_audio_source(path)
    with bf.BlobFile(path, "rb", streaming=True, buffer_size=min(size, 8 * 1024 * 1024)) as stream:
        stream.seek(offset)
        data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"Truncated chunk payload {source!r}: expected {size} bytes, got {len(data)}")
    return data


class _BoundedAudioFile(io.RawIOBase):
    """Expose one embedded audio record as an independent seekable file."""

    def __init__(self, stream, offset, size):
        self.stream = stream
        self.offset = offset
        self.size = size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, min(self.size, base + offset))
        return self.position

    def readinto(self, buffer):
        count = min(len(buffer), self.size - self.position)
        if count <= 0:
            return 0
        self.stream.seek(self.offset + self.position)
        data = self.stream.read(count)
        if len(data) != count:
            raise EOFError("Truncated embedded audio record")
        buffer[:count] = data
        self.position += count
        return count


@contextmanager
def open_chunk_audio(source: str):
    """Seek within an embedded audio record without reading the whole record."""
    import blobfile as bf

    from verl.audio_cache import resolve_audio_source

    reference = split_chunk_audio_source(source)
    if reference is None:
        raise ValueError(f"Expected a chunk byte reference: {source!r}")
    path, offset, size = reference
    with bf.BlobFile(resolve_audio_source(path), "rb", streaming=True, buffer_size=min(size, 512 * 1024)) as f:
        with _BoundedAudioFile(f, offset, size) as record:
            yield record

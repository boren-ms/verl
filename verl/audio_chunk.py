"""Byte references to encoded audio payloads inside audio/audios chunks."""

import re

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

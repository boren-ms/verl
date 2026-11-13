"""Direct references to uncompressed ZIP members, as described by Bitbank."""

import re

_ZIP_REFERENCE = re.compile(r"(.+\.zip)!(\d+):([1-9]\d*)", re.IGNORECASE)


def split_zip_audio_source(source: str) -> tuple[str, int, int] | None:
    """Parse ``archive.zip!data_offset:byte_length`` (without time selectors)."""
    match = _ZIP_REFERENCE.fullmatch(source)
    if match:
        return match[1], int(match[2]), int(match[3])
    if ".zip!" in source.lower():
        raise ValueError(f"Invalid ZIP audio reference: {source!r}")
    return None


def read_zip_audio(source: str) -> bytes:
    """Read only a member's bytes, preferring an already cached local archive."""
    import blobfile as bf

    from verl.audio_cache import resolve_audio_source

    reference = split_zip_audio_source(source)
    if reference is None:
        raise ValueError(f"Expected a ZIP audio reference: {source!r}")
    archive, offset, size = reference
    archive = resolve_audio_source(archive)
    with bf.BlobFile(archive, "rb", streaming=True, buffer_size=min(size, 8 * 1024 * 1024)) as stream:
        stream.seek(offset)
        data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"Truncated ZIP audio member {source!r}: expected {size} bytes, got {len(data)}")
    return data

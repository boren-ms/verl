"""Pack a dataset's audio into one ZIP and publish a replacement JSONL.

Example:
    python -m scripts.pack_audio_zip \
        az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/data.jsonl \
        --zip-path az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/audio.zip \
        --output-jsonl az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/data_zip.jsonl
"""

import argparse
import gzip
import json
import logging
import os
import struct
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blobfile as bf

from verl.audio_cache import _split_audio_source, resolve_audio_source
from verl.audio_zip import split_zip_audio_source

logger = logging.getLogger(__name__)


def _read_source(source: str) -> bytes:
    with bf.BlobFile(resolve_audio_source(source), "rb", streaming=True) as stream:
        data = stream.read()
    if not data:
        raise ValueError(f"Empty audio file: {source}")
    return data


def pack_audio_zip(
    manifest: str,
    zip_path: str,
    output_jsonl: str,
    audio_root: str | None = None,
    audio_field: str = "audio_path",
    workers: int = 16,
    work_dir: str | None = None,
) -> dict[str, int | str]:
    """Preserve row order/metadata and deduplicate physical audio files.

    ZIP_STORED and ZIP64 make entries directly range-readable, including archives
    over 4 GiB. Publish the manifest only after the completed archive is uploaded.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = os.path.expanduser(manifest)
    zip_path = os.path.expanduser(zip_path)
    output_jsonl = os.path.expanduser(output_jsonl)
    if "://" not in zip_path:
        zip_path = os.path.abspath(zip_path)
    if not zip_path.lower().endswith(".zip") or not output_jsonl.endswith(".jsonl"):
        raise ValueError("Outputs must end in .zip and .jsonl")
    for output in (zip_path, output_jsonl):
        if bf.exists(output):
            raise FileExistsError(f"Refusing to overwrite {output}")
    audio_root = os.path.expanduser(audio_root) if audio_root else bf.dirname(manifest)
    rows = []
    sources = {}
    with bf.BlobFile(manifest, "rb") as raw:
        stream = gzip.GzipFile(fileobj=raw) if manifest.endswith(".gz") else raw
        try:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc
                if not isinstance(row, dict) or not isinstance(row.get(audio_field), str) or not row[audio_field]:
                    raise ValueError(f"Missing string {audio_field!r} at {manifest}:{line_number}")
                source, selector = _split_audio_source(row[audio_field])
                if ".zip!" in row[audio_field].lower():
                    raise ValueError(f"Already packed audio at {manifest}:{line_number}")
                if selector.startswith(":"):
                    raise ValueError(f"Chunk containers are not standalone audio at {manifest}:{line_number}")
                if "audio_zip_member" in row:
                    raise ValueError(f"Reserved field audio_zip_member at {manifest}:{line_number}")
                source = os.path.expanduser(source)
                if "://" not in source and not os.path.isabs(source):
                    source = bf.join(audio_root, source)
                sources.setdefault(source, None)
                rows.append((row, source, selector))
        finally:
            if stream is not raw:
                stream.close()
    if not rows:
        raise ValueError(f"No audio records in {manifest}")

    references = {}
    source_paths = list(sources)
    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix="pack-audio-", dir=work_dir) as temporary:
        local_zip = Path(temporary) / "audio.zip"
        local_manifest = Path(temporary) / "data.jsonl"
        with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(source_paths), workers):
                    batch = source_paths[start : start + workers]
                    for index, (source, data) in enumerate(
                        zip(batch, executor.map(_read_source, batch), strict=True), start
                    ):
                        member = f"audio/{index:08d}{Path(source).suffix}"
                        with archive.open(member, "w", force_zip64=True) as entry:
                            entry.write(data)
                        info = archive.getinfo(member)
                        # ZIP64 adds extra fields; read the actual local header lengths.
                        stream = archive.fp
                        if stream is None:
                            raise RuntimeError("ZIP writer closed unexpectedly")
                        end = stream.tell()
                        stream.seek(info.header_offset)
                        header = stream.read(30)
                        name_size, extra_size = struct.unpack_from("<HH", header, 26)
                        offset = info.header_offset + 30 + name_size + extra_size
                        stream.seek(end)
                        references[source] = (f"{zip_path}!{offset}:{info.file_size}", member)
                        total_bytes += info.file_size
                    logger.info("Packed %d/%d files", min(start + workers, len(source_paths)), len(source_paths))
        with zipfile.ZipFile(local_zip) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"ZIP CRC validation failed for {bad_member}")
        with local_manifest.open("w", encoding="utf-8") as stream:
            for row, source, selector in rows:
                reference, member = references[source]
                split_zip_audio_source(reference)
                packed_row = dict(row)
                packed_row.setdefault(f"original_{audio_field}", row[audio_field])
                packed_row[audio_field] = reference + selector
                packed_row["audio_zip_member"] = member
                stream.write(json.dumps(packed_row, ensure_ascii=False) + "\n")
        for destination in (zip_path, output_jsonl):
            bf.makedirs(bf.dirname(destination) or ".")
        bf.copy(str(local_zip), zip_path, overwrite=False)
        bf.copy(str(local_manifest), output_jsonl, overwrite=False)
    return {
        "rows": len(rows),
        "unique_files": len(sources),
        "audio_bytes": total_bytes,
        "zip_path": zip_path,
        "output_jsonl": output_jsonl,
    }


def pack_parquet_audio_zip(
    manifest: str,
    zip_path: str,
    output_jsonl: str,
    workers: int = 16,
    work_dir: str | None = None,
) -> dict[str, int | str]:
    """Externalize Parquet ``audio.bytes`` without re-encoding or losing metadata."""
    import pyarrow.parquet as pq

    paths = sorted(bf.glob(manifest)) if any(char in manifest for char in "*?[") else [manifest]
    if not paths:
        raise FileNotFoundError(f"No Parquet files matched: {manifest}")
    for output in (zip_path, output_jsonl):
        if bf.exists(output):
            raise FileExistsError(f"Refusing to overwrite {output}")
    with tempfile.TemporaryDirectory(prefix="parquet-audio-", dir=work_dir) as temporary:
        staged_manifest = Path(temporary) / "data.jsonl"
        count = 0
        with staged_manifest.open("w", encoding="utf-8") as stream:
            for path in paths:
                with bf.BlobFile(path, "rb") as parquet_stream:
                    parquet = pq.ParquetFile(parquet_stream)
                    for batch in parquet.iter_batches(batch_size=32):
                        for row in batch.to_pylist():
                            audio = row.pop("audio", None)
                            if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
                                raise ValueError(f"Missing audio.bytes in {path}, row {count}")
                            if "audio_path" in row:
                                raise ValueError(f"Conflicting audio_path in {path}, row {count}")
                            suffix = Path(audio.get("path") or "audio.wav").suffix
                            audio_path = Path(temporary) / f"{count:08d}{suffix}"
                            audio_path.write_bytes(audio["bytes"])
                            row["audio_path"] = str(audio_path)
                            row["original_audio_path"] = audio.get("path")
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            count += 1
        return pack_audio_zip(
            str(staged_manifest), zip_path, output_jsonl, workers=workers, work_dir=work_dir
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--audio-root", help="Base for relative audio paths (default: manifest directory)")
    parser.add_argument("--audio-field", default="audio_path")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--work-dir", help="Local scratch directory with space for the complete ZIP")
    parser.add_argument("--input-format", choices=("jsonl", "parquet"), default="jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    options = vars(args)
    input_format = options.pop("input_format")
    if input_format == "parquet":
        if options.pop("audio_root") is not None or options.pop("audio_field") != "audio_path":
            parser.error("--audio-root and --audio-field apply only to JSONL input")
        summary = pack_parquet_audio_zip(**options)
    else:
        summary = pack_audio_zip(**options)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build resumable JSONL shards over original audio chunks copied to Orange."""

import argparse
import concurrent.futures
import gzip
import json
import math
import os
import struct
import threading
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from recipe.phimm.tools.prepare_moe_diarization import (
    SOURCES,
    annotations,
    make_row,
    read_info,
    recording_group,
    windows,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CORP = "https://tsstd01uks.blob.core.windows.net"
INPUT = CORP + "/data/users/xioxiao/am_data/en_long"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        retry = Retry(total=4, backoff_factor=1, status_forcelist=(408, 429, 500, 502, 503, 504))
        LOCAL.session.mount("https://", HTTPAdapter(max_retries=retry))
    return LOCAL.session


def read(key, sas, byte_range=None):
    url = f"{INPUT}/{quote(key, safe='/')}?{sas}"
    headers = {"x-ms-version": "2021-12-02"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    # Discard response URLs on failures: the source URL contains a credential.
    try:
        with session().get(url, headers=headers, timeout=(15, 120)) as response:
            if response.status_code != (206 if byte_range else 200):
                raise ValueError(f"{key}: HTTP {response.status_code}")
            content = response.content
            size = None
            if byte_range:
                actual, size = response.headers["Content-Range"].split("/")
                start, end = byte_range
                size = int(size)
                end = min(end, size - 1)
                if actual != f"bytes {start}-{end}" or len(content) != end - start + 1:
                    raise ValueError(f"{key}: invalid range response")
            return content, size
    except requests.RequestException:
        raise OSError(f"{key}: blob request failed after retries") from None


def wav_metadata(header, size):
    """Return sample rate, frame count and format from a partial RIFF WAV read."""
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("record is not RIFF/WAVE")
    end = struct.unpack_from("<I", header, 4)[0] + 8
    if end > size:
        raise ValueError("RIFF size exceeds audio record length")
    off = 12
    fmt = None
    while off + 8 <= len(header):
        ident, length = struct.unpack_from("<4sI", header, off)
        off += 8
        if off + length > end:
            raise ValueError("WAV subchunk exceeds RIFF size")
        if ident == b"fmt " and length >= 16 and off + 16 <= len(header):
            fmt = struct.unpack_from("<HHIIHH", header, off)
        if ident == b"data" and fmt is not None:
            code, channels, rate, _, align, bits = fmt
            if (
                code not in (1, 3, 6, 7, 65534)
                or channels < 1
                or rate < 1
                or bits < 8
                or align != channels * ((bits + 7) // 8)
                or length % align
            ):
                raise ValueError("unsupported or invalid WAV frame format")
            return rate, length // align, channels
        off += length + (length % 2)
    return None


def source_rows(source, path, split, entry, sas, orange_root, dest, record_limit):
    chunk = entry["name"]
    assert Path(chunk).name == chunk
    shard = dest / f"{chunk}.jsonl.gz"
    audit_path = dest / f"{chunk}.audit.json"
    if shard.is_file() and audit_path.is_file():
        return json.loads(audit_path.read_text())
    base = f"{path}/chunks/{chunk}"
    info = read_info(read(base + ".info", sas)[0], entry["count"])
    audio_key = base + ".audio"
    envelope, blob_size = read(audio_key, sas, (0, 511))
    if envelope[:9] != b"Audio\x01\x00\x00\x00":
        raise ValueError(f"{base}: invalid Audio chunk header")
    all_indices = set()
    offset = 9
    stats = Counter()
    groups = set()
    details = []
    limit = record_limit or entry["count"]
    part = shard.with_suffix(".incomplete")
    with gzip.open(part, "wt", encoding="utf-8", compresslevel=3) as out:
        for i in range(min(limit, entry["count"])):
            head = envelope[offset:] if i == 0 else read(audio_key, sas, (offset, offset + 511))[0]
            index, size = struct.unpack_from("<II", head)
            payload_offset = offset + 8
            offset = payload_offset + size
            if index not in info or index in all_indices or offset > blob_size or not size:
                raise ValueError(f"{base}: invalid audio record index or size: {index}")
            all_indices.add(index)
            meta = wav_metadata(head[8 : 8 + size], size)
            if meta is None:
                for n in (8192, 65536, 1048576):
                    header = read(audio_key, sas, (payload_offset, payload_offset + min(size, n) - 1))[0]
                    meta = wav_metadata(header, size)
                    if meta is not None:
                        break
            if meta is None:
                raise ValueError(f"{base}: no PCM WAV data header for record {index}")
            rate, frames, channels = meta
            duration = frames / rate
            stats["records"] += 1
            stats["source_audio_s"] += duration
            record = info[index]
            # A malformed label is quarantined with its source ID. The audited
            # audio stays available on Orange along with the untouched .info.
            try:
                items, kind = annotations(record)
            except (ValueError, KeyError, TypeError) as e:
                details.append({"record_index": index, "status": "invalid_annotation", "error": type(e).__name__})
                stats["invalid_annotation_records"] += 1
                continue
            stats["annotation_items"] += len(items)
            stats["out_of_waveform_items"] += sum(item[1] > duration for item in items)
            nrows = 0
            for window in windows(items, duration):
                first_frame = max(0, math.floor(window[0][0] * rate))
                last_frame = min(frames, math.ceil(max(item[1] for item in window) * rate))
                start = first_frame / rate
                end = last_frame / rate
                if not 5 * rate <= last_frame - first_frame <= 30 * rate:
                    stats["length_filtered_items"] += len(window)
                    continue
                audio_ref = f"{orange_root.rstrip('/')}/{audio_key}!{payload_offset}:{size}#{start}:{end}"
                name = Path(f"{source}_{chunk}_{index}_{first_frame}_{last_frame}.wav")
                row = make_row(window, kind, start, end, source, split, record, chunk, index, name)
                row["audio_path"] = audio_ref
                row["source_audio_duration_s"] = round(duration, 6)
                row["source_sample_rate"] = rate
                row["source_channels"] = channels
                groups.add(recording_group(row))
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["clips"] += 1
                stats["audio_s"] += (last_frame - first_frame) / rate
                stats["reference_words"] += len(row["text"].split())
                nrows += 1
            if nrows == 0:
                stats["records_without_clips"] += 1
                details.append({"record_index": index, "status": "no_eligible_windows"})
    if not record_limit and (all_indices != set(info) or offset != blob_size):
        raise ValueError(f"{base}: audio records do not cover the expected chunk")
    part.replace(shard)
    result = {
        "source": source,
        "chunk": chunk,
        "split": split,
        "audio_bytes": blob_size,
        "stats": dict(stats),
        "recording_groups": sorted(groups),
        "excluded_records": details,
    }
    part = audit_path.with_suffix(".incomplete")
    part.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    part.replace(audit_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orange-source-root", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunks-per-source", type=int, default=0)
    parser.add_argument("--records-per-chunk", type=int, default=0)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    cache = json.loads((Path.home() / ".sas/azure_blob_sas_cache.json").read_text())
    sas = cache[CORP + "/data"]["sas"].lstrip("?")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = {
        "orange_source_root": args.orange_source_root,
        "chunks_per_source": args.chunks_per_source,
        "records_per_chunk": args.records_per_chunk,
        "layout": 1,
    }
    lock_path = args.out / "build_parameters.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Cannot resume shards under changed build parameters")
    else:
        temporary_lock = lock_path.with_suffix(f".{os.getpid()}.tmp")
        temporary_lock.write_text(json.dumps(lock, indent=2) + "\n")
        try:
            lock_path.hardlink_to(temporary_lock)
        except FileExistsError:
            if json.loads(lock_path.read_text()) != lock:
                raise ValueError("Cannot resume shards under changed build parameters") from None
        finally:
            temporary_lock.unlink()
    totals = {}
    counts = Counter()
    groups = {}
    failures = []
    for source in args.sources:
        path, split = SOURCES[source]
        local_manifest = (args.manifests / f"{source}.json").read_bytes()
        remote_manifest, _ = read(path + "/chunks/file_set.json", sas)
        assert json.loads(local_manifest) == json.loads(remote_manifest), f"{source}: source manifest changed"
        entries = json.loads(local_manifest)["fileInfo"][: args.chunks_per_source or None]
        dest = args.out / ("moe_" + source)
        dest.mkdir(exist_ok=True)
        agg = Counter()
        group = set()
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            jobs = {
                pool.submit(
                    source_rows,
                    source,
                    path,
                    split,
                    entry,
                    sas,
                    args.orange_source_root,
                    dest,
                    args.records_per_chunk,
                ): entry["name"]
                for entry in entries
            }
            for i, job in enumerate(concurrent.futures.as_completed(jobs), start=1):
                try:
                    result = job.result()
                    agg.update(result["stats"])
                    agg["audio_bytes"] += result["audio_bytes"]
                    group.update(result["recording_groups"])
                except (ValueError, KeyError, OSError, struct.error) as error:
                    failures.append({"source": source, "chunk": jobs[job], "error": str(error)})
                    print(f"{source}/{jobs[job]}: {error}", flush=True)
                if i % 25 == 0 or i == len(entries):
                    print(
                        f"{source}: {i}/{len(entries)} chunks, {agg['clips']} clips, {len(failures)} chunk errors",
                        flush=True,
                    )
        totals[source] = dict(agg)
        totals[source]["input_records"] = sum(x["count"] for x in entries)
        counts[source] = len(entries)
        groups[source] = group
        (dest / "summary.json").write_text(json.dumps(totals[source], indent=2) + "\n")
    summary_name = (
        "summary.json" if set(args.sources) == set(SOURCES) else "summary-" + "-".join(args.sources) + ".json"
    )
    (args.out / summary_name).write_text(
        json.dumps({"totals": totals, "chunks": counts, "failures": failures}, indent=2) + "\n"
    )
    if groups.get("ami_train", set()) & groups.get("ami_dev", set()):
        raise SystemExit("AMI train/dev overlap at meeting series level")
    if failures:
        raise SystemExit("One or more chunks failed; resume the build after investigating")


if __name__ == "__main__":
    main()

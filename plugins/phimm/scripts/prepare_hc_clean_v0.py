#!/usr/bin/env python3
"""Prepare verified hc_clean_v0 manifests and byte-preserving ZIP audio shards.

Run with the repo's Python environment containing azure-storage-blob and soundfile.
This builds local artifacts only; publish with azcopy through orngtransfer.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import soundfile as sf
from azure.storage.blob import ContainerClient


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--orange-prefix", required=True)
    parser.add_argument("--sas-cache", type=Path, default=Path("~/.sas/azure_blob_sas_cache.json").expanduser())
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--shard-mib", type=int, default=256)
    args = parser.parse_args()
    root, out = args.release_dir, args.output_dir
    require(args.workers > 0 and args.shard_mib > 0, "Workers and shard size must be positive")
    require(args.orange_prefix.startswith("az://orngwus2cresco/"), "Expected final Orange prefix")
    manifest = json.loads((root / "MANIFEST.json").read_bytes())
    receipt = json.loads((root / "RELEASE.json").read_bytes())
    require(manifest["release"] == "boren10k_fast2609_entity_train_test_v1", "Unexpected release")
    require(sha256(root / "MANIFEST.json") == receipt["manifest"]["sha256"], "Manifest hash mismatch")
    require(sha256(root / receipt["archive"]["path"]) == receipt["archive"]["sha256"], "ZIP hash mismatch")
    for name, metadata in manifest["files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "Unsafe release member")
        require(path.stat().st_size == metadata["bytes"] and sha256(path) == metadata["sha256"], name)
        if "records" in metadata:
            require(len(rows(path)) == metadata["records"], "Row count mismatch: " + name)
    index = rows(root / "audio/index.jsonl")
    require(len(index) == len({row["id"] for row in index}) == 14794, "Invalid audio index")
    account = "https://highperf01safn.blob.core.windows.net/data"
    sas = json.loads(args.sas_cache.read_bytes())[account]["sas"].lstrip("?")
    client = ContainerClient(account.rsplit("/", 1)[0], "data", credential=sas, retry_total=3)
    args.audio_cache.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    readme = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/data/pname/README.md"
    (out / "README.md").write_bytes(readme.read_bytes())

    def fetch(record):
        uid = record["id"]
        require(Path(uid).name == uid and uid.endswith(".wav"), "Unsafe audio ID")
        destination = args.audio_cache / uid
        blob, count, ordinal = record["audio_path"].rsplit(":", 2)
        size, offset = record["payload_bytes"], record["payload_offset"]
        require(blob.startswith("am_data/en/human_caption_v2/") and blob.endswith(".audio"), "Unexpected audio path")
        require(0 <= int(ordinal) < int(count) and offset >= 17 and 44 <= size <= 64 * 1024**2, "Invalid range")
        if destination.exists():
            wav = destination.read_bytes()
        else:
            # Validate the native frame as well as the exact released WAV digest.
            framed = client.get_blob_client(blob).download_blob(offset=offset - 8, length=size + 8).readall()
            require(len(framed) == size + 8, "Short audio range: " + uid)
            require(struct.unpack_from("<II", framed) == (int(ordinal), size), "Native framing mismatch: " + uid)
            wav = framed[8:]
        require(
            len(wav) == size and hashlib.sha256(wav).hexdigest() == record["wav_sha256"], "WAV hash mismatch: " + uid
        )
        require(wav[:4] == b"RIFF" and wav[8:12] == b"WAVE", "Not a WAV: " + uid)
        info = sf.info(io.BytesIO(wav))
        require(info.frames > 0 and info.samplerate > 0, "Empty audio: " + uid)
        if not destination.exists():
            temporary = destination.with_suffix(".partial")
            temporary.write_bytes(wav)
            temporary.replace(destination)
        return uid, {"audio_length": info.duration, "sample_rate": info.samplerate, "channels": info.channels}

    audio_info = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, record) for record in index]
        for future in concurrent.futures.as_completed(futures):
            uid, info = future.result()
            audio_info[uid] = info
            if len(audio_info) % 500 == 0:
                print(f"Verified audio {len(audio_info)}/{len(index)}", flush=True)
    client.close()

    references, inventory = {}, {}
    for split in ("train", "test"):
        shard, archive, used = -1, None, 0
        try:
            for record in (r for r in index if r["split"] == split):
                if archive is None or used + record["payload_bytes"] > args.shard_mib * 1024**2:
                    if archive is not None:
                        archive.close()
                    shard += 1
                    relative = f"audio/{split}-{shard:04d}.zip"
                    archive = zipfile.ZipFile(out / relative, "w", compression=zipfile.ZIP_STORED)
                    used = 0
                member = zipfile.ZipInfo(record["id"], date_time=(2026, 11, 19, 0, 0, 0))
                member.compress_type = zipfile.ZIP_STORED
                archive.writestr(member, (args.audio_cache / record["id"]).read_bytes())
                archive.fp.flush()
                with (out / relative).open("rb") as stream:
                    stream.seek(member.header_offset)
                    header = stream.read(30)
                name_size, extra_size = struct.unpack_from("<HH", header, 26)
                offset = member.header_offset + 30 + name_size + extra_size
                references[record["id"]] = f"{args.orange_prefix.rstrip('/')}/{relative}!{offset}:{member.file_size}"
                used += member.file_size
        finally:
            if archive is not None:
                archive.close()
    index_by_id = {r["id"]: r for r in index}
    split_ids = {}
    definitions = [
        ("train", "train/train.jsonl", "train/entity_targets.jsonl", 7261, 8609),
        ("val_entity_matched", "test/test.jsonl", "test/test_targets.jsonl", 7533, 20829),
    ]
    for split, source, sidecar, count, occurrences in definitions:
        (out / "targets").mkdir(exist_ok=True)
        (out / "targets" / f"{split}_targets.jsonl").write_bytes((root / sidecar).read_bytes())
        source_rows, annotations = rows(root / source), rows(root / sidecar)
        targets = {r["id"]: r["targets"] for r in annotations}
        ids = {r["id"] for r in source_rows}
        require(len(source_rows) == len(ids) == len(annotations) == len(targets) == count, "Invalid split size")
        require(ids == targets.keys(), "Target join mismatch")
        require(sum(len(t) for t in targets.values()) == occurrences, "Target count mismatch")
        split_ids[split] = ids
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in source_rows:
                uid, text = row["id"], row["display"]
                require(row["audio_path"] == index_by_id[uid]["audio_path"], "Audio join mismatch")
                require(index_by_id[uid]["split"] == ("train" if split == "train" else "test"), "Audio split mismatch")
                keywords = []
                for target in targets[uid]:
                    surface = target["reference_text"] if split == "train" else target["text"]
                    require(
                        text[target["display_start"] : target["display_end"]] == surface, "Target span mismatch: " + uid
                    )
                    if surface not in keywords:
                        keywords.append(surface)
                prepared = {
                    "id": uid,
                    "audio_path": references[uid],
                    "text": text,
                    "keywords": keywords,
                    "language": "English",
                    "split": split,
                    "source_audio_path": row["audio_path"],
                    "wav_sha256": index_by_id[uid]["wav_sha256"],
                    "target_occurrences": len(targets[uid]),
                    **audio_info[uid],
                }
                stream.write(json.dumps(prepared, ensure_ascii=False) + "\n")
        durations = [audio_info[uid]["audio_length"] for uid in ids]
        inventory[split] = {
            "rows": count,
            "target_occurrences": occurrences,
            "source_manifest": source,
            "source_targets": sidecar,
            "hours": sum(durations) / 3600,
            "max_audio_seconds": max(durations),
            "over_40_seconds": sum(d > 40 for d in durations),
        }
    require(not split_ids["train"] & split_ids["val_entity_matched"], "Train/test ID overlap")
    summary = {
        "dataset": "hc_clean_v0",
        "release_manifest_sha256": sha256(root / "MANIFEST.json"),
        "orange_prefix": args.orange_prefix,
        "splits": inventory,
        "audio_utterances": len(index),
        "audio_payload_bytes": sum(r["payload_bytes"] for r in index),
        "all_audio_sha256_verified": True,
        "reference_field": "display",
        "audio_policy": "Original WAV bytes in ZIP_STORED shards; no reencoding or segmentation",
        "keyword_policy": "Unique annotated surface forms; lexical keyword metrics differ from release occurrence EER",
    }
    write_json(out / "preparation.json", summary)
    files = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    write_json(out / "SHA256SUMS.json", files)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

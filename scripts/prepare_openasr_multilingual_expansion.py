"""Build pinned test snapshots as range-readable WAV ZIPs and JSONL manifests.

Publish only complete archives, manifests and provenance under a new output
root. French recordings use the shared 30-second Smart VAD and full references.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import blobfile as bf
import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf


def publish(local, root, uploader="blobfile"):
    if uploader == "bbb":
        import boostedblob as bbb
        from boostedblob.globals import config

        config.chunk_size = 1024 * 1024
        config.connect_timeout = 60
        config.read_timeout = 180

        async def upload():
            async with bbb.BoostExecutor(16) as executor:
                for filename in ["audio.zip", "data_zip.jsonl", "provenance.json"]:
                    source = local / filename
                    destination = root + "/" + filename
                    md5 = hashlib.file_digest(source.open("rb"), "md5").hexdigest()
                    if not await bbb.exists(destination):
                        print(f"Publishing {destination} ({source.stat().st_size} bytes)", flush=True)
                        await bbb.copyfile(str(source), destination, executor, overwrite=False)
                    stat = await bbb.stat(destination)
                    if stat.size != source.stat().st_size or stat.md5 != md5:
                        raise ValueError(f"Published artifact fingerprint mismatch: {destination}")
                    print(f"Verified {destination}", flush=True)

        asyncio.run(upload())
        return
    # Local uplink can take longer than blobfile's default 10s socket timeout.
    bf.configure(azure_write_chunk_size=1024 * 1024, connect_timeout=180, read_timeout=180, retry_limit=5)
    for filename in ["audio.zip", "data_zip.jsonl", "provenance.json"]:
        destination = root + "/" + filename
        if bf.exists(destination):
            stat = bf.stat(destination)
            local_file = local / filename
            md5 = hashlib.file_digest(local_file.open("rb"), "md5").hexdigest()
            if stat.size != local_file.stat().st_size or stat.md5 != md5:
                raise FileExistsError(f"Refusing to overwrite a different artifact: {destination}")
            print(f"Reusing verified {destination}", flush=True)
            continue
        print(f"Publishing {destination}", flush=True)
        bf.copy(str(local / filename), destination)


def prepare(spec, output_root, local_root, prepare_only=False, uploader="blobfile"):
    name = spec["result_name"]
    local = Path(local_root) / name
    local.mkdir(parents=True, exist_ok=True)
    parquet = local / "test.parquet"
    url = f'https://huggingface.co/datasets/{spec["hf_repo"]}/resolve/{spec["source_revision"]}/{spec["parquet_file"]}'
    if not parquet.exists():
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            partial = parquet.with_suffix(".partial")
            with partial.open("wb") as stream:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    stream.write(chunk)
            partial.rename(parquet)
    source_sha = hashlib.file_digest(parquet.open("rb"), "sha256").hexdigest()
    table = pq.ParquetFile(parquet)
    expected = int(spec["examples"])
    if table.metadata.num_rows != expected:
        raise ValueError(f'{name}: expected {expected} rows, got {table.metadata.num_rows}')
    root = output_root.rstrip("/") + "/" + name
    remote_zip = root + "/audio.zip"
    records = []
    long_form = spec["task"] == "openasr_long"
    chunker = None
    if long_form:
        from recipe.phimm.utils.svad.svad import SVadChunker
        chunker = SVadChunker(max_len_sec=30.0, verbose=False)
    with zipfile.ZipFile(local / "audio.zip", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        index = 0
        for batch in table.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                audio_info = row["audio"]
                if not audio_info.get("bytes"):
                    raise ValueError(f"{name}/{index}: snapshot has no embedded audio")
                audio, sr = sf.read(io.BytesIO(audio_info["bytes"]), dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if len(audio) == 0:
                    raise ValueError(f"{name}/{index}: empty audio")
                duration = len(audio) / sr
                text = next((row[key] for key in ["text", "sentence", "normalized_text", "transcript", "transcription"] if key in row and row[key] is not None), None)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{name}/{index}: missing reference")
                # Dataset loader preserves source references; score after concatenation.
                if long_form:
                    from recipe.phimm.utils.open_asr_normalizer.expansion_normalizer import MultilingualNormalizer
                    text = MultilingualNormalizer(remove_diacritics=False)(text, lang="fr")
                audio_id = str(next((row[key] for key in ["file_name", "audio_id", "id"] if row.get(key) is not None), index))
                wav = io.BytesIO()
                sf.write(wav, audio, sr, format="WAV", subtype="FLOAT")
                member = f"{index:06d}.wav"
                payload = wav.getvalue()
                archive.writestr(member, payload)
                info = archive.getinfo(member)
                # ZIP_STORED members can be fetched by byte range, as in pack_audio_zip.
                archive.fp.seek(info.header_offset)
                header = archive.fp.read(30)
                name_len, extra_len = struct.unpack_from("<HH", header, 26)
                archive.fp.seek(0, 2)
                offset = info.header_offset + 30 + name_len + extra_len
                audio_path = f"{remote_zip}!{offset}:{len(payload)}"
                base = {"id": f"{name}_{index}_{audio_id}", "dataset": name, "text": text,
                        "audio_path": audio_path, "audio_length_s": duration,
                        "sampling_rate": sr, "duration": duration,
                        "source_revision": spec["source_revision"], "source_row": index,
                        "metric": spec["metric"], "include_in_average_wer": False}
                if long_form:
                    from recipe.phimm.utils.audio import resample_audio
                    vad_audio = audio
                    if sr != 16000:
                        vad_audio, _ = resample_audio(vad_audio, sr, 16000)
                    spans = chunker.chunk(np.asarray(vad_audio, dtype=np.float32), 16000) if duration > 30 else [(0.0, duration)]
                    spans = [(float(start), float(end)) for start, end in spans if end - start >= 0.1]
                    if not spans:
                        raise ValueError(f"{name}/{index}: Smart VAD returned no segments")
                    for seg, (start, end) in enumerate(spans):
                        records.append({**base, "audio_path": f"{audio_path}#{round(start, 3)}:{round(end, 3)}",
                                        "parent_audio_path": audio_path, "seg_index": seg,
                                        "n_segments": len(spans), "seg_start": round(start, 3), "seg_end": round(end, 3)})
                else:
                    records.append(base)
                index += 1
                if index % 100 == 0 or index == expected:
                    print(f"{name}: {index}/{expected} parents, {len(records)} rows", flush=True)
    manifest = local / "data_zip.jsonl"
    with manifest.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    provenance = {**spec, "parents": index, "rows": len(records), "source_parquet_sha256": source_sha,
                  "zip_sha256": hashlib.file_digest((local / "audio.zip").open("rb"), "sha256").hexdigest(),
                  "manifest_sha256": hashlib.file_digest(manifest.open("rb"), "sha256").hexdigest(),
                  "segmenter": "verl Smart VAD max_len_sec=30 min_seg_sec=0.1" if long_form else None}
    (local / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if not prepare_only:
        publish(local, root, uploader)
    print(json.dumps(provenance), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="recipe/phimm/config/data/openasr_multilingual_expansion.tsv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--publish-only", action="store_true", help="Publish previously completed local archives without rebuilding")
    parser.add_argument("--prepare-only", action="store_true", help="Build completed local archives without uploading")
    parser.add_argument("--uploader", choices=["blobfile", "bbb"], default="blobfile")
    args = parser.parse_args()
    specs = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    for spec in specs:
        if args.datasets and spec["result_name"] not in args.datasets:
            continue
        if args.publish_only:
            local = Path(args.local_root) / spec["result_name"]
            proof = json.loads((local / "provenance.json").read_text())
            for filename, key in [("audio.zip", "zip_sha256"), ("data_zip.jsonl", "manifest_sha256")]:
                if hashlib.file_digest((local / filename).open("rb"), "sha256").hexdigest() != proof[key]:
                    raise ValueError(f"Local artifact fingerprint changed: {filename}")
            publish(local, args.output_root.rstrip("/") + "/" + spec["result_name"], args.uploader)
        else:
            prepare(spec, args.output_root, args.local_root, args.prepare_only, args.uploader)


if __name__ == "__main__":
    main()

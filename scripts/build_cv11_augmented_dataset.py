#!/usr/bin/env python3
"""Build full CV11 English training audio with speed -> RIR -> additive noise.

Downloads only the pinned train TSV/tars, verifies Hub LFS SHA256, joins every
archive member to the TSV, and retains original transcripts without filtering.
Each completed shard is resumable. Publish data.jsonl only after all audio has
been uploaded and all expected source IDs have been accounted for.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import blobfile as bf
from huggingface_hub import HfApi
import numpy as np
import requests
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.augment_audio import (
    linear_resample, mix_noise_at_snr, peak_normalize, read_audio,
    resample_to_sample_rate, reverberate_audio, speed_perturb, validate_signal,
)

DEFAULT_CONFIG = "recipe/phimm/config/data/cache/gen_cv11_en_train_speed_rir_noise_snr0_12.yaml"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


SOURCE_BLOB_DIR = None


def download_source(cfg, name, destination, expected_sha=None):
    if destination.exists():
        if expected_sha and file_sha256(destination) != expected_sha:
            raise ValueError(f"Existing source checksum mismatch: {destination}")
        return
    url = f"https://huggingface.co/datasets/{cfg['source_repo']}/resolve/{cfg['source_revision']}/{name}"
    for attempt in range(5):
        try:
            temporary = destination.with_suffix(destination.suffix + ".part")
            if SOURCE_BLOB_DIR:
                # Sources are staged from the pinned Hub revision by a separate
                # downloader. Wait for each atomic Blob upload, not a partial file.
                deadline = time.monotonic() + 6 * 3600
                while not bf.exists(SOURCE_BLOB_DIR.rstrip("/") + "/" + name):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Timed out waiting for staged source: {name}")
                    time.sleep(15)
                bf.copy(SOURCE_BLOB_DIR.rstrip("/") + "/" + name, str(temporary), overwrite=True, parallel=True)
            else:
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(4 * 1024 * 1024):
                            stream.write(block)
            if expected_sha and file_sha256(temporary) != expected_sha:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            temporary.replace(destination)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def load_metadata(path, expected_rows):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
    if len(rows) != expected_rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError("Source TSV count/uniqueness mismatch")
    if any(not r["sentence"].strip() or r.get("locale") != "en" for r in rows):
        raise ValueError("Source TSV contains empty text or non-English locale")
    return {row["path"]: (index, row["sentence"]) for index, row in enumerate(rows)}


def load_resources(manifest, directory, required_license):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not rows or len({r["audio_path"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate resource pool")
    for row in rows:
        local = directory / Path(row["audio_path"]).name
        if row.get("license") != required_license or file_sha256(local) != row["sha256"]:
            raise ValueError(f"Resource license/checksum mismatch: {local}")
        validate_signal(read_audio(str(local))[0], str(local))
    return rows


@lru_cache(maxsize=256)
def cached_resource(path):
    return read_audio(path)


def make_example(index, filename, text, original_bytes, archive_name, shard, cfg, rirs, noises,
                 rir_dir, noise_dir, audio_dir, fingerprint):
    speech, original_sr = sf.read(io.BytesIO(original_bytes), dtype="float32")
    if speech.ndim == 2:
        speech = speech.mean(axis=1)
    validate_signal(speech, filename)
    original_duration = len(speech) / original_sr
    speech = resample_to_sample_rate(speech, original_sr, cfg["sample_rate"]).astype(np.float32)
    rng = random.Random(cfg["seed"] + index)
    speed = rng.uniform(*cfg["speed_range"])
    snr = rng.uniform(*cfg["snr_range"])
    noise_record = rng.choice(noises)
    crop_seed = rng.randrange(2 ** 63)
    rir_rng = random.Random(cfg["rir_seed"] + index)
    rir_record = rir_rng.choice(rirs)
    rir, rir_sr = cached_resource(str(rir_dir / Path(rir_record["audio_path"]).name))
    noise, noise_sr = cached_resource(str(noise_dir / Path(noise_record["audio_path"]).name))
    sped = speed_perturb(speech, speed)
    reverberant, rir_info = reverberate_audio(sped, rir, cfg["sample_rate"], rir_sr, rir_rng)
    if noise_sr != cfg["sample_rate"]:
        noise = linear_resample(noise, max(1, round(len(noise) * cfg["sample_rate"] / noise_sr)))
    crop_rng = random.Random(crop_seed)
    crop_start = crop_rng.randint(0, len(noise) - len(sped)) if len(noise) >= len(sped) else 0
    mixture = mix_noise_at_snr(reverberant, noise, snr, random.Random(crop_seed))
    power = float(np.mean(reverberant.astype(float) ** 2))
    measured = float(10 * np.log10(power / np.mean((mixture.astype(float) - reverberant) ** 2)))
    if abs(measured - snr) > 0.001:
        raise ValueError(f"SNR mismatch: {filename}")
    # Keep ordinary levels unchanged, but lift very quiet mixtures above the
    # PCM16 quantization floor. A common gain preserves the speech/noise ratio.
    peak_limit = 0.99 / float(np.max(np.abs(mixture)))
    mixture_rms = float(np.sqrt(np.mean(mixture.astype(float) ** 2)))
    peak_gain = min(peak_limit, max(1.0, 0.01 / mixture_rms))
    output = (mixture * peak_gain).astype(np.float32)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, output, cfg["sample_rate"], format="WAV", subtype="PCM_16")
    wav = wav_buffer.getvalue()
    decoded, actual_sr = sf.read(io.BytesIO(wav), dtype="float32")
    validate_signal(decoded, "Output PCM16")
    measured_pcm = float(10 * np.log10(power / np.mean((decoded.astype(float) / peak_gain - reverberant) ** 2)))
    if actual_sr != cfg["sample_rate"] or len(decoded) != len(sped) or abs(measured_pcm - snr) > 0.02:
        raise ValueError(f"PCM16 validation failed: {filename}")
    name = Path(filename).stem + ".wav"
    destination = audio_dir / name
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(wav)
    temporary.replace(destination)
    return {
        "id": Path(filename).stem,
        "audio_path": f"{cfg['output_dir']}/audio/{shard}/{name}",
        "text": text, "language": "English", "data_source": cfg["data_source"],
        "source_dataset": cfg["source_upstream"], "source_release": cfg["source_release"],
        "source_language": "en", "source_split": cfg["source_split"], "source_index": index,
        "source_repo": cfg["source_repo"], "source_revision": cfg["source_revision"],
        "source_archive": archive_name, "source_audio_member": filename,
        "source_audio_sha256": sha256(original_bytes), "source_license": cfg["source_license"],
        "source_duration": original_duration, "source_sample_rate": original_sr,
        "speed_factor": speed, "snr_db": snr, "measured_snr_db_before_pcm16": measured,
        "measured_snr_db_after_pcm16": measured_pcm, "peak_gain": peak_gain,
        "rir_path": rir_record["audio_path"], "rir_sha256": rir_record["sha256"],
        "rir_license": rir_record["license"], "rir_room": rir_record.get("room"), **rir_info,
        "noise_path": noise_record["audio_path"], "noise_sha256": noise_record["sha256"],
        "noise_license": noise_record["license"], "noise_room": noise_record.get("room"),
        "noise_crop_seed": crop_seed, "noise_crop_start_sample": crop_start,
        "augmentation_order": ["speed", "rir", "noise"], "seed": cfg["seed"], "rir_seed": cfg["rir_seed"],
        "sample_rate": actual_sr, "duration": len(decoded) / actual_sr,
        "audio_num_samples": len(decoded), "audio_num_bytes": len(wav), "audio_sha256": sha256(wav),
        "build_fingerprint": fingerprint,
    }


def upload_directory(local, remote, concurrency):
    for attempt in range(5):
        result = subprocess.run(["bbb", "sync", str(local) + "/", remote + "/", "--concurrency", str(concurrency), "-q"])
        if result.returncode == 0:
            return
        if attempt == 4:
            result.check_returncode()
        time.sleep(2 ** attempt)


def publish_shard(output_dir, remote, concurrency, marker, fingerprint, completed, rows):
    upload_directory(output_dir, remote, concurrency)
    atomic_json(marker, {"fingerprint": fingerprint, "rows": rows, "manifest_sha256": file_sha256(completed)})


def initialize_worker(cfg, rirs, noises, rir_dir, noise_dir, fingerprint):
    global WORKER_CONTEXT
    WORKER_CONTEXT = (cfg, rirs, noises, rir_dir, noise_dir, fingerprint)


def process_example(index, filename, text, original_bytes, archive_name, shard, audio_dir):
    cfg, rirs, noises, rir_dir, noise_dir, fingerprint = WORKER_CONTEXT
    return make_example(index, filename, text, original_bytes, archive_name, shard, cfg,
                        rirs, noises, rir_dir, noise_dir, audio_dir, fingerprint)


def run(args):
    global SOURCE_BLOB_DIR
    SOURCE_BLOB_DIR = args.source_blob_dir
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["source_split"] != "train" or cfg["source_language"] != "en":
        raise ValueError("This builder requires English TRAIN")
    if not 0 < cfg["speed_range"][0] <= cfg["speed_range"][1] or not 0 <= cfg["snr_range"][0] <= cfg["snr_range"][1]:
        raise ValueError("Invalid augmentation ranges")
    if bf.exists(cfg["output_dir"] + "/data.jsonl"):
        raise FileExistsError("Final remote manifest already exists; refusing overwrite")
    root = Path(args.work_dir)
    source_dir = root / "source"
    artifacts = root / "artifacts"
    shard_dir = root / "shards"
    for directory in [source_dir, artifacts, shard_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    resource_dir = Path(args.resource_dir)
    resources = artifacts / "resources"
    resources.mkdir(exist_ok=True)
    for name in ["rirs.jsonl", "noises.jsonl", "ATTRIBUTION.md", "cc-by-4.0.txt", "read_me.txt", "selection.json", "source_page.html"]:
        shutil.copyfile(resource_dir / name, resources / name)
    rirs = load_resources(resources / "rirs.jsonl", resource_dir / "rirs_final", cfg["resource_license"])
    noises = load_resources(resources / "noises.jsonl", resource_dir / "noises_final", cfg["resource_license"])
    if args.source_files_json:
        source_files = json.loads(Path(args.source_files_json).read_text())
        if source_files["repo"] != cfg["source_repo"] or source_files["revision"] != cfg["source_revision"]:
            raise ValueError("Staged Hub metadata does not match configured revision")
        hub_files = [SimpleNamespace(rfilename=f["filename"], size=f["size"],
                     lfs=SimpleNamespace(sha256=f["sha256"])) for f in source_files["files"]]
    else:
        hub_files = HfApi().dataset_info(cfg["source_repo"], revision=cfg["source_revision"], files_metadata=True).siblings
    archives = sorted([f for f in hub_files if f.rfilename.startswith("en_train_") and f.rfilename.endswith(".tar")],
                      key=lambda f: int(Path(f.rfilename).stem.split("_")[-1]))
    if not archives or any(f.lfs is None for f in archives):
        raise ValueError("Missing train archives or LFS SHA256 metadata")
    tsv = next(f for f in hub_files if f.rfilename == "train.tsv")
    download_source(cfg, "train.tsv", source_dir / "train.tsv", tsv.lfs.sha256)
    metadata = load_metadata(source_dir / "train.tsv", cfg["expected_rows"])
    provenance = {
        "config": cfg, "source_tsv_sha256": file_sha256(source_dir / "train.tsv"),
        "archives": [{"filename": f.rfilename, "size": f.size, "sha256": f.lfs.sha256} for f in archives],
        "rirs_sha256": file_sha256(resources / "rirs.jsonl"), "noises_sha256": file_sha256(resources / "noises.jsonl"),
        "code_sha256": file_sha256(__file__),
        "dsp_sha256": file_sha256(Path(__file__).with_name("augment_audio.py")),
    }
    fingerprint = sha256(json.dumps(provenance, sort_keys=True).encode())
    provenance_file = root / "provenance.json"
    if provenance_file.exists() and json.loads(provenance_file.read_text()) != provenance:
        raise ValueError("Build inputs changed; use a new work directory")
    atomic_json(provenance_file, provenance)
    shutil.copyfile(provenance_file, artifacts / "provenance.json")
    shutil.copyfile(source_dir / "train.tsv", artifacts / "source_train.tsv")
    shutil.copyfile(args.config, artifacts / "generation.yaml")
    print(json.dumps({"source_rows": len(metadata), "archives": len(archives), "fingerprint": fingerprint}), flush=True)
    accounted = set()
    started = time.monotonic()
    # Overlap the next archive download and completed audio uploads with CPU DSP.
    downloads = ThreadPoolExecutor(max_workers=1)
    uploads = ThreadPoolExecutor(max_workers=1)
    upload_futures = []
    next_download = None
    for archive_index, archive in enumerate(archives):
        if next_download is not None:
            next_download.result()
            next_download = None
        if archive_index + 1 < len(archives) and not args.max_shards:
            following = archives[archive_index + 1]
            next_download = downloads.submit(download_source, cfg, following.rfilename,
                                               source_dir / following.rfilename, following.lfs.sha256)
        for future in upload_futures:
            if future.done():
                future.result()
        shard = Path(archive.rfilename).stem
        completed = shard_dir / f"{shard}.jsonl"
        output_dir = artifacts / "audio" / shard
        output_dir.mkdir(parents=True, exist_ok=True)
        if not completed.exists():
            print(f"Downloading {archive.rfilename} ({archive.size} bytes)", flush=True)
            local_archive = source_dir / archive.rfilename
            download_source(cfg, archive.rfilename, local_archive, archive.lfs.sha256)
            pending = []
            results = []
            seen = set()
            with tarfile.open(local_archive) as tar, ProcessPoolExecutor(
                max_workers=cfg["workers"], initializer=initialize_worker,
                initargs=(cfg, rirs, noises, resource_dir / "rirs_final", resource_dir / "noises_final", fingerprint),
            ) as executor:
                for member in tar:
                    if not member.isfile():
                        continue
                    filename = Path(member.name).name
                    if filename not in metadata or filename in seen or filename in accounted:
                        raise ValueError(f"Unexpected/duplicate archive member: {filename}")
                    seen.add(filename)
                    index, text = metadata[filename]
                    audio_bytes = tar.extractfile(member).read()
                    pending.append(executor.submit(process_example, index, filename, text, audio_bytes,
                        archive.rfilename, shard, output_dir))
                    if len(pending) >= cfg["workers"] * 4:
                        results.extend(f.result() for f in pending)
                        pending.clear()
                        if len(results) % 2048 == 0:
                            print(f"{shard}: generated {len(results)}", flush=True)
                results.extend(f.result() for f in pending)
            temporary = completed.with_suffix(".tmp")
            with temporary.open("w") as stream:
                for row in sorted(results, key=lambda row: row["source_index"]):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(completed)
        else:
            results = [json.loads(line) for line in completed.read_text().splitlines()]
        if any(row["build_fingerprint"] != fingerprint for row in results):
            raise ValueError("Shard fingerprint mismatch")
        for row in results:
            original_name = row["id"] + ".mp3"
            if original_name in accounted or metadata[original_name][1] != row["text"]:
                raise ValueError("Duplicate ID or transcript mismatch")
            accounted.add(original_name)
        uploaded = shard_dir / f"{shard}.uploaded.json"
        if uploaded.exists():
            saved = json.loads(uploaded.read_text())
            if saved["fingerprint"] != fingerprint or saved["manifest_sha256"] != file_sha256(completed):
                raise ValueError("Uploaded shard checkpoint mismatch")
        if args.publish and not uploaded.exists():
            print(f"Queued upload {shard}: {len(results)} WAVs", flush=True)
            upload_futures.append(uploads.submit(publish_shard, output_dir, cfg["output_dir"] + "/audio/" + shard,
                cfg["upload_concurrency"], uploaded, fingerprint, completed, len(results)))
        print(f"Generated {shard}; total {len(accounted)}/{len(metadata)}, elapsed {time.monotonic()-started:.1f}s", flush=True)
        if args.max_shards and len(list(shard_dir.glob("*.jsonl"))) >= args.max_shards:
            uploads.shutdown(wait=True)
            for future in upload_futures:
                future.result()
            downloads.shutdown(wait=True)
            print("Limited-shard smoke run; final manifest NOT published", flush=True)
            return
    downloads.shutdown(wait=True)
    uploads.shutdown(wait=True)
    for future in upload_futures:
        future.result()
    if accounted != set(metadata):
        raise ValueError(f"Missing {len(set(metadata) - accounted)} source IDs")
    counts = {"num_rows": 0, "audio_hours": 0.0, "source_audio_hours": 0.0,
              "max_snr_error_db": 0.0, "max_pcm16_snr_error_db": 0.0}
    used_rirs, used_noises = set(), set()
    snrs, speeds = [], []
    with (artifacts / "data.jsonl.tmp").open("w") as final:
        for archive in archives:
            for line in (shard_dir / f"{Path(archive.rfilename).stem}.jsonl").open():
                row = json.loads(line)
                final.write(line)
                counts["num_rows"] += 1
                counts["audio_hours"] += row["duration"] / 3600
                counts["source_audio_hours"] += row["source_duration"] / 3600
                counts["max_snr_error_db"] = max(counts["max_snr_error_db"], abs(row["snr_db"] - row["measured_snr_db_before_pcm16"]))
                counts["max_pcm16_snr_error_db"] = max(counts["max_pcm16_snr_error_db"], abs(row["snr_db"] - row["measured_snr_db_after_pcm16"]))
                used_rirs.add(row["rir_path"]); used_noises.add(row["noise_path"])
                snrs.append(row["snr_db"]); speeds.append(row["speed_factor"])
    (artifacts / "data.jsonl.tmp").replace(artifacts / "data.jsonl")
    summary = {**counts, "build_fingerprint": fingerprint, "manifest_sha256": file_sha256(artifacts / "data.jsonl"),
               "rir_pool_size": len(rirs), "noise_pool_size": len(noises),
               "rir_files_used": len(used_rirs), "noise_files_used": len(used_noises),
               "snr_min_mean_max": [min(snrs), float(np.mean(snrs)), max(snrs)],
               "speed_min_mean_max": [min(speeds), float(np.mean(speeds)), max(speeds)],
               "source_split": "train", "missing_or_filtered_rows": 0}
    atomic_json(artifacts / "summary.json", summary)
    if args.publish:
        upload_directory(resources, cfg["output_dir"] + "/resources", cfg["upload_concurrency"])
        for name in ["provenance.json", "source_train.tsv", "generation.yaml", "summary.json"]:
            bf.copy(str(artifacts / name), cfg["output_dir"] + "/" + name)
        if bf.exists(cfg["output_dir"] + "/data.jsonl"):
            raise FileExistsError("Final manifest appeared during build; refusing overwrite")
        bf.copy(str(artifacts / "data.jsonl"), cfg["output_dir"] + "/data.jsonl")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--publish", action="store_true", help="Upload WAVs and publish final manifest LAST")
    parser.add_argument("--source-blob-dir", help="Read staged, checksum-verified source files from Blob")
    parser.add_argument("--source-files-json", help="Pinned Hub file metadata for offline remote execution")
    parser.add_argument("--max-shards", type=int, help="Smoke run; never publishes final manifest")
    run(parser.parse_args())

#!/usr/bin/env python3
"""Stage pinned CV11 source files through Blob for offline remote generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import subprocess
import os

import blobfile as bf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_cv11_augmented_dataset import download_source, file_sha256


def stage(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    directory = Path(args.source_dir)
    metadata = json.loads((directory / "hub_files.json").read_text())
    if metadata["repo"] != cfg["source_repo"] or metadata["revision"] != cfg["source_revision"]:
        raise ValueError("Pinned source metadata mismatch")

    def upload(local, name):
        destination = args.blob_dir + "/" + name
        if bf.exists(destination):
            return
        if args.bridge_url:
            # Green-accessible bridge, then server-side copy to Orange. No SAS
            # on these endpoints; azcopy reuses the existing Green CLI login.
            account, container, key = destination.removeprefix("az://").split("/", 2)
            final_url = f"https://{account}.blob.core.windows.net/{container}/{key}"
            bridge_url = args.bridge_url.rstrip("/") + "/" + name
            env = {**os.environ, "AZCOPY_AUTO_LOGIN_TYPE": "AZCLI", "AZCOPY_CONCURRENCY_VALUE": "32"}
            subprocess.run(["azcopy", "copy", str(local), bridge_url, "--put-md5=true", "--overwrite=false"], env=env, check=True)
            subprocess.run(["azcopy", "copy", bridge_url, final_url, "--s2s-preserve-access-tier=false", "--overwrite=false"], env=env, check=True)
        else:
            subprocess.run(["bbb", "cp", str(local), destination, "--concurrency", "16", "-q"], check=True)

    for name in ["hub_files.json", "but_resources.tar"]:
        upload(directory / name, name)
        print(json.dumps({"staged": name, "sha256": file_sha256(directory / name)}), flush=True)

    def transfer(record):
        name = record["filename"]
        local = directory / name
        download_source(cfg, name, local, record["sha256"])
        if local.stat().st_size != record["size"]:
            raise ValueError(f"Source size mismatch: {name}")
        upload(local, name)
        print(json.dumps({"staged": name, "sha256": record["sha256"], "size": record["size"]}), flush=True)

    files = sorted(metadata["files"], key=lambda f: (-1 if f["filename"] == "train.tsv" else int(Path(f["filename"]).stem.rsplit("_", 1)[-1])))
    for record in files[:2]:
        transfer(record)
    print("REMOTE_BUILD_CAN_START", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(transfer, record) for record in files[2:]]
        for future in as_completed(futures):
            future.result()
    print("ALL_SOURCES_STAGED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="recipe/phimm/config/data/cache/gen_cv11_en_train_speed_rir_noise_snr0_12.yaml")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--blob-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bridge-url", help="orngtransfer URL for local upload followed by server-side Orange copy")
    stage(parser.parse_args())

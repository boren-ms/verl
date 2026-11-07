#!/usr/bin/env python3
"""Materialize pinned public CV11 archives server-side, then bridge to Orange.

Azure fetches each public signed Hub CDN URL into orngtransfer; the local machine
handles only HTTP headers and Azure control requests. Azure CLI credentials are
sent only to the Azure destination, never to the public source URL. The remote
dataset builder verifies the expected LFS SHA256 of every completed archive.
Signed CDN URLs are neither printed nor saved. Subsequent Azure-to-Azure bridge
copies use azcopy, following the transfer-to-orange workflow.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

from azure.identity import AzureCliCredential
from azure.storage.blob import BlobClient
import blobfile as bf
import requests


def ingest(record, metadata, args):
    name = record["filename"]
    if name == "train.tsv":
        return None
    started = time.monotonic()
    account, container, prefix = args.bridge_dir.removeprefix("az://").split("/", 2)
    target = BlobClient(account_url=f"https://{account}.blob.core.windows.net", container_name=container,
                        blob_name=prefix.rstrip("/") + "/" + name, credential=AzureCliCredential())
    for attempt in range(4):
        try:
            if not target.exists():
                source = f"https://huggingface.co/datasets/{metadata['repo']}/resolve/{metadata['revision']}/{name}?download=true"
                with requests.get(source, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if int(response.headers["Content-Length"]) != record["size"]:
                        raise ValueError(f"Pinned source length mismatch: {name}")
                    source_url = response.url
                target.upload_blob_from_url(source_url, overwrite=False, include_source_blob_properties=False, timeout=600)
            if target.get_blob_properties().size != record["size"]:
                raise ValueError(f"Bridge source size mismatch: {name}")
            destination = args.output_dir.rstrip("/") + "/" + name
            if not bf.exists(destination):
                final_account, final_container, key = destination.removeprefix("az://").split("/", 2)
                source_url = f"https://{account}.blob.core.windows.net/{container}/{prefix.rstrip('/')}/{name}"
                final_url = f"https://{final_account}.blob.core.windows.net/{final_container}/{key}"
                env = {**os.environ, "AZCOPY_AUTO_LOGIN_TYPE": "AZCLI", "AZCOPY_CONCURRENCY_VALUE": "32"}
                log = Path(args.log_dir) / f"{Path(name).stem}.log"
                with log.open("a") as stream:
                    subprocess.run(["azcopy", "copy", source_url, final_url,
                                    "--s2s-preserve-access-tier=false", "--overwrite=false"],
                                   env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = {"filename": name, "size": record["size"], "expected_sha256": record["sha256"],
                      "elapsed_seconds": time.monotonic() - started, "staged_in_orange": True}
            print(json.dumps(result), flush=True)
            return result
        except Exception as exc:
            # Exception URLs can contain short-lived CDN signatures. Log only
            # the error class; the per-file azcopy log uses unsigned Azure URLs.
            print(json.dumps({"filename": name, "attempt": attempt + 1, "error_type": type(exc).__name__}), flush=True)
            if attempt == 3:
                raise RuntimeError(f"Server ingestion failed for {name}: {type(exc).__name__}") from None
            time.sleep(2 ** attempt)


def run(args):
    metadata = json.loads(Path(args.metadata).read_text())
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(ingest, record, metadata, args) for record in metadata["files"]]
        results = [future.result() for future in as_completed(futures)]
    Path(args.log_dir, "summary.json").write_text(json.dumps([r for r in results if r], indent=2))
    print("ALL_ARCHIVES_STAGED_SERVER_SIDE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())

#!/usr/bin/env python3
"""Download blobs from an Azure Blob Storage prefix using a SAS token.

Example:
    python scripts/download_blob_with_sas.py \
      --url "https://tsstd01safn.blob.core.windows.net/data/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/" \
      --dest "~/data"

If --sas is omitted and URL has no SAS query, the script will try to auto-generate
container SAS from Azure CLI login credentials.

SAS tokens are cached in ~/.sas/azure_blob_sas_cache.json and reused for the
same container when still valid.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


SAS_CACHE_DIR = Path("~/.sas").expanduser()
SAS_CACHE_FILE = SAS_CACHE_DIR / "azure_blob_sas_cache.json"


@dataclass
class BlobEntry:
    name: str
    size: int


def _split_blob_url(url: str) -> tuple[str, str, str]:
    """Return (container_url, prefix, sas_from_url) from a blob folder URL."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    path = parsed.path.lstrip("/")
    if not path:
        raise ValueError("URL must include container and optional prefix")

    parts = path.split("/", 1)
    container = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    container_url = f"{parsed.scheme}://{parsed.netloc}/{container}"
    return container_url, prefix, parsed.query


def _normalize_sas(sas: str) -> str:
    sas = sas.strip()
    return sas[1:] if sas.startswith("?") else sas


def _parse_account_and_container(container_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(container_url)
    host = parsed.netloc
    if not host.endswith(".blob.core.windows.net"):
        raise ValueError(f"Unexpected Azure Blob host: {host}")
    account = host.split(".", 1)[0]
    container = parsed.path.strip("/")
    if not container:
        raise ValueError(f"Missing container in URL: {container_url}")
    return account, container


MICROSOFT_TENANT = "microsoft.com"


def _ensure_az_tenant(tenant: str) -> None:
    """Switch az CLI to the given tenant if not already active."""
    result = subprocess.run(
        ["az", "account", "show", "-o", "json"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            current = info.get("tenantId", "") or info.get("tenantDefaultDomain", "")
            if tenant.lower() in (current.lower(), info.get("tenantDefaultDomain", "").lower()):
                return
        except json.JSONDecodeError:
            pass

    print(f"Switching az CLI to tenant '{tenant}'...")
    login_result = subprocess.run(
        ["az", "login", "--tenant", tenant],
        check=False, capture_output=True, text=True,
    )
    if login_result.returncode != 0:
        raise RuntimeError(
            f"Failed to az login --tenant {tenant}. "
            f"Error: {login_result.stderr.strip()}"
        )
    print(f"Logged into tenant '{tenant}' successfully.")


def _generate_sas_via_az_cli(container_url: str, expiry_hours: int) -> str:
    account, container = _parse_account_and_container(container_url)

    if shutil.which("az") is None:
        raise RuntimeError("Azure CLI not found. Install Azure CLI or pass --sas explicitly.")

    expiry = (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).strftime("%Y-%m-%dT%H:%MZ")
    cmd = [
        "az",
        "storage",
        "container",
        "generate-sas",
        "--as-user",
        "--auth-mode",
        "login",
        "--account-name",
        account,
        "--name",
        container,
        "--permissions",
        "rl",
        "--expiry",
        expiry,
        "-o",
        "tsv",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        # SAS generation failed — try switching to Microsoft corp tenant and retry
        print(f"SAS generation failed with current tenant. Switching to {MICROSOFT_TENANT}...")
        _ensure_az_tenant(MICROSOFT_TENANT)
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                "Failed to auto-generate SAS via Azure CLI after switching to "
                f"{MICROSOFT_TENANT}. Error: {stderr}"
            )

    sas = result.stdout.strip().strip('"')
    if not sas:
        raise RuntimeError("Azure CLI returned an empty SAS token. Provide --sas manually.")
    return sas


def _parse_sas_expiry(sas: str) -> datetime | None:
    params = dict(urllib.parse.parse_qsl(_normalize_sas(sas), keep_blank_values=True))
    expiry = params.get("se")
    if not expiry:
        return None
    expiry = expiry.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(expiry)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_sas_usable(sas: str, min_remaining_minutes: int = 30) -> bool:
    expiry = _parse_sas_expiry(sas)
    if expiry is None:
        return True
    remaining = expiry - datetime.now(timezone.utc)
    return remaining >= timedelta(minutes=min_remaining_minutes)


def _read_sas_cache() -> dict[str, dict[str, str]]:
    if not SAS_CACHE_FILE.exists():
        return {}
    try:
        with open(SAS_CACHE_FILE, encoding="utf8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _write_sas_cache(cache: dict[str, dict[str, str]]) -> None:
    SAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAS_CACHE_FILE, "w", encoding="utf8") as f:
        json.dump(cache, f, ensure_ascii=True, indent=2, sort_keys=True)


def _load_cached_sas(container_url: str) -> str:
    cache = _read_sas_cache()
    entry = cache.get(container_url, {})
    sas = entry.get("sas", "")
    if not sas:
        return ""
    if not _is_sas_usable(sas):
        return ""
    return sas


def _store_cached_sas(container_url: str, sas: str) -> None:
    normalized = _normalize_sas(sas)
    cache = _read_sas_cache()
    expiry = _parse_sas_expiry(normalized)
    cache[container_url] = {
        "sas": normalized,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expiry_utc": expiry.strftime("%Y-%m-%dT%H:%M:%SZ") if expiry else "",
    }
    _write_sas_cache(cache)


def _with_sas(base_url: str, sas: str, extra_params: dict[str, str] | None = None) -> str:
    params = urllib.parse.parse_qsl(_normalize_sas(sas), keep_blank_values=True)
    if extra_params:
        params.extend(extra_params.items())
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def _list_blobs(container_url: str, prefix: str, sas: str) -> list[BlobEntry]:
    blobs: list[BlobEntry] = []
    marker = ""

    while True:
        params = {
            "restype": "container",
            "comp": "list",
            "prefix": prefix,
            "maxresults": "5000",
        }
        if marker:
            params["marker"] = marker

        request_url = _with_sas(container_url, sas, params)
        with urllib.request.urlopen(request_url) as resp:  # nosec B310
            xml_payload = resp.read()

        root = ET.fromstring(xml_payload)
        for blob_node in root.findall("./Blobs/Blob"):
            name = blob_node.findtext("Name")
            if not name or name.endswith("/"):
                continue
            size_text = blob_node.findtext("Properties/Content-Length") or "0"
            blobs.append(BlobEntry(name=name, size=int(size_text)))

        next_marker = root.findtext("NextMarker") or ""
        if not next_marker:
            break
        marker = next_marker

    return blobs


def _download_one(container_url: str, sas: str, blob: BlobEntry, dest_root: Path, lock: threading.Lock, strip_prefix: str = "") -> tuple[str, bool]:
    rel_name = blob.name
    if strip_prefix and rel_name.startswith(strip_prefix):
        rel_name = rel_name[len(strip_prefix):]
    local_path = dest_root / rel_name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size == blob.size:
        return blob.name, False

    encoded_name = urllib.parse.quote(blob.name, safe="/")
    blob_url = _with_sas(f"{container_url}/{encoded_name}", sas)

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with urllib.request.urlopen(blob_url) as resp, open(tmp_path, "wb") as out:  # nosec B310
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    os.replace(tmp_path, local_path)
    with lock:
        print(f"Downloaded: {blob.name}")
    return blob.name, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Azure Blob files from a folder URL using SAS")
    parser.add_argument(
        "--url",
        required=True,
        help="Blob folder URL, e.g. https://<account>.blob.core.windows.net/<container>/<prefix>/",
    )
    parser.add_argument(
        "--sas",
        default="",
        help="SAS token string (with or without leading '?'); optional if URL already includes SAS query",
    )
    parser.add_argument(
        "--sas-expiry-hours",
        type=int,
        default=168,
        help="Auto-generated SAS validity in hours when --sas is not provided (default: 168)",
    )
    parser.add_argument(
        "--dest",
        default="~/data",
        help="Local destination root (default: ~/data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
    )
    parser.add_argument(
        "--strip-prefix",
        action="store_true",
        default=True,
        help="Strip the URL prefix from blob names so files land directly in --dest (default: True)",
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Keep the full blob prefix path under --dest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    container_url, prefix, sas_from_url = _split_blob_url(args.url)
    sas = args.sas or sas_from_url

    if not sas:
        cached = _load_cached_sas(container_url)
        if cached:
            sas = cached
            print(f"Using cached SAS from {SAS_CACHE_FILE}")

    if not sas:
        print("No SAS provided; trying Azure CLI to generate one with login credentials...")
        sas = _generate_sas_via_az_cli(container_url, max(1, args.sas_expiry_hours))

    _store_cached_sas(container_url, sas)

    dest_root = Path(args.dest).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"Container: {container_url}")
    print(f"Prefix: {prefix or '(root)'}")
    print(f"Destination: {dest_root}")

    blobs = _list_blobs(container_url, prefix, sas)
    if not blobs:
        print("No blobs found for prefix.")
        return

    total_size = sum(b.size for b in blobs)
    print(f"Found {len(blobs)} blobs, total {total_size / (1024 ** 3):.2f} GiB")

    lock = threading.Lock()
    downloaded = 0
    skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        sp = prefix if not args.keep_prefix else ""
        futures = [pool.submit(_download_one, container_url, sas, b, dest_root, lock, sp) for b in blobs]
        for fut in concurrent.futures.as_completed(futures):
            _, was_downloaded = fut.result()
            if was_downloaded:
                downloaded += 1
            else:
                skipped += 1

    print(f"Done. downloaded={downloaded}, skipped={skipped}")


if __name__ == "__main__":
    main()

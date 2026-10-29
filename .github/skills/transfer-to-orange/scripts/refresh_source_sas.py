#!/usr/bin/env python3
"""Refresh a source-container SAS cache entry without printing the token."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_CACHE = Path("~/.sas/azure_blob_sas_cache.json").expanduser()
CORP_TENANT = "microsoft.com"
GREEN_TENANT = "8b9ebe14-d942-49e7-ace9-14496d0caff0"


def parse_container_url(source_url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme != "https" or not parsed.netloc.endswith(".blob.core.windows.net"):
        raise ValueError("source URL must be an HTTPS Azure Blob URL")
    path_parts = parsed.path.strip("/").split("/", 1)
    if not path_parts[0]:
        raise ValueError("source URL must include a container")
    account = parsed.netloc.split(".", 1)[0]
    container = path_parts[0]
    container_url = f"https://{parsed.netloc}/{container}"
    return account, container, container_url


def parse_expiry(sas: str) -> datetime | None:
    params = dict(urllib.parse.parse_qsl(sas.lstrip("?"), keep_blank_values=True))
    value = params.get("se")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"SAS cache is not a JSON object: {path}")
    return value


def write_cache_atomic(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(cache, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def az_login(tenant: str) -> None:
    """Authenticate with Azure CLI's browser flow; device code is forbidden."""
    result = subprocess.run(
        ["az", "login", "--tenant", tenant, "--allow-no-subscriptions", "--output", "none"],
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Browser-based Azure CLI login failed for tenant {tenant}. "
            f"Device-code fallback is disabled; complete `az login --tenant {tenant}` "
            "in the browser and retry."
        )


def generate_sas(account: str, container: str, expiry: datetime) -> str:
    result = subprocess.run(
        [
            "az", "storage", "container", "generate-sas",
            "--as-user", "--auth-mode", "login",
            "--account-name", account,
            "--name", container,
            "--permissions", "rl",
            "--expiry", expiry.strftime("%Y-%m-%dT%H:%MZ"),
            "--output", "tsv",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        error = result.stderr.strip()
        raise RuntimeError(f"SAS generation failed for {account}/{container}: {error}")
    sas = result.stdout.strip().strip('"').lstrip("?")
    if not sas:
        raise RuntimeError("Azure CLI returned an empty SAS token")
    return sas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh a cached read/list SAS for an Azure Blob source container."
    )
    parser.add_argument("--source-url", required=True, help="Source container or prefix URL")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--expiry-hours", type=int, default=168)
    parser.add_argument("--min-valid-minutes", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    account, container, container_url = parse_container_url(args.source_url)
    cache = read_cache(args.cache.expanduser())
    cached_sas = str(cache.get(container_url, {}).get("sas", ""))
    cached_expiry = parse_expiry(cached_sas)
    minimum_expiry = datetime.now(timezone.utc) + timedelta(minutes=args.min_valid_minutes)
    if not args.force and cached_sas and cached_expiry and cached_expiry >= minimum_expiry:
        print(f"Cached SAS remains usable for {container_url} until {cached_expiry.isoformat()}")
        return 0

    refresh_expiry = datetime.now(timezone.utc) + timedelta(hours=args.expiry_hours)
    refresh_error: BaseException | None = None
    try:
        az_login(CORP_TENANT)
        sas = generate_sas(account, container, refresh_expiry)
        actual_expiry = parse_expiry(sas)
        cache[container_url] = {
            "sas": sas,
            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expiry_utc": actual_expiry.strftime("%Y-%m-%dT%H:%M:%SZ") if actual_expiry else "",
        }
        write_cache_atomic(args.cache.expanduser(), cache)
        print(
            f"Refreshed SAS cache for {container_url}; "
            f"expires {cache[container_url]['expiry_utc']}"
        )
    except BaseException as error:
        refresh_error = error
    finally:
        try:
            az_login(GREEN_TENANT)
        except BaseException as restore_error:
            if refresh_error is None:
                refresh_error = restore_error
            else:
                print(f"WARNING: failed to restore Green tenant: {restore_error}")

    if refresh_error is not None:
        raise refresh_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

---
name: transfer-to-orange
description: "Transfer external Azure Blob data to Orange (orng) with azcopy through the orngtransfer bridge, refreshing the source SAS cache when needed. Use for highperf-to-Orange, two-stage blob transfers, and copying datasets to Orange storage."
---

# Transfer Data to Orange via azcopy

Two-stage azcopy transfer from an external Azure Blob account (commonly `highperf01*`) to Orange (orng), routed through the shared `orngtransfer` storage account.

## When to Use

- Transfer datasets, audio data, or models from an external Azure Blob account to Orange storage
- User says "copy to orange", "transfer to orng", "move data to orange blob"
- Two-stage pipeline: source Azure Blob account → orngtransfer → orngwus2cresco (all from dev machine)

## Architecture

There is no direct Corp → Orange path. Data flows through the Green-accessible `orngtransfer` bridge.

```
Stage 1 (from dev machine):  source account (with SAS)  →  orngtransfer/data/speech/...
Stage 2 (from dev machine):  orngtransfer/data/speech/...  →  orngwus2cresco/data/speech/...
```

Both stages run from the dev machine with green tenant login. No orange pod access needed.

Ref: [Penny wiki: Transfers Orange <=> Green](~/code/Penny.wiki/Orange-wiki/Project-Orange/Models-and-data/Transfers%3A-Orange-%3C=%3E-Green.md), [Transfers Corp <=> Orange](~/code/Penny.wiki/Orange-wiki/Project-Orange/Models-and-data/Transfers%3A-Corp-%3C=%3E-Orange.md)

## URL Reference

| Role | Base URL |
|------|----------|
| Source | `https://<source-account>.blob.core.windows.net/<container>/<relative_path>` |
| Stage 1 dest / Stage 2 source | `https://orngtransfer.blob.core.windows.net/data/speech/<relative_path>` |
| Final dest | `https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>` |

## Prerequisites

### Auth setup

**Microsoft corp tenant** supplies a read/list SAS for the exact source account and container. The refresh helper derives both values from the source URL, updates the cache without printing the token, and restores the Green login:
```bash
python .github/skills/transfer-to-orange/scripts/refresh_source_sas.py \
  --source-url "https://<source-account>.blob.core.windows.net/<container>/<relative_path>/"
```

Run it in a terminal that can open and surface Azure CLI's normal browser authentication. **Never use `az login --use-device-code` or any other device-code flow.** If the browser cannot open, the callback expires, or browser authentication fails, stop and ask the user to complete a fresh browser-based `az login --tenant <tenant>`; do not fall back to device code. Never print the SAS or embed it in commentary/final responses.

**Green tenant** (for `orngtransfer` and final Orange endpoints):
```bash
# Switch az CLI back to green tenant for AZCOPY_AUTO_LOGIN_TYPE=AZCLI
az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0
```

**Important:** All azcopy commands to `orngtransfer` and Orange endpoints must use:
```bash
export AZCOPY_AUTO_LOGIN_TYPE=AZCLI
```
This makes azcopy reuse the `az login` session. No separate `azcopy login` needed.

## Procedure

### Step 0 — Compute the relative path

The user provides a source path. If the path starts with `/datablob1/`, strip that prefix to get the relative path. Otherwise, use as-is.

**Example:**
- Full source path: `/datablob1/am_data/publicdataset_process/SLR94_.../mls_english/train/Chunkfiles/`
- Relative path: `am_data/publicdataset_process/SLR94_.../mls_english/train/Chunkfiles/`

**Verify the path exists:**
```bash
azcopy list "https://<source-account>.blob.core.windows.net/<container>/<relative_path>?<SAS>" | head -5
```

### Step 1 — Load or refresh the source SAS

Derive the container cache key from the source URL, for example `https://highperf01eus2.blob.core.windows.net/data`. Inspect only whether an entry exists and its `expiry_utc`; do not print the full cache because it contains secrets.

```bash
jq --arg key "https://<source-account>.blob.core.windows.net/<container>" \
  '{has_sas: ((.[$key].sas // "") | length > 0), expiry_utc: .[$key].expiry_utc}' \
  /home/boren/.sas/azure_blob_sas_cache.json
```

If the entry is missing, expired, has less than 30 minutes remaining, or source listing fails with a SAS authentication error, run the refresh helper and retry the source listing once. Do not repeatedly refresh on path-not-found, authorization, or other non-expiry errors.

```bash
python .github/skills/transfer-to-orange/scripts/refresh_source_sas.py \
  --source-url "<full-source-URL>"
```

The helper switches to the Microsoft corp tenant with browser authentication, generates a seven-day `rl` SAS for the exact source container, atomically updates `/home/boren/.sas/azure_blob_sas_cache.json` with mode `0600`, and restores the Green tenant with browser authentication. If browser login needs user interaction, surface the browser prompt and wait. If it fails or the account denies SAS generation, report that exact blocker and stop; never use device code or fall back to another account or credential.

Construct full URLs with SAS by appending `?<sas_value>` to the blob URL.

Only the external source blob account uses a SAS token:
- `https://<source-account>.blob.core.windows.net/<container>`

The bridge and Orange endpoints (`orngtransfer`, `orngwus2cresco`) authenticate via **`AZCOPY_AUTO_LOGIN_TYPE=AZCLI`**, which reuses the `az login` session (green tenant). Ensure the user has run `az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0` and set `export AZCOPY_AUTO_LOGIN_TYPE=AZCLI` before proceeding.

### Step 2 — Test with 1 file first

Before copying the full dataset, always test with a single file.

```bash
# List files at source to pick one
azcopy list "https://<source-account>.blob.core.windows.net/<container>/<relative_path>?<SAS>" | head -5

# Stage 1: Copy 1 file to orngtransfer
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://<source-account>.blob.core.windows.net/<container>/<relative_path>/<one_file>?<SAS_source>" \
  "https://orngtransfer.blob.core.windows.net/data/speech/<relative_path>/<one_file>" \
  --s2s-preserve-access-tier=false

# Stage 2: Copy the same file to its final Orange path
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://orngtransfer.blob.core.windows.net/data/speech/<relative_path>/<one_file>" \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>/<one_file>" \
  --s2s-preserve-access-tier=false
```

Verify the single file at the final Orange path before proceeding with the full copy.

### Step 3 — Stage 1: azcopy source → orngtransfer

**CRITICAL:** Each async terminal is a new shell session. The `$SAS` variable from a previous terminal is NOT inherited. You must `export SAS="..."` at the top of every async terminal command block before using `$SAS` in URLs. If `$SAS` is empty, azcopy falls back to OAuth with the green tenant against the corp source and fails with `401 InvalidAuthenticationInfo / Issuer validation failed`.

```bash
export SAS="<sas_value_from_cache>"

AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://<source-account>.blob.core.windows.net/<container>/<relative_path>/*?${SAS}" \
  "https://orngtransfer.blob.core.windows.net/data/speech/<relative_path>/" \
  --s2s-preserve-access-tier=false --recursive
```

- Run in async terminal — large datasets may take hours.
- Monitor with `azcopy jobs list` and `azcopy jobs show <job-id>`.

### Step 4 — Stage 2: azcopy orngtransfer → orngwus2cresco

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://orngtransfer.blob.core.windows.net/data/speech/<relative_path>/*" \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>/" \
  --s2s-preserve-access-tier=false --recursive
```

This runs directly from the dev machine — no orange pod needed.

If a job ends with transient `connection reset by peer` failures, resume only its failed transfers:

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy jobs resume <job-id>
```

### Step 5 — Verify

```bash
bbb ls az://orngwus2cresco/data/speech/<relative_path>/
```

Compare file count and sizes against the source listing from Step 2.

## Important Notes

- **Always use `azcopy`** for all copy operations (not bbb, sciclone, or az cli).
- **Always use `orngtransfer/data` as the bridge for datasets transferred from outside Orange.**
- **Always set `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`** for `orngtransfer` and Orange endpoints (reuses `az login` session).
- **Always add `--s2s-preserve-access-tier=false`** — the bridge and Orange storage accounts don't support blob access tiers.
- **Only source blobs use SAS tokens** from `/home/boren/.sas/azure_blob_sas_cache.json`. `orngtransfer` and Orange endpoints use `az login` (green tenant).
- **Refresh stale source SAS automatically** with the bundled helper, then retry source listing once. The helper must restore the Green tenant before transfer work continues.
- **Browser login only:** never run Azure CLI with `--use-device-code`. If normal browser authentication fails, stop and request a fresh browser login instead of changing authentication modes.
- **Always test with 1 file** before doing the full recursive copy.
- Source accounts and containers vary. Do not substitute `highperf01safn/data`; derive them from the supplied URL. A `data` container does not always have the `datablob1/` prefix, so verify the actual path with `azcopy list` first.
- **Run long copies in async terminal mode** — they can take hours for large datasets.
- The relative path is computed by stripping `/datablob1/` from the source path.
- Ensure trailing `/` on destination directory URLs.
- `azcopy` is installed at `/usr/local/bin/azcopy` (v10.29.1).
- **`$SAS` must be set in every new terminal session** — async terminals do NOT inherit shell variables from previous terminals. Always `export SAS="..."` inline before using `${SAS}` in azcopy URLs.
- **Common 401 error**: If you see `InvalidAuthenticationInfo / Issuer validation failed` on the source blob, it means `$SAS` is empty and azcopy is trying OAuth (green tenant) against the corp source. Fix: ensure `$SAS` is exported in the current terminal.

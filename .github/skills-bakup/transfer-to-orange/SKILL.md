---
name: transfer-to-orange
description: "Transfer data from highperf01safn blob to Orange (orng) via azcopy through GRN gateway. Use when: copying data to orange, transfer to orng, azcopy blob to orange, highperf to orngtransfer, two-stage blob transfer, copy dataset to orange storage."
argument-hint: "<path_under_datablob1> (e.g. am_data/publicdataset_process/SLR94_.../mls_english/train/Chunkfiles/)"
---

# Transfer Data to Orange via azcopy

Two-stage azcopy transfer from highperf01safn blob storage to Orange (orng) blob storage, routed through the GRN gateway.

## When to Use

- Transfer datasets, audio data, or models from highperf01safn to Orange storage
- User says "copy to orange", "transfer to orng", "move data to orange blob"
- Two-stage pipeline: highperf01safn → grngenaiexternal → orngwus2cresco (all from dev machine)

## Architecture

There is no direct Corp → Orange path. Data flows: Corp → Green → Orange.

```
Stage 1 (from dev machine):  highperf01safn (with SAS)  →  grngenaiexternal/inbound/speech/...
Stage 2 (from dev machine):  grngenaiexternal/inbound/speech/...  →  orngwus2cresco/data/speech/...
```

Both stages run from the dev machine with green tenant login. No orange pod access needed.

Ref: [Penny wiki: Transfers Corp <=> Orange](~/code/Penny.wiki/Orange-wiki/Project-Orange/Models-and-data/Transfers%3A-Corp-%3C=%3E-Orange.md), [Tangerine: Upload Model and Data](~/code/Penny.wiki/Orange-wiki/Tangerine/Upload-Model-and-Data.md)

## URL Reference

| Role | Base URL |
|------|----------|
| Source | `https://highperf01safn.blob.core.windows.net/data/<relative_path>` |
| Stage 1 dest / Stage 2 source | `https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>` |
| Final dest | `https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>` |

## Prerequisites

### Auth setup

**Microsoft corp tenant** (for highperf01safn SAS token):
```bash
# Switch to corp tenant and regenerate SAS
az login --tenant microsoft.com
EXPIRY=$(date -u -d '+7 days' '+%Y-%m-%dT%H:%MZ')
SAS=$(az storage container generate-sas --as-user --auth-mode login \
  --account-name highperf01safn --name data --permissions rl --expiry "$EXPIRY" -o tsv)
echo "$SAS"
```
Update the SAS in `/home/boren/.sas/azure_blob_sas_cache.json` manually (or use `download_blob_with_sas.py` which auto-regenerates).

**Green tenant** (for GRN/orng destination endpoints):
```bash
# Switch az CLI back to green tenant for AZCOPY_AUTO_LOGIN_TYPE=AZCLI
az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0
```

**Important:** All azcopy commands to GRN/orng destinations must use:
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
azcopy list "https://highperf01safn.blob.core.windows.net/data/<relative_path>?<SAS>" | head -5
```

### Step 1 — Load SAS tokens

Read SAS tokens from the cache file for all endpoints that need them:

```bash
cat /home/boren/.sas/azure_blob_sas_cache.json
```

Extract the `sas` value for each endpoint key. **Check `expiry_utc`** — if expired, warn the user and stop. They need to regenerate the token first.

Construct full URLs with SAS by appending `?<sas_value>` to the blob URL.

Only source blob accounts use SAS tokens:
- `https://highperf01safn.blob.core.windows.net/data`

The GRN and orng endpoints (`grngenaiexternal`, `orngwus2cresco`) authenticate via **`AZCOPY_AUTO_LOGIN_TYPE=AZCLI`** which reuses the `az login` session (green tenant). Ensure the user has run `az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0` and set `export AZCOPY_AUTO_LOGIN_TYPE=AZCLI` before proceeding.

### Step 2 — Test with 1 file first

Before copying the full dataset, always test with a single file.

```bash
# List files at source to pick one
azcopy list "https://highperf01safn.blob.core.windows.net/data/<relative_path>?<SAS>" | head -5

# Stage 1: Copy 1 file to grngenaiexternal
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://highperf01safn.blob.core.windows.net/data/<relative_path>/<one_file>?<SAS_highperf>" \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/<one_file>" \
  --s2s-preserve-access-tier=false
```

Verify the single file transferred successfully before proceeding with the full copy.

### Step 3 — Stage 1: azcopy source → grngenaiexternal

**CRITICAL:** Each async terminal is a new shell session. The `$SAS` variable from a previous terminal is NOT inherited. You must `export SAS="..."` at the top of every async terminal command block before using `$SAS` in URLs. If `$SAS` is empty, azcopy falls back to OAuth with the green tenant against the corp source and fails with `401 InvalidAuthenticationInfo / Issuer validation failed`.

```bash
export SAS="<sas_value_from_cache>"

AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://highperf01safn.blob.core.windows.net/data/<relative_path>/*?${SAS}" \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/" \
  --s2s-preserve-access-tier=false --recursive
```

- Run in async terminal — large datasets may take hours.
- Monitor with `azcopy jobs list` and `azcopy jobs show <job-id>`.

### Step 4 — Stage 2: azcopy grngenaiexternal → orngwus2cresco

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/*" \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>/" \
  --s2s-preserve-access-tier=false --recursive
```

This runs directly from the dev machine — no orange pod needed.

### Step 5 — Verify

```bash
bbb ls az://orngwus2cresco/data/speech/<relative_path>/
```

Compare file count and sizes against the source listing from Step 2.

## Important Notes

- **Always use `azcopy`** for all copy operations (not bbb, sciclone, or az cli).
- **Always set `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`** for GRN/orng destinations (reuses `az login` session).
- **Always add `--s2s-preserve-access-tier=false`** — GRN/orng storage accounts don't support blob access tiers.
- **Only source blobs use SAS tokens** from `/home/boren/.sas/azure_blob_sas_cache.json`. GRN/orng endpoints use `az login` (green tenant).
- **Check SAS expiry** before starting — if expired, regenerate via `az login --tenant microsoft.com` (see Prerequisites).
- **Always test with 1 file** before doing the full recursive copy.
- The `data` container in highperf01safn does NOT always have the `datablob1/` prefix — verify the actual path with `azcopy list` first.
- **Run long copies in async terminal mode** — they can take hours for large datasets.
- The relative path is computed by stripping `/datablob1/` from the source path.
- Ensure trailing `/` on destination directory URLs.
- `azcopy` is installed at `/usr/local/bin/azcopy` (v10.29.1).
- **`$SAS` must be set in every new terminal session** — async terminals do NOT inherit shell variables from previous terminals. Always `export SAS="..."` inline before using `${SAS}` in azcopy URLs.
- **Common 401 error**: If you see `InvalidAuthenticationInfo / Issuer validation failed` on the source blob, it means `$SAS` is empty and azcopy is trying OAuth (green tenant) against the corp source. Fix: ensure `$SAS` is exported in the current terminal.

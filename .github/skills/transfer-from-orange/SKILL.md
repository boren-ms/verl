---
name: transfer-from-orange
description: "Transfer data from Orange (orngwus2cresco) blob back to corp tsstd01safn via azcopy through the GRN gateway. Use when: copying from orange, transfer from orng, orange to corp, orng to tsstd01safn, download orange dataset to corp, pull eval results from orange, two-stage blob transfer back, azcopy orange to corp, reverse transfer to orange."
argument-hint: "<path_under_speech> (e.g. users/boren/eval_results/openasr/...)"
---

# Transfer Data from Orange via azcopy

Two-stage azcopy transfer from Orange (`orngwus2cresco`) blob storage back to corp (`tsstd01safn`), routed through the GRN gateway. This is the inverse of the `transfer-to-orange` skill.

## When to Use

- Pull eval results, trained checkpoints, datasets, or models from Orange back to corp `tsstd01safn`
- User says "copy from orange", "transfer from orng", "pull data back from orange", "orange to corp", "orng to highperf"
- Two-stage pipeline: `orngwus2cresco` → `grngenaiexternal/inbound/speech/...` → `tsstd01safn/data/...` (all from dev machine)

## Architecture

There is no direct Orange → Corp path. Data flows: Orange → Green → Corp.

```
Stage 1 (from dev machine):  orngwus2cresco/data/speech/...     →  grngenaiexternal/inbound/speech/...
Stage 2 (from dev machine):  grngenaiexternal/inbound/speech/...  →  tsstd01safn/data/...  (write SAS)
```

Both stages run from the dev machine. The green devbox already has read access to `orngwus2cresco`, so no orange pod access is needed.

Ref: [Penny wiki: Transfers Corp <=> Orange](~/code/Penny.wiki/Orange-wiki/Project-Orange/Models-and-data/Transfers%3A-Corp-%3C=%3E-Orange.md) (section "Moving data from Orange to Corp"), [Transfers: Orange <=> Green](~/code/Penny.wiki/Orange-wiki/Project-Orange/Models-and-data/Transfers%3A-Orange-%3C=%3E-Green.md)

## URL Reference

| Role | Base URL |
|------|----------|
| Source | `https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>` |
| Stage 1 dest / Stage 2 source | `https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>` |
| Final dest | `https://tsstd01safn.blob.core.windows.net/data/<relative_path>` |

Note the asymmetry: orange stores under `data/speech/...` while tsstd01safn stores under `data/...` (no `speech/` segment). The relative path is the part **after** `speech/` on orange and the part **after** `data/` on corp.

## Prerequisites

### Auth setup

**Green tenant** (for `orngwus2cresco` source AND `grngenaiexternal` staging — both ends of Stage 1 and the source of Stage 2):
```bash
az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0
export AZCOPY_AUTO_LOGIN_TYPE=AZCLI
```
This makes azcopy reuse the `az login` session. No separate `azcopy login` needed.

**Microsoft corp tenant** (for `tsstd01safn` **write** SAS — the Stage 2 destination):
```bash
# Switch to corp tenant and generate a WRITE-capable SAS
az login --tenant microsoft.com
EXPIRY=$(date -u -d '+7 days' '+%Y-%m-%dT%H:%MZ')
SAS=$(az storage container generate-sas --as-user --auth-mode login \
  --account-name tsstd01safn --name data --permissions rwcl --expiry "$EXPIRY" -o tsv)
echo "$SAS"
```
**Important:** check the cache entry for `tsstd01safn` in `/home/boren/.sas/azure_blob_sas_cache.json` — the `permissions` field (or the embedded `sp=` query param) **must include `w` and `c`**. A read-only `rl` SAS will return **403 AuthorizationPermissionMismatch** on writes. If missing, regenerate with `--permissions rwcl` (read, write, create, list) and either update the cache entry in place or keep a separate `tsstd01safn_write` entry — your choice.

After generating, switch back to the green tenant for the actual azcopy run:
```bash
az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0
```

## Procedure

### Step 0 — Compute the relative path

The user provides a source path under orange. Strip the leading container/prefix to get the relative path.

**Example:**
- Full orange path: `az://orngwus2cresco/data/speech/users/boren/eval_results/openasr/run42/`
- Relative path: `users/boren/eval_results/openasr/run42/`
- Final corp path: `https://tsstd01safn.blob.core.windows.net/data/users/boren/eval_results/openasr/run42/`

**Verify the source path exists** (green tenant, AZCLI auth):
```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy list \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>" | head -5
```

### Step 1 — Load and validate the corp write SAS

Read the SAS for `tsstd01safn` from the cache:
```bash
cat /home/boren/.sas/azure_blob_sas_cache.json
```

Check three things on the entry:
1. `expiry_utc` is in the future. If expired, regenerate (see Prerequisites) before continuing.
2. `permissions` (or the embedded `sp=` query param) contains `w` and `c`. If it only has `rl`, regenerate with `--permissions rwcl`.
3. `account_name` is `tsstd01safn` and container is `data`.

Only the corp destination uses a SAS token in this direction. `orngwus2cresco` and `grngenaiexternal` authenticate via `AZCOPY_AUTO_LOGIN_TYPE=AZCLI` (green tenant `az login`).

### Step 2 — Test with 1 file first

Before copying the full tree, always test with a single file end-to-end.

```bash
export SAS="<sas_value_from_cache>"

# Pick one file from the source listing
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy list \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>" | head -5

# Stage 1: orange → grn (one file)
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>/<one_file>" \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/<one_file>" \
  --s2s-preserve-access-tier=false

# Stage 2: grn → corp (one file) — SAS goes on the DESTINATION
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/<one_file>" \
  "https://tsstd01safn.blob.core.windows.net/data/<relative_path>/<one_file>?${SAS}" \
  --s2s-preserve-access-tier=false
```

Verify the single file landed on corp before proceeding with the full copy.

### Step 3 — Stage 1: azcopy orngwus2cresco → grngenaiexternal

**CRITICAL:** Each async terminal is a new shell session. The `$SAS` variable from a previous terminal is NOT inherited. Always `export SAS="..."` at the top of every async terminal command block before using `${SAS}` in URLs. (In Stage 1 you don't need `$SAS`, but you'll need it for Stage 2.)

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/<relative_path>/*" \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/" \
  --s2s-preserve-access-tier=false --recursive
```

- Run in async terminal — large datasets may take hours.
- Monitor with `azcopy jobs list` and `azcopy jobs show <job-id>`.

### Step 4 — Stage 2: azcopy grngenaiexternal → tsstd01safn

```bash
export SAS="<sas_value_from_cache>"   # required in every new async terminal

AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/*" \
  "https://tsstd01safn.blob.core.windows.net/data/<relative_path>/?${SAS}" \
  --s2s-preserve-access-tier=false --recursive
```

In this direction the SAS is on the **destination URL**, not the source.

### Step 5 — Verify

```bash
# Listing via SAS
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy list \
  "https://tsstd01safn.blob.core.windows.net/data/<relative_path>?${SAS}" | head

# Or, if you have a corp-side bbb mount configured
bbb ls az://tsstd01safn/data/<relative_path>/
```

Compare the file count and sizes against the source listing from Step 0.

## Important Notes

- **SAS goes on the DESTINATION in this direction**, not the source. A read-only `rl` SAS will fail with `403 AuthorizationPermissionMismatch` — regenerate with `--permissions rwcl`.
- **Always use `azcopy`** for all copy operations (not bbb, sciclone, or az cli).
- **Always set `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`** so azcopy reuses the green-tenant `az login` session for `orngwus2cresco` and `grngenaiexternal`.
- **Always add `--s2s-preserve-access-tier=false`** — the GRN intermediate doesn't expose blob access tier metadata, so the corp destination would otherwise reject the copy. `tsstd01safn` supports tiers but it's fine to land with the default tier.
- **Only the corp destination uses a SAS token** from `/home/boren/.sas/azure_blob_sas_cache.json`. `orngwus2cresco` and `grngenaiexternal` use `az login` (green tenant).
- **Check SAS expiry and permissions** before starting — if missing `w`/`c`, regenerate via `az login --tenant microsoft.com` (see Prerequisites).
- **Always test with 1 file** end-to-end (both stages) before doing the full recursive copy.
- **Run long copies in async terminal mode** — they can take hours for large datasets.
- The relative path on orange lives under `data/speech/...`, but on corp it lives under `data/...` (no `speech/` segment). Don't double up the `speech/` segment on the corp destination.
- Ensure trailing `/` on destination directory URLs (place `?${SAS}` after the trailing slash on Stage 2).
- `azcopy` is installed at `/usr/local/bin/azcopy` (v10.29.1).
- **`$SAS` must be exported in every new terminal session** — async terminals do NOT inherit shell variables from previous terminals. Always `export SAS="..."` inline before using `${SAS}` in azcopy URLs.
- **Skip Stage 2 to land on local disk:** if you only need the data on the dev machine, replace the Stage 2 destination with a local path:
  ```bash
  AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
    "https://grngenaiexternal.blob.core.windows.net/inbound/speech/<relative_path>/*" \
    "/datablob1/<relative_path>/" --recursive
  ```
- The write SAS is container-scoped (`--name data`) to limit blast radius; use account-scoped only if you need to write across multiple containers.

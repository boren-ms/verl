---
name: sciclone-cross-region-copy
description: "Copy Azure blob data across regions using sciclone copy. Use when syncing data, audio cache, models, or datasets to other Azure regions (e.g. southcentralus, uksouth, westus2). Extracts az:// paths from YAML configs or user input, generates a shell script, and runs sciclone copy for each path × region."
argument-hint: "Config YAML path or az:// paths, plus target regions"
---

# Sciclone Cross-Region Copy

Copy Azure blob storage data to other regions using the `sciclone copy` CLI tool.

## When to Use
- User wants to replicate data (audio cache, datasets, models) across Azure regions
- User says "copy to other regions", "sync to scus/uks", "replicate data"
- User provides a YAML config with `az://` paths and wants them in additional regions
- User provides `az://` paths directly and target region names

## Key Facts
- `sciclone copy <source_az_path> <destination_region>` — copies data to a region
- `--follow` (default) waits for transfer; `--no-follow` returns immediately
- Destination is a **region name** (e.g. `southcentralus`, `uksouth`, `westus2`), not a storage account
- Storage account naming convention: `orng<region_short>cresco` (e.g. `orngwus2cresco`, `orngscuscresco`, `orngukscresco`)

## Procedure

### Step 1 — Gather source paths
Extract `az://` paths from the user's input:
- If a **YAML config** is provided, read it and extract all `cache_path`, `data_dir`, or other `az://` fields
- If **direct paths** are given, use them as-is
- Confirm the paths exist if uncertain (use `bbb ls <path>`)

### Step 2 — Determine target regions
Get target regions from the user. Common region names:
| Short | Full region name |
|-------|-----------------|
| wus2  | westus2         |
| scus  | southcentralus  |
| uks   | uksouth         |

Use the **full region name** with `sciclone copy` (e.g. `southcentralus`, not `scus`).

### Step 3 — Generate and run the copy script
Update or create `sciclone_copy.sh` in the workspace root with:

```bash
#!/bin/bash
set -x

paths=(
  az://account/container/path1/
  az://account/container/path2/
)

targets=(southcentralus uksouth)

for tgt in "${targets[@]}"; do
  for src in "${paths[@]}"; do
    echo "=== Copying ${src} to ${tgt} ==="
    sciclone copy "${src}" "${tgt}"
  done
done
```

- Ensure trailing `/` on directory paths
- Run as a **background terminal** since copies can be long-running
- Use `--no-follow` if the user wants to fire-and-forget

### Step 4 — Monitor progress
Use sciclone's operation tracking commands:

```bash
# List all recent operations (last 24h) with status
sciclone list

# Follow a specific operation in real-time (by ID from list output)
sciclone follow <operation-id>

# Show detailed status of a specific operation
sciclone show <operation-id>

# Cancel a running operation
sciclone cancel <operation-id>
```

- `sciclone list` shows all recent copy/clone/delete operations with ✅/⏳ status
- When the user asks about progress, run `sciclone list` to check
- Use `sciclone show <id>` for detailed info on a specific transfer

### Step 5 — Verify
- Confirm all operations show ✅ in `sciclone list`
- Optionally verify with `bbb ls az://orng<region>cresco/...` at the destination

---
name: blob-download-sync
description: "Download files from an Azure Blob URL to local disk, then sync to orng blob storage. Use when: downloading eval results, syncing model outputs to orng, copying blob data to az://orngwus2cresco, download and sync Azure blob, blob to orng transfer."
argument-hint: "<source_blob_url> <orng_dest_path>"
---

# Blob Download & Sync to Orng

Download files from an Azure Blob Storage URL to a local staging directory, then sync them to orng blob storage (`az://orngwus2cresco/data/boren/data/...`).

## When to Use

- Download eval outputs, model results, or datasets from Azure blob and push to orng.
- Two-step transfer: Azure blob → local → orng blob.
- Bulk file sync from any `https://*.blob.core.windows.net/` URL to orng.

## Procedure

### Step 1: Download from Azure Blob to Local

Use `scripts/download_blob_with_sas.py` to download from the source blob URL to a local path (default staging: `~/data/download`).

```bash
python scripts/download_blob_with_sas.py \
  --url "<source_blob_url>" \
  --dest "<local_dest_path>"
```

- The script auto-generates and caches SAS tokens via Azure CLI if none provided.
- Files already downloaded with matching size are skipped.
- Default 8 parallel workers.
- By default, the blob prefix is stripped so files land directly in `--dest`. Use `--keep-prefix` to preserve the full blob directory structure.

After download, the files are directly in `<local_dest_path>/` when using `--strip-prefix`.

### Step 2: Sync Local to Orng Blob

Use `bbb sync` to push the downloaded files to the orng destination:

```bash
bbb sync <local_source_dir> <orng_dest_path>
```

Example:
```bash
bbb sync ~/data/download/projects/foo/bar/ az://orngwus2cresco/data/boren/data/results/my-model/bar/
```

- `bbb sync` copies files that are missing or changed at the destination.
- Add `--delete` to remove destination files not present in source.
- Add `-x '<regex>'` to exclude files matching a pattern.

## Example End-to-End

```bash
# 1. Download eval output from Azure blob (files land directly in dest by default)
python scripts/download_blob_with_sas.py \
  --url "https://tsstd01safn.blob.core.windows.net/data/projects/phi-fastllm/amlt-results/my-run/85000/eval_output/sr_openasr_ml/" \
  --dest "~/data/download/sr_openasr_ml"

# 2. Sync to orng
bbb sync ~/data/download/sr_openasr_ml/ \
  az://orngwus2cresco/data/boren/data/results/my-run/sr_openasr_ml/
```

## Notes

- The user specifies both the local staging path and the orng destination path.
- Ask the user for the orng destination if not provided.
- Verify the download completed (check file count) before starting the sync.

---
name: blob-split-fast-download
description: "Fast-download a single large file from Azure Blob (e.g. orng) to local by splitting it into <50MB zip parts on a remote brix node (in-region with the blob), uploading the parts back to blob, then downloading parts in parallel locally and reassembling with md5 verification. Use when: bbb cp of a single large pt/bin file from az://orngwus2cresco is too slow, transferring 1-100GB single files from orange blob to corp dev box, accelerating one-file blob downloads via parallel split chunks. Triggers: \"download is slow\", \"split and download in parallel\", \"speed up blob download\", \"fast pull large file from orng\"."
argument-hint: "<blob_url_of_file> [local_dest_dir] [remote_node]"
---

# Fast Blob Download via Remote-Side Zip Split

A single `bbb cp` of a large file from orng blob to the local dev box is often slow (single TCP stream + far region). This skill accelerates it by:

1. Doing the work on a remote brix node that is **in the same Azure region** as the blob (e.g. `verl-n1-i*` for orng `westus2`), where blob I/O is ~LAN-fast.
2. Splitting the file into many ≤48 MB zip parts on the node.
3. Uploading the parts back to blob next to the original.
4. Downloading the parts to the local box **in parallel** (many small files >> one huge file over the WAN).
5. Reassembling and md5-verifying.

## When to Use

- A single file (typically `.pt`, `.bin`, `.safetensors`) of 1-100 GB lives on `az://orngwus2cresco/...` and `bbb cp` to local is too slow.
- You have at least one Ready brix node in-region with the blob (`prod-westus2-cw-*` for orng).

Do NOT use this for many small files — plain `bbb sync --concurrency N` is already parallel across files.

## Procedure

### 0. Pick an in-region remote node

```bash
brix pools --all 2>&1 | grep Ready | head
```

Pick any Ready `verl-n1-*` node in `prod-westus2-*` (same region as orng). Confirm tools:

```bash
brix ssh <NODE> -- 'which azcopy az; az account show -o tsv --query name'
```

Remote auth: `az` CLI is logged in as a service principal on the node. `azcopy login --login-type=MSI` does NOT work (IMDS blocked), and `blobfile`/`bbb` on the node have no credentials. Use `az storage blob ... --auth-mode login`, which uses the SP token.

### 1. Remote: download, split, upload parts

Write this script and run via `brix ssh <NODE> -- 'bash /tmp/remote_split.sh'`:

```bash
#!/usr/bin/env bash
set -euo pipefail
WORK=/tmp/delta_xfer            # tmp dir on remote
ACCT=orngwus2cresco
CONT=data
BLOB=boren/outputs/.../delta.pt         # blob path under container
DST_DIR=boren/outputs/.../delta_split   # where to put split parts

mkdir -p "$WORK" && cd "$WORK"

echo "=== Downloading source ==="
az storage blob download --auth-mode login --account-name $ACCT --container-name $CONT \
  --name "$BLOB" --file delta.pt --max-connections 32 --no-progress -o none

md5sum delta.pt | tee delta.md5

echo "=== zip split (48m, store-only) ==="
rm -rf splits && mkdir splits && cd splits
zip -s 48m -0 -q delta.zip ../delta.pt   # produces delta.z01..delta.zNN + delta.zip
cd ..

echo "=== Clearing remote split dir ==="
az storage blob delete-batch --auth-mode login --account-name $ACCT --source $CONT \
  --pattern "$DST_DIR/*" -o none 2>/dev/null || true

echo "=== Uploading splits in parallel ==="
az storage blob upload-batch --auth-mode login --account-name $ACCT --destination $CONT \
  --destination-path "$DST_DIR" --source splits --max-connections 16 --overwrite -o none

md5sum splits/* | sed 's|splits/||' > splits.md5
az storage blob upload --auth-mode login --account-name $ACCT --container-name $CONT \
  --name "$DST_DIR/splits.md5" --file splits.md5 --overwrite -o none
az storage blob upload --auth-mode login --account-name $ACCT --container-name $CONT \
  --name "$DST_DIR/delta.md5"  --file delta.md5  --overwrite -o none
```

Notes:
- `zip -s 48m -0` = split at 48 MB (under the 50 MB request), stored (no compression — weights don't compress anyway).
- The last part is named `delta.zip` (contains the central directory); the rest are `delta.z01 … delta.zNN`.
- Use `--auth-mode login` everywhere; never rely on MSI/IMDS on brix pods.

### 2. Local: parallel download + reassemble + md5 verify

```bash
#!/usr/bin/env bash
set -euo pipefail
DST=${1:-/tmp/delta_dl}
SRC=az://orngwus2cresco/data/boren/outputs/.../delta_split
mkdir -p "$DST/splits" && cd "$DST"

bbb cp "$SRC/delta.md5"  delta.md5
bbb cp "$SRC/splits.md5" splits/splits.md5

bbb ls "$SRC/" | grep -E 'delta\.(z[0-9]+|zip)$' > parts.list
N=$(wc -l < parts.list); echo "$N parts"

t0=$SECONDS
xargs -a parts.list -n 1 -P 16 -I {} bbb cp --concurrency 4 -q "{}" splits/
echo "Download took $((SECONDS - t0))s"

(cd splits && md5sum -c splits.md5 | tail -3)

# CRITICAL: reassemble by concatenating parts in order, then `unzip -p`.
# `zip -s 0 --out` and `zip -F` only rebuild metadata; they do NOT copy split
# payloads. Plain cat works because `zip -s` stores data sequentially across
# z01..zNN with the central directory in the trailing .zip file.
( cd splits && cat $(ls delta.z[0-9][0-9] | sort) delta.zip ) > joined.zip
unzip -p joined.zip ../delta.pt > delta.pt 2>/dev/null || true   # unzip prints expected split warnings

md5sum -c delta.md5 && echo "MD5 OK"

rm -rf splits joined.zip parts.list
ls -lh "$DST"
```

Tunable: `xargs -P 16` × `bbb cp --concurrency 4` — local bandwidth is usually the bottleneck, not blob.

### 3. Move to final location

```bash
mkdir -p ~/data/ckp/<run>/<step>
mv /tmp/delta_dl/delta.pt ~/data/ckp/<run>/<step>/delta.pt
mv /tmp/delta_dl/delta.md5 ~/data/ckp/<run>/<step>/delta.md5
(cd ~/data/ckp/<run>/<step> && md5sum -c delta.md5)
```

### 4. Cleanup

```bash
# remote tmp
brix ssh <NODE> -- 'rm -rf /tmp/delta_xfer /tmp/remote_split.sh'
# blob splits (optional — keep if you'll re-pull again)
bbb rmtree az://orngwus2cresco/.../delta_split/
```

## Gotchas

- **Do not** try to reassemble with `zip -s 0 --out`, `zip -F`, or `unzip` directly on `delta.zip` — they only see the metadata and produce a tiny (~tens of MB) bogus archive. Always `cat` the parts in order first.
- `unzip` will print warnings like "extra bytes at beginning" and "bad zipfile offset (attempting to re-compensate)" — these are expected for multi-part zips and `unzip -p` still extracts correctly. Verify by md5.
- On brix nodes: no `pip` in `$PATH`, no IMDS, no `bbb` credentials. Use `/opt/conda/bin/python` if you need Python, and `az storage blob --auth-mode login` for blob ops.
- File-size sanity: the sum of all part sizes must equal the original file size; `splits.md5` lets you catch a corrupted part before reassembly.
- Naming assumes ≤99 split parts (z01..z99). For larger files, use `zip -s 100m` or switch to `split -b 48m` + plain `cat` (drop the zip layer entirely).

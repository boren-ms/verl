---
name: copy-to-corp-blob
description: "Upload local files/dirs from the dev machine to corp tsstd01safn blob storage via azcopy with a write SAS. Default mapping: ~/data/<path>  ⇄  https://tsstd01safn.blob.core.windows.net/data/users/boren/data/<path>. Use when: copying local files to corp blob, uploading checkpoints/results to tsstd01safn, syncing ~/data to corp, push to safn, local to corp blob, copy to azure speech blob."
argument-hint: "<local_path> [remote_subpath]   e.g. ~/data/ckp/foo  (defaults to users/boren/data/ckp/foo)"
---

# Copy Local Data to Corp Blob (tsstd01safn)

Single-stage azcopy upload from the local dev machine to corp `tsstd01safn`. This is a direct local → corp transfer using a write-capable SAS — **no GRN/orange gateway needed**.

## When to Use
- Push local checkpoints, eval outputs, or datasets from `~/data/...` to corp blob
- User says: "copy to corp blob", "upload to tsstd01safn", "push to azure speech blob", "sync ~/data to corp", "copy to remote azure blob"
- Direct local → corp path (no orange involvement)

## Default Path Mapping

| Local | Corp blob |
|-------|-----------|
| `~/data/<subpath>` | `https://tsstd01safn.blob.core.windows.net/data/users/boren/data/<subpath>` |

So:
- `~/data/ckp/foo/` → `data/users/boren/data/ckp/foo/`
- `~/data/eval_results/run42/` → `data/users/boren/data/eval_results/run42/`

The user may override the remote subpath. Otherwise compute it by stripping the leading `~/data/` (or `/home/boren/data/`) from the local path and prepending `users/boren/data/`.

## Prerequisites

### azcopy & az login
- `azcopy` at `/usr/local/bin/azcopy` (v10.29+).
- Green tenant login is fine for the actual transfer source side (it's local files), but the **destination SAS must be issued from the corp tenant**.

### Corp write SAS (REQUIRED)

The cached SAS at `/home/boren/.sas/azure_blob_sas_cache.json` is often **read-only** (`sp=rl`). Writes will fail with `403 AuthorizationPermissionMismatch`.

**Always check `sp` first**:
```bash
python3 -c "
import json, urllib.parse
v = json.load(open('/home/boren/.sas/azure_blob_sas_cache.json'))['https://tsstd01safn.blob.core.windows.net/data']
sas = v['sas'] if isinstance(v, dict) else v
qs = urllib.parse.parse_qs(sas.lstrip('?'))
print('sp =', qs.get('sp'), '  se =', qs.get('se'))
"
```

If `sp` lacks `w` and `c`, regenerate under the **corp** tenant (subscription owned by `boren@microsoft.com`, e.g. `PJ-COGNITIVESERVICES` = `a04bb0a4-25ba-444a-aea6-eb5c26d75f0e`):
```bash
az account set --subscription a04bb0a4-25ba-444a-aea6-eb5c26d75f0e
EXPIRY=$(date -u -d '+7 days' '+%Y-%m-%dT%H:%MZ')
export SAS=$(az storage container generate-sas --as-user --auth-mode login \
  --account-name tsstd01safn --name data --permissions rwcl --expiry "$EXPIRY" -o tsv)
echo "${SAS:0:60}..."
```
`--permissions rwcl` = read, write, create, list.

**Persist the refreshed SAS back to the cache** so future tools / sessions pick it up:
```bash
python3 -c "
import json, os, urllib.parse
path = '/home/boren/.sas/azure_blob_sas_cache.json'
v = json.load(open(path))
sas = os.environ['SAS']
expiry = urllib.parse.parse_qs(sas)['se'][0]
v['https://tsstd01safn.blob.core.windows.net/data'] = {'sas': sas, 'expiry_utc': expiry}
json.dump(v, open(path, 'w'), indent=2)
print('cache updated, sp=rcwl, se=', expiry)
"
```
Do this whenever you regenerate (or the cached `sp` lacks `w`+`c`, or the cached `se` is past/near expiry). Preserve the existing dict shape (`{"sas": ..., "expiry_utc": ...}`) and the other accounts' entries.

After generating, switch back to the green tenant if other tools need it:
```bash
az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0 >/dev/null 2>&1
```
(For local→corp `azcopy` itself, only `$SAS` on the destination matters; `AZCOPY_AUTO_LOGIN_TYPE` is not required because the source is local files.)

## Procedure

### Step 0 — Compute the remote path
Strip `~/data/` (or `/home/boren/data/`) from the local path; prepend `users/boren/data/`.

```
local : /home/boren/data/ckp/remax_qwen_bad_repeat_bracket_all_e1/
remote: https://tsstd01safn.blob.core.windows.net/data/users/boren/data/ckp/remax_qwen_bad_repeat_bracket_all_e1/
```

If the user explicitly gives a remote subpath, use it verbatim under `data/`.

### Step 1 — Verify local source
```bash
ls -lh <local_path>
du -sh <local_path>
```

### Step 2 — Ensure SAS has `w`+`c`
See Prerequisites. Export `$SAS` in the current shell.

### Step 3 — Upload

**Single file:**
```bash
azcopy copy "<local_file>" \
  "https://tsstd01safn.blob.core.windows.net/data/<remote_path>?${SAS}"
```

**Directory (recursive, expands contents into the destination prefix):**
```bash
azcopy copy "<local_dir>/*" \
  "https://tsstd01safn.blob.core.windows.net/data/<remote_dir>/?${SAS}" \
  --recursive
```

- Use trailing `/` on the destination directory URL, with `?${SAS}` placed **after** the trailing slash.
- For large uploads run in async terminal mode.
- No `--s2s-preserve-access-tier` flag needed for local→blob.

### Step 4 — Verify
```bash
azcopy list "https://tsstd01safn.blob.core.windows.net/data/<remote_dir>?${SAS}" | head -20
```
Compare file count and sizes against the local listing from Step 1.

## Notes & Pitfalls
- **Read-only SAS is the most common failure** — `sp=rl` will silently work for `azcopy list` but fail on `copy`. Always check `sp` before uploading.
- **SAS goes on the DESTINATION** (only). Place it after a trailing `/` for directory uploads.
- **`$SAS` is shell-scoped** — re-export in every new (especially async) terminal session.
- **Path mapping**: corp uses `data/<...>` (no `speech/` segment, unlike orange's `data/speech/<...>`). The default user root is `data/users/boren/data/`.
- Use `bbb ls az://tsstd01safn/data/<...>` if a corp-side `bbb` mount is configured; otherwise `azcopy list` with SAS.
- For an incremental mirror, use `azcopy sync` instead of `azcopy copy`:
  ```bash
  azcopy sync "<local_dir>" "https://tsstd01safn.blob.core.windows.net/data/<remote_dir>?${SAS}" --recursive
  ```
- If only certain extensions should be uploaded, add `--include-pattern '*.pt;*.json'` (and `--exclude-pattern` to drop).
- This skill is **local → corp only**. For corp → orange use `transfer-to-orange`; for orange → corp use `transfer-from-orange`.

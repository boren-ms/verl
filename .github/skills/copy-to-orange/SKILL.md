---
name: copy-to-orange
description: "Upload local files or directories directly to Orange (orng) with bbb cp/sync. Use for local-to-Orange data, configuration, and checkpoint uploads; use transfer-to-orange for Corp Blob-to-Orange transfers."
---

# Copy Local Files to Orange

Use `bbb cp` or `bbb sync` for local-to-Orange uploads, including text files,
datasets, and large checkpoints. Upload directly to the final Orange `az://`
path; no Corp upload, source SAS, or `orngtransfer` stage is needed.

Preserve the destination specified by the user or existing configuration.
For files under `/home/boren/data/`, the default mapping is:

```text
/home/boren/data/<subpath>
→ az://orngwus2cresco/data/speech/users/boren/data/<subpath>
```

## Copy files

Use `bbb cp` for a single file or a few selected files:

```bash
bbb cp /home/boren/data/example/config.yaml \
  az://orngwus2cresco/data/speech/users/boren/data/example/config.yaml
```

`bbb cp` overwrites an existing destination file. Inspect existing targets and
replace only files covered by the requested upload or update.

## Sync directories

Use `bbb sync` for a directory or an incremental update:

```bash
bbb sync /home/boren/data/example/train/ \
  az://orngwus2cresco/data/speech/users/boren/data/example/train/
```

Sync copies missing or changed files. Omit `--delete` unless removal of remote
files was explicitly requested. Use `--concurrency N` on either command when
needed to control transfer load. For JSONL-only preparation, copy selected
manifests or sync their dedicated directories rather than uploading unrelated
logs and local configuration.

## Verify and update configuration

Verify the remote file count and sizes with `bbb ls`; compare contents or
checksums where practical. Update dataset YAMLs to the final Orange `az://`
paths only after their target files are available. Keep local source copies
unless the user requests their removal.

If credentials or transfer requests fail, report the actual error. Do not
change the destination to Corp or create a Corp staging copy. For a Corp Blob
source rather than a local source, use
[transfer-to-orange](../transfer-to-orange/SKILL.md).

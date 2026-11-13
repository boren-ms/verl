---
name: digits-chunk-training-prepare
description: "Transfer N complete ASR ChunkFiles to Orange, merge their JSON metadata into h<N>.json, and create a matching training-data YAML. Use when: sample ChunkFiles, transfer N chunks, copy chunk audio/json/transcription to orng, create h100.json, merge chunk JSON files, or create chunk training YAML."
argument-hint: "<N> <source ChunkFiles URL> [train-config-name]"
---

# Chunk Sample Dataset

Create a bounded ASR training sample from a `ChunkFiles/` source. The result
contains exactly `N` complete chunks, each with `.audio`, `.json`, and
`.transcription`, a canonical parent-level `h<N>.json` manifest, and an
optional `recipe/phimm/config/data/train_data/<name>.yaml` configuration.

## Inputs and Output Contract

- `N`: Number of logical chunk stems, not number of blob files. For example,
  `N=100` transfers 300 blobs.
- `SOURCE_URL`: The source `ChunkFiles/` URL. Corp sources use a SAS token;
  GRN and Orange endpoints use `AZCOPY_AUTO_LOGIN_TYPE=AZCLI` with the Green
  tenant login.
- `PARENT_REL`: Source-relative directory above `ChunkFiles/`, for example
  `am_data/gpt_tts/multi_locale_tts/tier1/en-us/repeat/feature_extraction_sim`.
- `TRAIN_CONFIG_NAME`: Optional configuration stem, such as
  `enus_digits_chunk_100`.

At Orange, write the artifacts to:

```text
az://orngwus2cresco/data/speech/<PARENT_REL>/ChunkFiles/
az://orngwus2cresco/data/speech/<PARENT_REL>/h<N>.json
```

The merged `h<N>.json` is a single object with the shared `fileType` value and
a concatenated `fileInfo` array. Do not create an array of the individual
chunk JSON objects: the `chunk` dataset loader expects the canonical
`file_set.json` shape.

## Prerequisites

1. Read `/home/boren/.sas/azure_blob_sas_cache.json`; verify the source SAS
   `expiry_utc` is in the future. Do not print the SAS value.
2. Confirm the Azure CLI is logged into the Green tenant:

   ```bash
   az account show --query tenantId -o tsv
   # Expected: 8b9ebe14-d942-49e7-ace9-14496d0caff0
   ```

3. Use `AZCOPY_AUTO_LOGIN_TYPE=AZCLI` on every transfer to GRN or Orange and
   include `--s2s-preserve-access-tier=false` for service-to-service copies.

## Procedure

### 1. Build a complete-chunk manifest

Set the input variables. For a `/datablob1/...` source path, first strip that
prefix to obtain `PARENT_REL`. Set `SOURCE_CONTAINER_URL` to the actual source
container URL, for example `https://tsstd01wus2.blob.core.windows.net/data`.

```bash
set -e
N=100
PARENT_REL='am_data/gpt_tts/multi_locale_tts/tier1/en-us/repeat/feature_extraction_sim'
CHUNK_REL="${PARENT_REL}/ChunkFiles"
SOURCE_CONTAINER_URL='https://tsstd01wus2.blob.core.windows.net/data'
SAS=$(jq -r '."https://tsstd01wus2.blob.core.windows.net/data".sas' /home/boren/.sas/azure_blob_sas_cache.json)
MANIFEST=$(mktemp /tmp/chunk_sample.XXXXXX)

azcopy list "${SOURCE_CONTAINER_URL}/${CHUNK_REL}/?${SAS}" \
  | awk -F';' '/^chunk_.*\.(audio|json|transcription);/{print $1}' \
  | sed -E 's/\.(audio|json|transcription)$//' \
  | sort -u \
  | head -"$N" \
  | while IFS= read -r stem; do
      printf '%s\n%s\n%s\n' "${stem}.audio" "${stem}.json" "${stem}.transcription"
    done > "$MANIFEST"

test "$(wc -l < "$MANIFEST")" -eq "$((N * 3))"
test "$(sed -E 's/\.(audio|json|transcription)$//' "$MANIFEST" | sort -u | wc -l)" -eq "$N"
```

This deliberately excludes `.feature` and `.info`. First transfer one named
audio blob through both hops and list it at Orange before starting the full
copy.

### 2. Transfer through GRN to Orange

Stage the manifest to GRN:

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "${SOURCE_CONTAINER_URL}/${CHUNK_REL}?${SAS}" \
  "https://grngenaiexternal.blob.core.windows.net/inbound/speech/${CHUNK_REL}/" \
  --list-of-files="$MANIFEST" --s2s-preserve-access-tier=false --recursive
```

Verify the GRN layout before the second hop. With `--list-of-files`, AzCopy may
place the data under an extra `ChunkFiles/` directory. Identify the prefix that
contains all `$((N * 3))` requested artifacts; use that exact prefix as
`GRN_CHUNK_URL`:

```bash
GRN_BASE="https://grngenaiexternal.blob.core.windows.net/inbound/speech/${CHUNK_REL}"
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy list "${GRN_BASE}/" | head
# Use either ${GRN_BASE}/ or ${GRN_BASE}/ChunkFiles/ based on that listing.
GRN_CHUNK_URL="${GRN_BASE}/ChunkFiles/"
```

Copy to Orange using the discovered source prefix. Keep the manifest so only
the requested artifact types are copied:

```bash
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy \
  "${GRN_CHUNK_URL}*" \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/${CHUNK_REL}/" \
  --list-of-files="$MANIFEST" --s2s-preserve-access-tier=false --recursive
```

Validate the destination against the manifest. The expected result is `N` of
each extension, no manifest differences, and three artifacts for every stem.

```bash
DEST_LIST=$(mktemp /tmp/chunk_sample_orange.XXXXXX)
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy list \
  "https://orngwus2cresco.blob.core.windows.net/data/speech/${CHUNK_REL}/" \
  | sed -n 's/; Content Length:.*//p' \
  | awk '/^chunk_.*\.(audio|json|transcription)$/' \
  | sort -u > "$DEST_LIST"

test "$(comm -3 <(sort "$MANIFEST") "$DEST_LIST" | wc -l)" -eq 0
test "$(sed -E 's/\.(audio|json|transcription)$//' "$DEST_LIST" | sort | uniq -c | awk '$1 != 3 {bad++} END {print bad+0}')" -eq 0
```

### 3. Merge the N chunk JSON files

Download only JSON files from the final Orange directory, then build the
canonical file-set object. This fails if JSON files disagree on `fileType`.

```bash
WORK_DIR=$(mktemp -d /tmp/chunk_sample_json.XXXXXX)
MERGED_FILE=$(mktemp /tmp/h${N}.XXXXXX.json)
ORANGE_CHUNK_URL="https://orngwus2cresco.blob.core.windows.net/data/speech/${CHUNK_REL}"

AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy "${ORANGE_CHUNK_URL}/*" "$WORK_DIR/" \
  --include-pattern='*.json'

jq -s '{
  fileType: (map(.fileType) | unique | if length == 1 then .[0] else error("inconsistent fileType") end),
  fileInfo: (map(.fileInfo[]) | sort_by(.name))
}' "$WORK_DIR"/*.json > "$MERGED_FILE"

jq -e --argjson n "$N" '(.fileInfo | length == $n) and ([.fileInfo[].name] | unique | length == $n)' "$MERGED_FILE"
```

Upload it at the parent directory and round-trip validate it:

```bash
ORANGE_PARENT_URL="https://orngwus2cresco.blob.core.windows.net/data/speech/${PARENT_REL}"
AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy copy "$MERGED_FILE" "${ORANGE_PARENT_URL}/h${N}.json" --overwrite=true
```

### 4. Create the training-data configuration

Create `recipe/phimm/config/data/train_data/<TRAIN_CONFIG_NAME>.yaml`. For one
sample parent, use this template; for multiple parent directories, add one
`h<N>.json` URL per source under `specs`.

```yaml
dataset_name: chunk
specs:
  - az://orngwus2cresco/data/speech/<PARENT_REL>/h<N>.json
chunk_types:
  - audio
  - transcription
num_proc: auto
add_task_info:
  task: lang_asr
post_process:
  add_field:
    fields:
      data_source: asr
  verl_format:
    prompt_key: prompt
```

Parse the YAML after writing it, and verify each spec URL exists in Orange.

## Completion Checklist

- Both Corp-to-GRN and GRN-to-Orange AzCopy jobs completed with zero failures.
- Orange has exactly `N` audio, `N` JSON, and `N` transcription blobs.
- `h<N>.json` has exactly `N` unique `fileInfo[].name` values.
- The training config parses and contains every generated parent-level
  `h<N>.json` URI.

## Failure Handling

- Source `401 InvalidAuthenticationInfo` normally means `${SAS}` was not set
  in the current shell. Reload it from the cache; do not fall back to Green
  OAuth for a Corp source.
- If GRN or Orange receives only the one-file test or zero files, list the
  exact source prefix. Do not assume the manifest path; check whether AzCopy
  nested a second `ChunkFiles/` component in GRN.
- If an AzCopy source URL contains a glob, use a trailing `/*` only. Filter
  extensions with `--include-pattern='*.json'`; `*.json` within a URL path is
  invalid.
- If validation fails, leave the files available for inspection but do not
  report the sample or merged manifest as complete.
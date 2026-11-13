---
name: generate-audio-dataset
description: "Create a training-ready ASR dataset from JSONL or plain text using Azure Text to Speech, preserve existing JSONL fields, randomly assign available voices, upload audio and manifest to Orange, and create a recipe/phimm training-data YAML. Use when: generate TTS audio from text, build a synthetic speech JSONL, upload TTS data to az://orngwus2cresco/data/boren/data/tts, create a TTS dataset YAML, or prepare synthetic audio for verl evaluation or training."
argument-hint: "<input-jsonl-or-text> <task> [train|val]"
---

# TTS JSONL Dataset Preparation

Turn a local JSONL or one-text-per-line file into a complete synthetic speech
dataset. Preserve existing JSON object fields while guaranteeing `id`, `text`,
and `audio_path`, generate one WAV per row, publish the task directory to
Orange, and create a loadable PhImm dataset YAML.

## Inputs and Output Contract

- `INPUT`: Local JSONL or UTF-8 text file. A JSONL row must be an object. A
  plain-text row becomes its `text` value.
- `TASK`: Filesystem-safe task slug, for example `earnings_error_161`.
- `CONFIG_SPLIT`: `train_data` by default; use `val_data` when requested.
- `TEXT_KEY`: Input text field, default `text`. Pass another key such as `ref`
  to the manifest helper when needed.
- `VOICE_LOCALE`: Azure voice locale, default `en-US`.

Create this layout locally:

```text
~/data/tts/<TASK>/
├── <TASK>.jsonl
└── audios/
    ├── <id-1>.wav
    └── ...
```

Publish the same layout to:

```text
az://orngwus2cresco/data/boren/data/tts/<TASK>/
```

The output JSONL contains `id`, `audio_path`, and `text` on every row. Existing
JSONL fields such as `ref`, `hyp`, `entity_type`, and `error_pattern` remain
unchanged. `audio_path` is a relative `audios/<id>.wav` path so the YAML can
map it to Orange without rewriting the manifest.

## Authentication Boundaries

Text to Speech and Orange use different Azure tenants. Do not synthesize while
logged into Green, and do not upload to Orange while logged into Microsoft.

1. Before Azure Speech discovery or synthesis, switch to the Microsoft tenant:

   ```bash
   az login --tenant microsoft.com
   az account show --query '{tenantId:tenantId,subscription:name}' -o json
   ```

   Confirm the account shown is the intended Microsoft account. The TTS script
   uses `DefaultAzureCredential`, which consumes this Azure CLI identity.

2. After local synthesis and validation, switch to the Green tenant for Orange:

   ```bash
   az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0
   test "$(az account show --query tenantId -o tsv)" = \
     '8b9ebe14-d942-49e7-ace9-14496d0caff0'
   ```

Never print access tokens, SAS values, or storage credentials.

## Procedure

### 1. Normalize the source manifest

From the repository root, set paths and create a canonical JSONL. Use a new
task directory unless the user explicitly requests overwriting an existing
dataset.

```bash
TASK='<task>'
INPUT='<local-jsonl-or-text-file>'
LOCAL_ROOT="$HOME/data/tts/$TASK"
MANIFEST="$LOCAL_ROOT/$TASK.jsonl"
REMOTE_ROOT="az://orngwus2cresco/data/boren/data/tts/$TASK"
CONFIG_SPLIT='train_data'
CONFIG="recipe/phimm/config/data/${CONFIG_SPLIT}/${TASK}.yaml"

mkdir -p "$LOCAL_ROOT/audios"
python .github/skills/generate-audio-dataset/scripts/prepare_manifest.py \
  "$INPUT" "$MANIFEST" --audio-dir audios --text-key text
```

For source JSONL where synthesis text is under another field, pass the field:

```bash
python .github/skills/generate-audio-dataset/scripts/prepare_manifest.py \
  "$INPUT" "$MANIFEST" --audio-dir audios --text-key ref
```

The helper copies the selected source field to canonical `text`, preserves all
other fields, preserves valid unique IDs, creates IDs for missing ones, and
rewrites every `audio_path` for this new task.

Validate the manifest before any paid synthesis calls:

```bash
python scripts/generate_azure_tts_jsonl.py "$MANIFEST" \
  --output-dir "$LOCAL_ROOT/audios" --random-voice \
  --voice-locale en-US --dry-run > /tmp/${TASK}_tts_plan.txt
test "$(wc -l < /tmp/${TASK}_tts_plan.txt)" -eq "$(wc -l < "$MANIFEST")"
```

### 2. Generate all audio under the Microsoft tenant

Switch to `microsoft.com` as described above. The script processes all rows by
default, skips existing WAVs unless `--overwrite` is supplied, and randomly
selects a live voice for each row.

```bash
python scripts/generate_azure_tts_jsonl.py "$MANIFEST" \
  --endpoint https://boren-8685-resource.cognitiveservices.azure.com/ \
  --output-dir "$LOCAL_ROOT/audios" \
  --random-voice --voice-locale en-US
```

Use `--seed <integer>` for reproducible voice assignments. Use
`--voice-pool <voice-1> <voice-2> ...` to constrain the live service pool. Use
`--max-rows N` only for an intentional sample; omitting it means all rows.

### 3. Validate local completeness

Fail before upload if counts, JSON, paths, or WAV decoding do not match.

```bash
python - "$MANIFEST" "$LOCAL_ROOT/audios" <<'PY'
import json
import sys
import wave
from pathlib import Path

manifest = Path(sys.argv[1])
audio_dir = Path(sys.argv[2])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
assert rows, "manifest is empty"
assert len({row["id"] for row in rows}) == len(rows), "duplicate IDs"
assert len({row["audio_path"] for row in rows}) == len(rows), "duplicate audio paths"
for row in rows:
    assert row["text"].strip(), f"empty text: {row['id']}"
    wav_path = audio_dir / Path(row["audio_path"]).name
    assert wav_path.is_file() and wav_path.stat().st_size > 44, f"missing WAV: {wav_path}"
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getnframes() > 0, f"empty WAV: {wav_path}"
print(f"validated_rows={len(rows)}")
PY
```

### 4. Upload the task directory to Orange

Switch to the Green tenant first. Sync the directory as one unit so the JSONL
and its `audios/` children share the same remote task prefix.

```bash
bbb sync "$LOCAL_ROOT/" "$REMOTE_ROOT/"
bbb ls "$REMOTE_ROOT/"
bbb ls "$REMOTE_ROOT/audios/"
```

Compare remote JSONL and WAV counts with local counts. Do not declare the
upload complete when `bbb sync` reports failures or remote counts differ.

### 5. Create the PhImm dataset YAML

Create `recipe/phimm/config/data/<CONFIG_SPLIT>/<TASK>.yaml` with this shape:

```yaml
# Synthetic ASR dataset generated with Azure Text to Speech
dataset_name: jsonl
jsonl_paths: az://orngwus2cresco/data/boren/data/tts/<TASK>/<TASK>.jsonl
pre_process:
  path_map:
    field: audio_path
    src_part: "audios/"
    dst_part: "az://orngwus2cresco/data/boren/data/tts/<TASK>/audios/"
add_task_info:
  task: lang_asr
  prefix_prob: 0.0
post_process:
  add_field:
    fields:
      data_source: <TASK>
  verl_format:
    prompt_key: prompt
```

Use the actual task slug in all four locations. Preserve extra JSONL fields;
the loader retains them unless later processing explicitly removes them.

### 6. Validate training or evaluation readiness

Parse the YAML, confirm its remote paths exist, and smoke-test the actual
dataset factory from the configured Python environment:

```bash
python - "$CONFIG" <<'PY'
import sys
import yaml
from recipe.phimm.data.dataset import create_audio_dataset

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
config["cache_name"] = None
config["cache_path"] = None
dataset = create_audio_dataset(**config)
assert len(dataset) > 0
assert {"audio_path", "text"}.issubset(dataset.column_names)
print(f"rows={len(dataset)} columns={dataset.column_names}")
PY
```

For `train_data`, include the new YAML from the intended training data
composition and run that config's normal cache or loader smoke test. For
`val_data`, add it to the intended evaluation config and verify one bounded
evaluation load before launching a full workload.

## Completion Checklist

- The canonical manifest has the expected nonzero row count and valid JSON.
- Every row has a unique `id`, nonempty `text`, and unique relative
  `audios/*.wav` path; all original fields are preserved.
- Every manifest row has one readable, nonempty local WAV.
- Synthesis ran under the `microsoft.com` Azure account.
- Upload ran after switching to the Green tenant.
- Orange contains the manifest and exactly the expected WAV files under
  `az://orngwus2cresco/data/boren/data/tts/<TASK>/`.
- The generated YAML parses, resolves Orange audio paths, and loads through
  `create_audio_dataset` with a nonzero row count.
- The YAML is wired into the requested training or evaluation composition.

## Failure Handling

- If Speech authentication fails or no voices are returned, verify
  `az account show` is using the Microsoft account, then rerun. Do not switch
  to Green for synthesis.
- If Orange upload authentication fails, verify the Green tenant ID and rerun
  `bbb sync`; do not regenerate already valid audio.
- If synthesis stops partway, rerun without `--overwrite`; valid existing WAVs
  are skipped. Run local completeness validation before upload.
- If a source JSONL row lacks the selected text field, stop and report its line
  number. Do not silently synthesize from another field.
- If IDs or sanitized filenames collide, fix the source IDs or choose a new ID
  prefix; do not overwrite one row's audio with another's.
- If dataset loading fails, inspect the first remote JSONL row and mapped audio
  URI before changing training configuration.

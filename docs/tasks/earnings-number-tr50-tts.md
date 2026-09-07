# Earnings Number TR50 TTS Dataset

## Goal

Convert `tmp/earnings-number-tr50.jsonl` into a complete synthetic speech dataset, publish it to Orange, and add a loadable PhImm training-data YAML.

Task parameters:

- Task slug: `earnings-number-tr50`
- Config split: `train_data`
- Text key: `text`
- Voice locale: `en-US`
- Expected rows: 3,600

## Paths

- Source: `tmp/earnings-number-tr50.jsonl`
- Local root: `~/data/tts/earnings-number-tr50/`
- Manifest: `~/data/tts/earnings-number-tr50/earnings-number-tr50.jsonl`
- Audio directory: `~/data/tts/earnings-number-tr50/audios/`
- Current retry log: `/tmp/earnings-number-tr50_tts_retry.log`
- Orange root: `az://orngwus2cresco/data/boren/data/tts/earnings-number-tr50/`
- YAML to create: `recipe/phimm/config/data/train_data/earnings-number-tr50.yaml`

## Completion Status

Completed 2026-09-07:

- The canonical manifest contains 3,600 unique rows and preserves the source
  `spoken`, `keywords`, and `scenario` fields.
- All 3,600 local WAVs are readable, nonempty, and match unique manifest paths.
- Synthesis completed under the Microsoft tenant.
- The complete task directory was uploaded under the Green tenant.
- Orange contains one manifest and exactly 3,600 WAV files.
- `recipe/phimm/config/data/train_data/earnings-number-tr50.yaml` loads all
  3,600 rows through `create_audio_dataset`.
- `recipe/phimm/config/v2607_time/remax_earnings_number_tr50_s200_bs64_cut05_gt.yaml`
  includes the dataset in a loadable training composition.

## Resume Procedure

### 1. Check the active synthesis pass

Do not start another generator while the current one is running.

```bash
pgrep -f -- 'generate_azure_tts_jsonl.py.*earnings-number-tr50' >/dev/null
```

Inspect progress without printing credentials or full process arguments:

```bash
find "$HOME/data/tts/earnings-number-tr50/audios" -maxdepth 1 \
  -type f -name '*.wav' | wc -l
grep -c '^Generated line' /tmp/earnings-number-tr50_tts_retry.log
grep -c '^Failed line' /tmp/earnings-number-tr50_tts_retry.log
```

### 2. Validate and retry missing audio

After the generator exits, validate all 3,600 manifest paths using the completeness script from `.github/skills/generate-audio-dataset/SKILL.md`. Each WAV must exist, be larger than 44 bytes, open with `wave.open`, and contain frames.

Delete only WAVs that fail those checks. Before retrying, confirm the Microsoft tenant:

```bash
test "$(az account show --query tenantId -o tsv)" = \
  '72f988bf-86f1-41af-91ab-2d7cd011db47'
```

If the tenant does not match, run `az login --tenant microsoft.com` and verify it again. Resume without overwriting valid audio:

```bash
python scripts/generate_azure_tts_jsonl.py \
  "$HOME/data/tts/earnings-number-tr50/earnings-number-tr50.jsonl" \
  --endpoint https://boren-8685-resource.cognitiveservices.azure.com/ \
  --output-dir "$HOME/data/tts/earnings-number-tr50/audios" \
  --random-voice --voice-locale en-US
```

Repeat validation and retry until all 3,600 WAVs are readable and no paths are missing.

### 3. Upload to Orange

Speech and Orange use different tenants. Only after local completeness is proven, switch to Green:

```bash
az login --tenant 8b9ebe14-d942-49e7-ace9-14496d0caff0
test "$(az account show --query tenantId -o tsv)" = \
  '8b9ebe14-d942-49e7-ace9-14496d0caff0'
```

Upload the complete task directory:

```bash
bbb sync "$HOME/data/tts/earnings-number-tr50/" \
  "az://orngwus2cresco/data/boren/data/tts/earnings-number-tr50/"
bbb ls az://orngwus2cresco/data/boren/data/tts/earnings-number-tr50/
bbb ls az://orngwus2cresco/data/boren/data/tts/earnings-number-tr50/audios/
```

Confirm one remote JSONL and exactly 3,600 remote WAV files. Do not mark upload complete if `bbb sync` reports failures or counts differ.

### 4. Create the PhImm YAML

Create `recipe/phimm/config/data/train_data/earnings-number-tr50.yaml`:

```yaml
# Synthetic ASR dataset generated with Azure Text to Speech
dataset_name: jsonl
jsonl_paths: az://orngwus2cresco/data/boren/data/tts/earnings-number-tr50/earnings-number-tr50.jsonl
pre_process:
  path_map:
    field: audio_path
    src_part: "audios/"
    dst_part: "az://orngwus2cresco/data/boren/data/tts/earnings-number-tr50/audios/"
add_task_info:
  task: lang_asr
  prefix_prob: 0.0
post_process:
  add_field:
    fields:
      data_source: earnings-number-tr50
  verl_format:
    prompt_key: prompt
```

Parse the YAML and smoke-test it with `create_audio_dataset` as specified in the skill. Verify a nonzero dataset with `audio_path` and `text` columns.

### 5. Wire the training composition

Add the YAML to the intended training-data composition. No target composition was specified in the original request, so identify or confirm the intended training config before making this final wiring change. Run that composition's normal loader/cache smoke test afterward.

## Completion Criteria

- 3,600 unique manifest rows and 3,600 readable local WAVs.
- Synthesis completed under the Microsoft tenant.
- Upload completed under the Green tenant.
- Orange contains the manifest and exactly 3,600 WAVs.
- The dataset YAML parses and loads through `create_audio_dataset`.
- The YAML is included in the intended training composition.

Never print access tokens, SAS values, storage credentials, or full credential-bearing process arguments while resuming this task.

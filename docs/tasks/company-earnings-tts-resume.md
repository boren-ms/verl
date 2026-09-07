# Company Earnings TTS Dataset Resume

Last verified: 2026-09-06

## Goal

Generate complete synthetic speech datasets, one country at a time, from:

| Market | Source | Task | Expected rows |
| --- | --- | --- | ---: |
| US | `tmp/earnings-company-us.jsonl` | `earnings-company-us` | 5,894 |
| UK | `tmp/earnings-company-uk.jsonl` | `earnings-company-uk` | 3,660 |
| HK | `tmp/earnings-company-hk.jsonl` | `earnings-company-hk` | 2,778 |

Follow `.github/skills/generate-audio-dataset/SKILL.md`. Preserve all source fields, synthesize the canonical `text` field with random `en-US` voices, upload each completed task to Orange, and create a `train_data` YAML.

## Current Status

All three datasets are complete.

| Market | Local valid WAVs | Orange WAVs | Train YAML |
| --- | ---: | ---: | --- |
| US | 5,894 | 5,894 | `recipe/phimm/config/data/train_data/earnings-company-us.yaml` |
| UK | 3,660 | 3,660 | `recipe/phimm/config/data/train_data/earnings-company-uk.yaml` |
| HK | 2,778 | 2,778 | `recipe/phimm/config/data/train_data/earnings-company-hk.yaml` |

- Every manifest-referenced local WAV is readable, larger than 44 bytes, and
  contains at least one frame.
- Each Orange task root contains one manifest and exactly the expected number
  of WAV files.
- All three YAMLs load through `create_audio_dataset` with their exact expected
  row counts and the formatted columns `audio_path`, `prompt`, `data_source`,
  `reward_model`, and `extra_info`.
- No aggregate training composition was specified, so the standalone
  `train_data` YAMLs were not added to a mixed composition.

## Authentication Boundary

Azure Speech and Orange require different tenants.

- Speech synthesis must run under the Microsoft tenant: `72f988bf-86f1-41af-91ab-2d7cd011db47`.
- Orange upload must run under the Green tenant: `8b9ebe14-d942-49e7-ace9-14496d0caff0`.
- Do not synthesize while logged into Green.
- Do not upload to Orange while logged into Microsoft.
- Never print access tokens, SAS values, or credentials.

Synthesis completed under the Microsoft tenant. Upload completed under the
Green tenant using the `AIPLATFORM-ORANGE-USERS` subscription.

## Resume Procedure

### 1. Finish US synthesis

1. Check whether `generate_azure_tts_jsonl.py` for `earnings-company-us` is still running. Report only PIDs, not full process arguments.
2. If it is running, let the current pass finish.
3. Validate every manifest-referenced WAV with `wave.open`, requiring file size greater than 44 bytes and at least one frame.
4. Delete only confirmed invalid WAVs. Never delete valid completed audio.
5. Verify the active tenant is Microsoft.
6. Rerun synthesis without `--overwrite`; the script skips valid existing WAVs:

```bash
python scripts/generate_azure_tts_jsonl.py \
  "$HOME/data/tts/earnings-company-us/earnings-company-us.jsonl" \
  --endpoint https://boren-8685-resource.cognitiveservices.azure.com/ \
  --output-dir "$HOME/data/tts/earnings-company-us/audios" \
  --random-voice --voice-locale en-US \
  >> /tmp/earnings-company-us_tts.log 2>&1
```

1. Repeat cleanup and resume passes until all 5,894 manifest rows have one readable, nonempty WAV and there are no missing or invalid files.

Azure timeouts can leave zero-byte WAVs. The generator skips any existing path regardless of validity, so invalid files must be removed before a retry.

### 2. Prepare and synthesize UK

Use task `earnings-company-uk` and source `tmp/earnings-company-uk.jsonl`.

1. Create `$HOME/data/tts/earnings-company-uk/audios/` only if the task directory remains absent.
2. Run `prepare_manifest.py` with `--audio-dir audios --text-key text`.
3. Run the TTS dry-run and require 3,660 plan lines.
4. Verify source-field preservation, unique IDs, and unique relative audio paths.
5. Under the Microsoft tenant, synthesize all rows with random `en-US` voices.
6. Validate and retry failures using the same process as US.

### 3. Prepare and synthesize HK

Repeat the UK procedure with task `earnings-company-hk`, source `tmp/earnings-company-hk.jsonl`, and expected count 2,778.

### 4. Upload completed datasets

After all local datasets pass validation, switch to the Green tenant and verify its tenant ID before upload.

Upload each whole task directory to:

```text
az://orngwus2cresco/data/boren/data/tts/earnings-company-us/
az://orngwus2cresco/data/boren/data/tts/earnings-company-uk/
az://orngwus2cresco/data/boren/data/tts/earnings-company-hk/
```

Use `bbb sync`, then compare remote manifest and WAV counts with local counts. Do not declare upload complete if `bbb sync` reports failures or counts differ.

### 5. Create and validate train YAMLs

Create:

```text
recipe/phimm/config/data/train_data/earnings-company-us.yaml
recipe/phimm/config/data/train_data/earnings-company-uk.yaml
recipe/phimm/config/data/train_data/earnings-company-hk.yaml
```

Each YAML must follow the skill template and use its task slug consistently in `jsonl_paths`, `dst_part`, and `data_source`. Parse each YAML and smoke-test it through `recipe.phimm.data.dataset.create_audio_dataset`, requiring a nonzero row count and `audio_path` plus `text` columns.

No training composition has yet been selected. Wire these YAMLs into the intended composition only after that target is known.

## Completion Criteria

- US has 5,894 valid local and remote WAVs.
- UK has 3,660 valid local and remote WAVs.
- HK has 2,778 valid local and remote WAVs.
- Each canonical manifest preserves all original fields and has unique `id` and `audio_path` values.
- No manifest-referenced WAV is missing, empty, corrupt, or frame-less.
- Synthesis is performed under Microsoft and upload under Green.
- All three Orange task roots contain the expected manifest and audio files.
- All three train YAMLs parse and load successfully through the actual dataset factory.

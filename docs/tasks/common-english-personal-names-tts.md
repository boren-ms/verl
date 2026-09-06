# Common English Personal Names TTS Dataset

Last updated: 2026-09-06

## Objective

Process every source manifest under `tmp/common-english-personal-names/utterances/*.jsonl` one by one using the `generate-audio-dataset` skill. For each country dataset:

1. Prepare a canonical JSONL manifest.
2. Generate one readable WAV per row with Azure Text to Speech.
3. Validate local completeness.
4. Upload the manifest and audio to Orange.
5. Create a PhImm training-data YAML.
6. Parse and smoke-test the dataset through `create_audio_dataset`.

Follow `.github/skills/generate-audio-dataset/SKILL.md` as the controlling procedure.

## Dataset Scope

There are 18 country-specific source JSONLs with 160,409 rows in total:

`bi`, `bw`, `ca`, `cm`, `fj`, `gb`, `gh`, `ie`, `in`, `jm`, `mt`, `na`, `ng`, `ph`, `sd`, `sg`, `us`, `za`

Task names use this pattern:

```text
common-english-personal-names-<country-code>
```

Local dataset layout:

```text
~/data/tts/common-english-personal-names-<country-code>/
├── common-english-personal-names-<country-code>.jsonl
└── audios/
```

Orange destination:

```text
az://orngwus2cresco/data/boren/data/tts/common-english-personal-names-<country-code>/
```

## Completed

- Audited all 18 source JSONLs.
- Confirmed 160,409 total rows.
- Prepared all 18 canonical manifests and local task directories.
- Confirmed source-field preservation, unique IDs, unique audio paths, and matching dry-run counts.
- Restored the Microsoft Azure identity required for synthesis.
- Started synthesis for the first dataset, `common-english-personal-names-bi`.

## Current Status

Synthesis is active in terminal ID `a20ef9bd-067a-407c-8077-25e63706a3f4`.

At the latest check, `bi` had reached row 1,859 of 4,419, approximately 42.1%. The process was still generating audio normally and was not waiting for input.

Known timeout rows observed so far in `bi`:

```text
215, 578, 652, 864, 1197, 1270, 1318, 1468, 1707, 1826
```

Some failed synthesis calls may leave invalid or partial WAV files. Do not assume every existing WAV is valid.

## Authentication Boundaries

Azure Speech synthesis and Orange upload use different tenants.

For synthesis, use the Microsoft tenant:

```text
tenant: 72f988bf-86f1-41af-91ab-2d7cd011db47
subscription: Speech Modeling Development Storage
user: boren@microsoft.com
endpoint: https://boren-8685-resource.cognitiveservices.azure.com/
```

For Orange upload, switch to the Green tenant:

```text
tenant: 8b9ebe14-d942-49e7-ace9-14496d0caff0
```

Never synthesize while logged into Green. Never upload to Orange while logged into Microsoft.

## Active Synthesis Command

The active `bi` pass was launched from the repository root with:

```bash
TASK=common-english-personal-names-bi
python scripts/generate_azure_tts_jsonl.py \
  "/home/boren/data/tts/$TASK/$TASK.jsonl" \
  --endpoint https://boren-8685-resource.cognitiveservices.azure.com/ \
  --output-dir "/home/boren/data/tts/$TASK/audios" \
  --random-voice --voice-locale en-US
```

## Resume Procedure

1. Check terminal `a20ef9bd-067a-407c-8077-25e63706a3f4`. If it is still producing rows, let the current `bi` pass finish.
2. Validate every manifest-referenced WAV by checking file size, opening it with Python `wave`, and confirming a positive frame count.
3. Delete only missing-output placeholders or verified invalid WAVs.
4. Rerun the same synthesis command without `--overwrite`. Existing valid WAVs will be skipped and missing or deleted rows will be retried.
5. Repeat validation and retry until all 4,419 `bi` rows have readable WAVs.
6. Process the remaining datasets sequentially in this order:

```text
bw, ca, cm, fj, gb, gh, ie, in, jm, mt, na, ng, ph, sd, sg, us, za
```

1. After all local synthesis and validation are complete, switch to the Green tenant and upload each complete task directory with `bbb sync`.
2. Verify the remote manifest and WAV counts exactly match local counts.
3. Create one YAML per task under `recipe/phimm/config/data/train_data/` using the skill template.
4. Wire the YAMLs into the intended training composition, parse each YAML, and smoke-test `create_audio_dataset`.

## Recovery Rules

- Never use `--overwrite` for recovery; preserve valid generated audio.
- A timeout is a row-level failure and does not mean the whole pass stopped.
- Remove a WAV only after verifying it is unreadable, empty, or otherwise invalid.
- Continue retry passes until manifest count, WAV count, and readable-WAV count all match.
- Do not upload a partially complete dataset.

## Remaining Work

- [ ] Finish and recover all `bi` synthesis rows.
- [ ] Synthesize and validate the other 17 country datasets sequentially.
- [ ] Switch to Green and upload all 18 complete datasets to Orange.
- [ ] Verify exact remote manifest and WAV counts.
- [ ] Create 18 PhImm training-data YAML files.
- [ ] Wire the YAMLs into the intended training composition.
- [ ] Parse YAMLs and run `create_audio_dataset` smoke tests.

# Inhouse en-US-entity-v3 longform evaluation

This evaluates the full English entity benchmark stored under
`Evaluation/InhouseASR/EWER/en-US-entity-v3`: 31 subsets, 755 original
recordings and 25,978 existing VAD audio clips. The `EWER` directory is the
dataset location; the primary metric here is official SpeechInsight **EER**.
No training or new audio segmentation is performed.

## Data preparation

```bash
python -m recipe.phimm.data.inhouse_entity \
  --cache-dir tmp/inhouse_entity_preparation --publish
```

The adapter reads each subset's original `test.jsonl` and existing
`vad_segments.jsonl`. It checks recording coverage, contiguous clip indices,
chronological order and full-reference identity before publishing one JSONL
row per original recording. Each row lists the ordered, unchanged VAD audio
paths. Source clips remain at their original Blob locations; only the
normalized manifests are published. Existing manifests must match exactly.

The normalized manifest root is:

```text
az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_EWER_en_US_entity_v3_presegment_v1/
```

Full entity annotations come from original `Transcription`. Original
`DisplayTranscription` is retained separately for DTER, including the fallback
to `Transcription` when display text is empty. The VAD manifest normally stores
the display reference, without timestamps, in its first clip's `Transcription`
field and leaves later clips empty. Those empty fields must not become empty
reference transcripts for individually scored clips.

Audio inventory verification on 2026-11-20 found all 25,978 referenced clip
paths. Header checks on the first and longest clip of every subset (62 files)
found mono 24 kHz audio, with sample counts matching filename bounds and a
maximum clip duration of 30 seconds. The source filenames express native
sample counts. The existing ASR loader resamples audio to 16 kHz.

## Decode and score

After syncing the checkout and registering its editable packages on a
verified, available two-node Ray cluster:

```bash
python -m recipe.phimm.main_asr_eval \
  --config-name eval_2611v1_inhouse_entity_longform --cfg job --resolve

python plugins/phimm/submit.py --evaluate \
  --submission-id inhouse-eer-2611v1-s38000-longform-20261120 \
  --config-name eval_2611v1_inhouse_entity_longform
```

The preset selects the 2611 V1 SFT step-38000 merged Hugging Face model,
`lora_rank=0`, `val_only=true` and `resume_mode=disable`. It uses the existing
verbatim English language-hint prompt and greedy validation decoding.
Segments are decoded independently and concatenated in recording order;
`asr.context_aware.enable=false`. Neither reference transcripts nor entity
lists enter the decoding prompt.

Each recording is scored once. Official SpeechInsight `GetMetrics --verbatim`
provides `NumTransEnts` and `NumTransEntsMatched` against its original tagged
`Transcription`:

```text
EER = sum(NumTransEnts - NumTransEntsMatched) / sum(NumTransEnts)
```

DTER uses the existing dfmetrics backend and the original display reference,
with timestamp markers removed. Both metrics preserve numerators and
denominators. `strict_eer=true` rejects backend failures and invalid entity
counts. These are not treated as perfect recognition.

## Accept and report results

Require the exact Ray submission to reach `SUCCEEDED`, with all generation
files and final metrics durably published. Then run:

```bash
python -m recipe.phimm.tools.inhouse_entity_report \
  --result-root az://orngwus2cresco/data/boren/outputs/v2611_eval/inhouse-eer-2611v1-s38000-longform-20261120/val_data_gen \
  --output-dir tmp/inhouse_entity_results
```

The report audits each original recording ID, both reference forms, ordered
audio paths, number of generated segment outputs and empty previous contexts.
It checks count-derived rates against saved trainer metrics and requires all
31 subsets / 755 recordings / 25,978 clips. Outputs include `summary.json`,
`datasets.csv` and `report.md`, with exact generation and input paths. Corpus
rates weight reference entities/tokens; macro rates give each subset equal
weight.

Focused CPU checks:

```bash
python -m pytest -q \
  tests/recipe/phimm/data/test_inhouse_entity_presegment.py \
  tests/recipe/phimm/reward/test_inhouse_entity_strict.py \
  tests/recipe/phimm/data/test_inhouse_entity_report.py
```

## Verified 2611 V1 result (2026-11-20)

Submission `inhouse-eer-2611v1-s38000-longform-20261120` reached Ray
`SUCCEEDED` using launch snapshot `8039c13fb9325fe2f21d02f6d790e7d890d8ae6e`.
The saved-detail audit verified all 31 subsets, 755 original recordings and
25,978 ordered pre-segmented clips, unchanged references, empty preceding
contexts, and agreement with every saved subset metric.

| Aggregation | EER | DTER |
|---|---:|---:|
| Corpus, summed errors / summed reference units | 9.6647% | 16.0050% |
| Equal weight per subset | 10.5701% | 15.5241% |

Corpus EER is `7365 / 76205`; corpus DTER is `276957 / 1730440`.
Use the official scorer's 76,205 reference entities as the EER denominator;
the 76,243 raw opening entity tags are an inventory count, not the metric
denominator. All 38 counts are explained: the official scorer excludes 36
entities inside `<OL>` overlap regions and merges two pairs of adjacent
entity spans during normalization (`E. M. P.` + `Amy` becomes `emp amy`;
`Coke` + `Pepsi` becomes `coke-pepsi`). No recordings or references are missing.

The durable run root is:

```text
az://orngwus2cresco/data/boren/outputs/v2611_eval/inhouse-eer-2611v1-s38000-longform-20261120
```

Final metrics are in `val_data_gen/0.metrics.json`; each subset's complete
generations are in `val_data_gen/<source>/0.jsonl`. The local report package
is `tmp/inhouse_eer_2611v1/results/`, containing `report.md`, `datasets.csv`
and `summary.json`.

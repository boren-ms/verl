# Inhouse English entity dataset — existing VAD pre-segment version

This is the historical 31-domain entity-v3 inventory. The active MoE EN-US suites
are [EER](eer.md) and [TER](ter.md),
with five datasets each. Evaluation presets select those explicit workbook lists.

Dataset information for [`english_entity_presegment.yaml`](english_entity_presegment.yaml).

Inventory date: **2026-11-20**. These statistics describe the exact normalized manifests used for the 2611 V1 evaluation. Their SHA-256 hashes were checked against the preparation inventory.

## Dataset and source paths

- **Dataset:** `en-US-entity-v3`, stored under `EWER`.
- **Language / modality:** English (`en-US`), speech audio with verbatim and display transcripts and entity annotations.
- **Coverage:** 28 Conversation subsets and 3 CustomerSpeech subsets.
- **Sample:** one original recording, identified by its parent UUID.
- **Segment:** one existing VAD audio clip belonging to that recording.
- **Input version:** original `vad_segments.jsonl` and its `vad_segments/` audio files. The distinct transcript-based `segments.jsonl` version is not used.

Original source root:

```text
az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3
```

Normalized manifest root consumed by the adjacent YAML:

```text
az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_EWER_en_US_entity_v3_presegment_v1/*.jsonl
```

The adapter reads each subset's `test.jsonl` and `vad_segments.jsonl` and emits one normalized row per original recording. The row contains an ordered list of existing audio paths; audio is not resegmented or rewritten.

## Corpus totals

| Quantity | Value |
|---|---:|
| Subsets | 31 |
| Original recordings / samples | 755 |
| Existing VAD segments | 25,978 |
| Verbatim reference words | 1,768,567 |
| Display reference words | 1,737,940 |
| Sum of segment audio (hours) | 175.75 |

Audio hours sum the durations of the VAD clips. They include clip overlap and exclude omitted gaps, so they are not the duration of the continuous original recordings.

## Per-subset counts and averages

Rows are Conversation subsets unless prefixed with **Customer /**. For the exact datasource key, prepend `eer_Conversation_` to a Conversation name, or `eer_CustomerSpeech_` to a Customer name. Ranges are minimum–maximum.

**Words/segment*** is the corpus ratio **total recording-reference words / total VAD segments**. It is not a measured distribution of individual clip transcripts: the VAD manifest stores the full recording reference on the first clip and leaves later references empty.

| Subset | Samples | Segments | Verbatim words | Clip hours | Segments/sample, mean [range] | Mean segment (s) | Mean words/sample | Words/segment* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CapitalMarket | 20 | 941 | 70,131 | 6.33 | 47.0 [13–77] | 24.22 | 3,506.6 | 74.5 |
| ConsumerGood | 16 | 711 | 55,179 | 4.97 | 44.4 [19–78] | 25.15 | 3,448.7 | 77.6 |
| DefenseIntelligence | 15 | 984 | 65,076 | 6.56 | 65.6 [14–77] | 23.99 | 4,338.4 | 66.1 |
| DeliciousHealthy | 35 | 725 | 54,460 | 4.98 | 20.7 [8–33] | 24.75 | 1,556.0 | 75.1 |
| DoctorPatientConsultation | 39 | 689 | 42,132 | 4.70 | 17.7 [6–47] | 24.58 | 1,080.3 | 61.1 |
| Energy | 15 | 855 | 55,810 | 5.68 | 57.0 [11–77] | 23.92 | 3,720.7 | 65.3 |
| EnhancedMedical | 48 | 1,103 | 75,278 | 7.25 | 23.0 [10–32] | 23.66 | 1,568.3 | 68.2 |
| Gaming | 32 | 733 | 56,980 | 4.97 | 22.9 [9–33] | 24.40 | 1,780.6 | 77.7 |
| Insurance | 35 | 733 | 54,337 | 5.09 | 20.9 [10–31] | 24.98 | 1,552.5 | 74.1 |
| K12HigherEdu | 14 | 864 | 60,593 | 5.81 | 61.7 [25–83] | 24.20 | 4,328.1 | 70.1 |
| LawEnforcement | 13 | 699 | 49,379 | 4.66 | 53.8 [11–88] | 24.00 | 3,798.4 | 70.6 |
| Legal | 15 | 1,046 | 67,802 | 6.96 | 69.7 [49–83] | 23.96 | 4,520.1 | 64.8 |
| LibraryMuseum | 15 | 799 | 47,731 | 5.27 | 53.3 [16–80] | 23.76 | 3,182.1 | 59.7 |
| LifeHealth | 16 | 796 | 57,079 | 5.35 | 49.8 [16–77] | 24.20 | 3,567.4 | 71.7 |
| Manufactory | 18 | 928 | 65,435 | 6.16 | 51.6 [30–73] | 23.90 | 3,635.3 | 70.5 |
| Media | 16 | 910 | 61,962 | 6.07 | 56.9 [11–80] | 24.02 | 3,872.6 | 68.1 |
| MedicalTeamMeetings | 12 | 732 | 41,465 | 5.09 | 61.0 [44–77] | 25.01 | 3,455.4 | 56.6 |
| PatientHistoryDictation | 12 | 820 | 45,204 | 5.93 | 68.3 [54–86] | 26.04 | 3,767.0 | 55.1 |
| PayorProvider | 14 | 763 | 57,821 | 5.26 | 54.5 [24–76] | 24.81 | 4,130.1 | 75.8 |
| Pharmaceutical | 17 | 801 | 50,578 | 5.26 | 47.1 [21–78] | 23.63 | 2,975.2 | 63.1 |
| PharmacyInstruction | 30 | 711 | 44,476 | 5.01 | 23.7 [11–55] | 25.38 | 1,482.5 | 62.6 |
| PublicSafetyJustice | 16 | 741 | 51,764 | 4.94 | 46.3 [17–77] | 24.02 | 3,235.2 | 69.9 |
| Retail | 15 | 766 | 54,680 | 5.14 | 51.1 [15–79] | 24.13 | 3,645.3 | 71.4 |
| ScienceTech | 36 | 754 | 52,005 | 5.09 | 20.9 [7–31] | 24.33 | 1,444.6 | 69.0 |
| SurgeryInstruction | 34 | 663 | 37,124 | 4.87 | 19.5 [15–30] | 26.46 | 1,091.9 | 56.0 |
| Sustain | 15 | 862 | 55,652 | 5.70 | 57.5 [15–80] | 23.79 | 3,710.1 | 64.6 |
| Telecommunication | 16 | 890 | 58,137 | 5.94 | 55.6 [14–78] | 24.05 | 3,633.6 | 65.3 |
| Vehicles | 34 | 734 | 49,295 | 4.87 | 21.6 [9–31] | 23.89 | 1,449.9 | 67.2 |
| Customer / Banking | 36 | 815 | 63,623 | 5.54 | 22.6 [11–32] | 24.49 | 1,767.3 | 78.1 |
| Customer / LawEnforcement | 37 | 866 | 60,432 | 5.86 | 23.4 [9–32] | 24.35 | 1,633.3 | 69.8 |
| Customer / Medical | 69 | 1,544 | 106,947 | 10.44 | 22.4 [8–30] | 24.34 | 1,550.0 | 69.3 |
| TOTAL | 755 | 25,978 | 1,768,567 | 175.75 | 34.4 [6–88] | 24.36 | 2,342.5 | 68.1 |

## Corpus distributions

| Quantity | Minimum | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Segments / recording | 6.00 | 34.41 | 25.00 | 75.00 | 88.00 |
| Audio seconds / segment | 0.06 | 24.36 | 27.15 | 30.00 | 30.00 |
| Verbatim words / recording | 337.00 | 2,342.47 | 1,783.00 | 5,232.50 | 6,797.00 |
| Sum of clip minutes / recording | 2.09 | 13.97 | 10.09 | 29.92 | 37.22 |

## Counting definitions and audio validation

- **Words:** remove XML-like annotation tags and timestamp markers, retain the text inside tags (including disfluency and overlap text), split on whitespace, and count tokens containing at least one alphanumeric character. Tags are removed without inserting whitespace so tags inside a word do not split it. Contractions and hyphenated forms remain one token; space-separated initials count separately.
- **Primary reference:** original `Transcription`, retained as `entity_transcription` in normalized manifests. Display-word totals use original `DisplayTranscription`, or the existing fallback to `Transcription` when display text is empty.
- **Inventory versus scoring:** these word counts do not apply the official scorer's normalization or exclusions. They are not the official DTER token denominator or the EER entity denominator.
- **Segment duration:** `(end_sample - start_sample) / 24000`, using native sample bounds from each filename and normalized manifest. Header checks on the first and longest clip of each subset (62 files) found mono 24 kHz audio and sample counts matching the filename bounds. All 25,978 referenced clip paths were verified present. Every clip's header was not individually inspected.
- **Loader:** the existing ASR audio loader resamples input audio to 16 kHz. Filename bounds must still be interpreted at the native 24 kHz rate.
- **Distributions:** linear-interpolated percentiles at `(N - 1) × quantile`. Sample-level statistics weight each original recording equally; segment-length statistics weight each VAD clip equally. The TOTAL row is computed from all recordings/clips, not by averaging subset means.

## Normalized row fields and evaluation behavior

| Field | Meaning |
|---|---|
| `id` | Original recording UUID. |
| `audio_path` | Ordered list of unchanged existing VAD audio paths. |
| `parent_audio_path` | Original recording path, used to retain parent identity. |
| `text` | Original display reference, with the existing fallback to the verbatim reference. |
| `entity_transcription` | Original tagged verbatim reference for entity scoring. |
| `n_segments` | Number of VAD clips in the recording. |
| `segment_starts_samples`, `segment_ends_samples` | Ordered clip bounds in native 24 kHz sample units. |
| `reference_scope` | `parent`: the reference belongs to the entire recording. |
| `source_manifest`, `reference_manifest` | Original VAD and full-recording manifest locations. |
| `data_source`, `language` | Subset-specific datasource key and English language label. |

The data YAML selects verbatim-style English prompts with a language hint. The 2611 V1 longform evaluation preset decodes clips independently, concatenates their outputs in recording order, and scores each original recording once. References and entity lists are not supplied in the decoding prompt.

## Preparation and related implementation

```bash
python -m recipe.phimm.data.inhouse_entity \
  --cache-dir tmp/inhouse_entity_preparation --publish
```

The adapter validates original/clip coverage, ordered contiguous clip indices, native bounds, and reference identity. Existing published manifests must match the generated bytes; changed manifests are not silently overwritten.

- [Data adapter](../../../../data/inhouse_entity.py)
- [2611 V1 longform evaluation preset](../../../v2611_dev/eval/eval_enus_eer.yaml)
- [Evaluation, scoring, and verified result documentation](../../../../../../../../../docs/experiments/inhouse_entity_longform.md)

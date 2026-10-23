# Entity error training set

Use [expanded_entity_train_v2.yaml](expanded_entity_train_v2.yaml), the highest-coverage combined set. This directory contains one training YAML; JSONL and audio are stored in Orange blob storage.

## Contents and coverage

The set contains **743 samples**: 602 LibriSpeech, 58 MLS, 47 Common Voice, and 36 Libriheavy.

| Metric | Coverage |
|---|---:|
| Canonical entity word forms | 54/60 (90.0%) |
| Error patterns | 58/65 (89.2%) |
| Weighted by observed error occurrences | 172/191 (90.1%) |

These metrics describe correct word forms present in training data, not demonstrated model error correction or guaranteed same-entity identity.

Remaining uncovered word forms: `fristoe`, `gobey`, `grandpap buzzard`, `hen lord`, `mainhall`, `pigeoncote`.

## Remote data

JSONL:

```text
az://orngwus2cresco/data/boren/data/LibriSpeech/entity_error_subset_20261023/remote_train_v1/expanded_entity_train_v2.jsonl
```

The manifest uses remote audio references throughout. The 83 previously local CV/Libriheavy WAV files were uploaded under the same prefix in `audio/`; existing LibriSpeech files and MLS audio chunks are reused at their original Orange paths. Local source copies remain as backups.

## Usage and validation

The data YAML overrides every loaded training row to `data_source: ls_kw` before verl formatting, providing a single sampling, reward-routing, and training-metrics group. The remote JSONL retains the original source provenance. Validation also uses `data_source: ls_kw`; its raw JSONL retains the original clean/other source labels.

Task formatting uses `lang_asr_verb`, `language: English`, `lang_hint: true`, and `prefix_prob: 0.0`. The LS keyword base imports these settings directly from this data YAML; validation formatting is defined in `data/val_data/ls_entity_val.yaml`.

Use this YAML as the complete targeted training subset, or mix it with a general training corpus using an explicitly chosen sampling weight. Historical component subsets overlap with this combined set and should not be added again. No sampling weight or training recipe is prescribed here.

All 101 uploaded payload files passed SHA256 readback checks. All 819 distinct audio files/chunks referenced across the exported subsets existed at verification time. The repository ASR dataset loader successfully loaded and formatted all 743 rows directly from this YAML and its remote JSONL.

## Interpretation limits

Selection was motivated by LibriSpeech test-clean/test-other errors, so these benchmarks are targeted diagnostic measures rather than independent evidence of generalization. Same spelling does not guarantee the same person/place or pronunciation.

The 36 new Libriheavy clips preserve published book-text labels and cut timestamps. They exclude LibriSpeech dev/test reader IDs, matching evaluation text, duplicates, and overlapping source-audio cuts. Original MP3 duration was checked against the published recording duration, and clips were converted to 16 kHz PCM. The original 707 samples were not retroactively audited at audio-fingerprint level. Not every clip has been manually listened to, and training gains have not been measured.

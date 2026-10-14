# Indian locale ~10k training subsets

Statistics for the 12 `*_in_10k.yaml` training-data configs.

## Selection and accounting

- Target: approximately 10,000 samples per locale, distributed across every
  retained dataset folder. All totals are within 2% of the target.
- Selection is deterministic (seed 42) and uses whole chunks. Original chunk
  names, sample counts and durations are preserved.
- Folder allocations favor balanced sample counts subject to whole-chunk sizes.
  They are not proportional to the full datasets' sizes.
- Excludes all `with_noise` and `_w_noise` variants and folders without `.audio`.
- Samples are summed from `fileInfo.count`; hours are summed from
  `fileInfo.duration` in seconds and divided by 3,600. Durations are
  manifest-reported, not measured by decoding audio.
- Original and simulated datasets are separate sources; no cross-source
  utterance deduplication or Mix/Native filtering is applied.
- Data sources remain `<locale>_tr`, for example `hi_in_tr`.
- Each source folder has its own subset manifest named `fileset.10k.json`.
  The filename identifies its share of the locale-wide subset, not 10,000
  samples in that individual folder.

## Per-locale totals

| Locale | Training YAML | Samples | Chunks | Folders | Hours |
|---|---|---:|---:|---:|---:|
| as-IN | [as_in_10k.yaml](./as_in_10k.yaml) | 9,850 | 20 | 2 | 17.05 |
| bn-IN | [bn_in_10k.yaml](./bn_in_10k.yaml) | 9,991 | 20 | 1 | 19.51 |
| gu-IN | [gu_in_10k.yaml](./gu_in_10k.yaml) | 10,076 | 21 | 1 | 20.63 |
| hi-IN | [hi_in_10k.yaml](./hi_in_10k.yaml) | 10,017 | 8 | 5 | 39.11 |
| kn-IN | [kn_in_10k.yaml](./kn_in_10k.yaml) | 9,984 | 20 | 1 | 19.13 |
| ml-IN | [ml_in_10k.yaml](./ml_in_10k.yaml) | 9,970 | 8 | 6 | 19.21 |
| mr-IN | [mr_in_10k.yaml](./mr_in_10k.yaml) | 9,880 | 20 | 2 | 19.11 |
| or-IN | [or_in_10k.yaml](./or_in_10k.yaml) | 9,892 | 20 | 2 | 17.01 |
| pa-IN | [pa_in_10k.yaml](./pa_in_10k.yaml) | 9,871 | 20 | 2 | 19.79 |
| ta-IN | [ta_in_10k.yaml](./ta_in_10k.yaml) | 9,989 | 20 | 1 | 19.01 |
| te-IN | [te_in_10k.yaml](./te_in_10k.yaml) | 9,873 | 11 | 2 | 16.10 |
| ur-IN | [ur_in_10k.yaml](./ur_in_10k.yaml) | 9,877 | 20 | 2 | 19.50 |
| **Total** | **12 configs** | **119,270** | **208** | **27** | **245.18** |

Totals use unrounded durations; individually rounded rows may not sum exactly.

## Per-folder allocation

| Locale | Dataset folder | Samples | Chunks | Hours |
|---|---|---:|---:|---:|
| as-IN | AI4Bharat | 4,999 | 10 | 8.73 |
| as-IN | AI4BharatSimu | 4,851 | 10 | 8.33 |
| bn-IN | AI4Bharat | 9,991 | 20 | 19.51 |
| gu-IN | AI4Bharat | 10,076 | 21 | 20.63 |
| hi-IN | Normalized_Combined_Corrected | 2,017 | 1 | 3.86 |
| hi-IN | IndicVoices | 2,000 | 1 | 4.57 |
| hi-IN | Callcenter_monostream | 2,000 | 1 | 13.22 |
| hi-IN | Callcenter_monostream_Sulekha | 2,000 | 1 | 13.96 |
| hi-IN | AI4Bharat | 2,000 | 4 | 3.50 |
| kn-IN | AI4Bharat | 9,984 | 20 | 19.13 |
| ml-IN | mlIN_KaryaVoice_B2_withwav | 1,998 | 1 | 3.46 |
| ml-IN | mlIN_KaryaVoice_withwav | 1,998 | 1 | 3.46 |
| ml-IN | mlIN_Kathbath_withwav | 1,997 | 1 | 5.89 |
| ml-IN | mlIN_Shruthilipi_withwav | 2,000 | 1 | 2.57 |
| ml-IN | AI4Bharat | 1,000 | 2 | 2.01 |
| ml-IN | AI4BharatSimu | 977 | 2 | 1.82 |
| mr-IN | AI4Bharat | 5,000 | 10 | 9.79 |
| mr-IN | AI4BharatSimu | 4,880 | 10 | 9.32 |
| or-IN | AI4Bharat | 4,997 | 10 | 8.67 |
| or-IN | AI4BharatSimu | 4,895 | 10 | 8.34 |
| pa-IN | AI4Bharat | 4,998 | 10 | 10.09 |
| pa-IN | AI4BharatSimu | 4,873 | 10 | 9.70 |
| ta-IN | AI4Bharat | 9,989 | 20 | 19.01 |
| te-IN | AudioFiltered2 | 5,880 | 3 | 7.95 |
| te-IN | AI4Bharat | 3,993 | 8 | 8.15 |
| ur-IN | AI4Bharat | 4,999 | 10 | 10.07 |
| ur-IN | AI4BharatSimu | 4,878 | 10 | 9.43 |

## Manifest and chunk locations

Subset manifests are published at:

```text
az://orngwus2cresco/data/speech/users/boren/DataSpecs/mlang_asr_data_2607/indian_10k/<locale>/<dataset>/fileset.10k.json
```

Here `<locale>` uses the YAML naming convention, for example `as_in`.
The YAMLs retain the original Orange chunk paths; subset creation does not
duplicate the audio or transcription chunks.

Local manifest copies and machine-readable statistics:

```text
/home/boren/data/asr_manifests/mlang_asr_data_2607/subsets_10k/<locale>/<dataset>/fileset.10k.json
/home/boren/data/asr_manifests/mlang_asr_data_2607/subsets_10k/summary.json
```

All 27 remote subset manifests were verified byte-for-byte against local
copies, and all 12 YAMLs passed Hydra composition. Selected records were
verified against their original source manifests.

At 2026-10-14 06:11 UTC, all 208 selected chunks' 1,040 associated files
(55,749,868,503 bytes, approximately 55.75 GB) were verified in Orange against
the source inventory's file sizes. The 10k subsets are fully available.
The separate bulk transfer of the remaining full-dataset chunks may still be
in progress; it is not required to load these subset manifests.

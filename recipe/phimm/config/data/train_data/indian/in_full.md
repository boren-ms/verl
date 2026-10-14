# Indian training data: full-source and final S2 statistics

Statistics snapshot: 2026-10-14 UTC. Original source accounting was collected at
04:54 UTC; final S2 accounting was collected at 05:23 UTC.

## Scope and definitions

This report covers 12 Indian locales and all 44 dataset sources referenced by the
[final S2 data recipe](../../prod/data-v3.5.4-s2-final.yaml).
It distinguishes three selections:

1. **Original full sources:** the original 2607 locale manifests and their
   `file_set.json` files, including noisy and feature-only sources.
2. **Final S2:** the manifests selected by the final S2 recipe. Hindi uses
   smaller sampled file sets; the other locales retain the original chunk records.
3. **Current locale YAMLs:** original file sets restricted to the 27 non-noise,
   audio-backed sources requested for transfer and verl training.

Counts and durations are taken from `fileInfo`: `count` is the number of
utterances in a chunk and `duration` is seconds. Hours are `sum(duration) / 3600`.
A logical chunk can have multiple blobs, such as `.audio`, `.feature`, `.info`,
`.transcription`, and `.json`.

**Audio-backed** means a matching, nonempty `.audio` blob exists for the exact
chunk name. The three no-audio folders contain features and are not necessarily
unusable by the production S2 feature-based loader.

Sampling-adjusted hours are arithmetic estimates:

```text
sampling-adjusted hours = manifest hours * sample_ratio * dataset weight
```

They are not realized sample counts or post-filter training durations.
Totals are summed before rounding. Original, simulated, and noisy datasets are
counted separately; utterances have not been deduplicated across datasets.

## Summary

| Selection | Sources | Chunks | Utterances | Raw hours | Sampling-adjusted hours |
|---|---:|---:|---:|---:|---:|
| Original full sources | 44 | 27,204 | 38,084,961 | 47,099.25 | Not applied |
| Original sources with audio | 41 | 21,980 | 27,556,728 | 30,572.49 | Not applied |
| Current locale YAMLs: non-noise and audio-backed | 27 | 13,987 | 11,890,285 | 20,775.90 | Not applied |
| Final S2, all sources | 44 | 23,294 | 30,867,927 | 34,140.73 | 29,130.87 |
| Final S2, audio-backed sources | 41 | 20,626 | 25,497,411 | 26,585.28 | 25,028.29 |
| Final S2 selections restricted to our 27 retained sources | 27 | 12,637 | 9,837,636 | 16,829.62 | 15,298.41 |

The current locale YAMLs do **not** apply Hindi's S2 sampling ratio or the
production S2 runtime filters.

## Original full sources by locale

The original locale manifests are located under:

```text
/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2607/tier2-IN/
    tier2_<locale>_non_verbatim_lid.json
```

| Locale | Sources | Chunks | Utterances | Raw hours |
|---|---:|---:|---:|---:|
| as-IN | 5 | 2,087 | 1,467,065 | 1,798.13 |
| bn-IN | 2 | 1,677 | 2,355,368 | 1,540.16 |
| gu-IN | 2 | 1,366 | 2,289,099 | 2,914.86 |
| hi-IN | 8 | 6,931 | 12,788,893 | 22,978.23 |
| kn-IN | 2 | 1,177 | 1,509,054 | 1,313.68 |
| ml-IN | 8 | 1,849 | 2,586,803 | 2,733.78 |
| mr-IN | 3 | 2,346 | 2,933,417 | 2,365.05 |
| or-IN | 4 | 2,006 | 2,008,920 | 1,914.84 |
| pa-IN | 3 | 1,523 | 1,234,581 | 1,488.28 |
| ta-IN | 2 | 2,860 | 4,479,943 | 2,694.29 |
| te-IN | 2 | 1,846 | 2,832,257 | 3,800.16 |
| ur-IN | 3 | 1,536 | 1,599,561 | 1,557.80 |
| **Total** | **44** | **27,204** | **38,084,961** | **47,099.25** |

## Current per-locale training YAMLs

These configs load `audio` and `transcription` chunks with `task: lang_asr`.
Each uses a distinct `<locale>_tr` data source. They omit all dataset names
containing `with_noise` or `_w_noise`, and every folder with no audio.

| Locale | YAML | Data source | Sources | Chunks | Utterances | Hours |
|---|---|---|---:|---:|---:|---:|
| as-IN | [as_in.yaml](./as_in.yaml) | `as_in_tr` | 2 | 1,788 | 880,871 | 1,511.26 |
| bn-IN | [bn_in.yaml](./bn_in.yaml) | `bn_in_tr` | 1 | 657 | 328,099 | 631.52 |
| gu-IN | [gu_in.yaml](./gu_in.yaml) | `gu_in_tr` | 1 | 206 | 102,545 | 207.94 |
| hi-IN | [hi_in.yaml](./hi_in.yaml) | `hi_in_tr` | 5 | 2,396 | 3,640,595 | 7,008.69 |
| kn-IN | [kn_in.yaml](./kn_in.yaml) | `kn_in_tr` | 1 | 557 | 277,956 | 533.23 |
| ml-IN | [ml_in.yaml](./ml_in.yaml) | `ml_in_tr` | 6 | 1,130 | 1,157,446 | 2,011.36 |
| mr-IN | [mr_in.yaml](./mr_in.yaml) | `mr_in_tr` | 2 | 1,146 | 564,701 | 1,093.50 |
| or-IN | [or_in.yaml](./or_in.yaml) | `or_in_tr` | 2 | 1,326 | 653,790 | 1,117.45 |
| pa-IN | [pa_in.yaml](./pa_in.yaml) | `pa_in_tr` | 2 | 1,148 | 566,465 | 1,145.23 |
| ta-IN | [ta_in.yaml](./ta_in.yaml) | `ta_in_tr` | 1 | 821 | 409,609 | 780.11 |
| te-IN | [te_in.yaml](./te_in.yaml) | `te_in_tr` | 2 | 1,846 | 2,832,257 | 3,800.16 |
| ur-IN | [ur_in.yaml](./ur_in.yaml) | `ur_in_tr` | 2 | 966 | 475,951 | 935.45 |
| **Total** | | | **27** | **13,987** | **11,890,285** | **20,775.90** |

The corresponding transfer inventory contains **71,402 chunk-directory blobs**
totaling **4,762,654,917,399 bytes** (4.76 decimal TB), plus 27 referenced file-set
manifests. Blob counts include companion files and directory markers, not just
logical audio chunks. This is the transfer scope, not a statement of completion.

## Final S2 statistics by locale

Raw chunk and utterance counts include every source selected by S2, including
noise variants and folders without audio.

| Locale | Sources | Raw chunks | Raw utterances | Raw hours | Sample ratio | Adjusted raw hours | Audio-backed hours | Adjusted audio-backed hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| as-IN | 5 | 2,087 | 1,467,065 | 1,798.13 | 1.0 | 1,798.13 | 1,629.69 | 1,629.69 |
| bn-IN | 2 | 1,677 | 2,355,368 | 1,540.16 | 1.0 | 1,540.16 | 1,540.16 | 1,540.16 |
| gu-IN | 2 | 1,366 | 2,289,099 | 2,914.86 | 1.0 | 2,914.86 | 2,914.86 | 2,914.86 |
| hi-IN | 8 | 3,021 | 5,571,859 | 10,019.71 | 0.5 | 5,009.86 | 3,113.98 | 1,556.99 |
| kn-IN | 2 | 1,177 | 1,509,054 | 1,313.68 | 1.0 | 1,313.68 | 1,313.68 | 1,313.68 |
| ml-IN | 8 | 1,849 | 2,586,803 | 2,733.78 | 1.0 | 2,733.78 | 2,252.51 | 2,252.51 |
| mr-IN | 3 | 2,346 | 2,933,417 | 2,365.05 | 1.0 | 2,365.05 | 2,365.05 | 2,365.05 |
| or-IN | 4 | 2,006 | 2,008,920 | 1,914.84 | 1.0 | 1,914.84 | 1,914.84 | 1,914.84 |
| pa-IN | 3 | 1,523 | 1,234,581 | 1,488.28 | 1.0 | 1,488.28 | 1,488.28 | 1,488.28 |
| ta-IN | 2 | 2,860 | 4,479,943 | 2,694.29 | 1.0 | 2,694.29 | 2,694.29 | 2,694.29 |
| te-IN | 2 | 1,846 | 2,832,257 | 3,800.16 | 1.0 | 3,800.16 | 3,800.16 | 3,800.16 |
| ur-IN | 3 | 1,536 | 1,599,561 | 1,557.80 | 1.0 | 1,557.80 | 1,557.80 | 1,557.80 |
| **Total** | **44** | **23,294** | **30,867,927** | **34,140.73** | | **29,130.87** | **26,585.28** | **25,028.29** |

The actual Indian block weight is **1.0**, despite its "2x exposure" heading.
Urdu is enabled despite the nearby comment saying it is deferred to stage 2.

## All 44 chunk folders and source-level statistics

The dataset name in the table identifies this exact folder:

```text
/datablob1/am_data/E2E/pjham/Multilingual/<language>/display_corrected_gpt2_lid/<dataset>/chunks/
```

`<language>` is the lowercase language code before `-IN`, such as `as`, `hi`, or
`te`. The source and Orange locations preserve the relative path:

```text
Source:
https://highperf01safn.blob.core.windows.net/data/am_data/E2E/pjham/Multilingual/<language>/display_corrected_gpt2_lid/<dataset>/chunks/

Orange:
az://orngwus2cresco/data/speech/am_data/E2E/pjham/Multilingual/<language>/display_corrected_gpt2_lid/<dataset>/chunks/
```

The following S2 counts and hours are **before** `sample_ratio` and runtime
filtering. "Included" refers to the current YAML/transfer scope, not transfer
completion. Excluded sources remain present in the S2 recipe.

| Locale | Dataset folder | Original hours | S2 chunks | S2 utterances | S2 hours | YAML/transfer selection |
|---|---|---:|---:|---:|---:|---|
| as-IN | `AI4Bharat` | 774.56 | 894 | 446,281 | 774.56 | included |
| as-IN | `AI4BharatSimu` | 736.69 | 894 | 434,590 | 736.69 | included |
| as-IN | `SpeechOcean_KingASR_30082023_with_noise` | 112.82 | 70 | 138,044 | 112.82 | noise |
| as-IN | `tn_transliterated_111222_with_noise` | 168.44 | 219 | 442,636 | 168.44 | noise; no audio |
| as-IN | `tn_transliterated_19072023_with_noise` | 5.61 | 10 | 5,514 | 5.61 | noise |
| bn-IN | `AI4Bharat` | 631.52 | 657 | 328,099 | 631.52 | included |
| bn-IN | `tn_transliterated_03012024_with_noise` | 908.64 | 1,020 | 2,027,269 | 908.64 | noise |
| gu-IN | `AI4Bharat` | 207.94 | 206 | 102,545 | 207.94 | included |
| gu-IN | `tn_transliterated_alldata_28052024_with_noise` | 2,706.93 | 1,160 | 2,186,554 | 2,706.93 | noise |
| hi-IN | `AI4Bharat` | 673.88 | 333 | 166,460 | 294.76 | included |
| hi-IN | `Callcenter_monostream` | 67.91 | 3 | 4,258 | 28.25 | included |
| hi-IN | `Callcenter_monostream_Sulekha` | 65.46 | 3 | 4,668 | 32.66 | included |
| hi-IN | `Callcenter_monostream_Sulekha_w_noise` | 30.13 | 2 | 4,000 | 25.73 | noise |
| hi-IN | `Callcenter_monostream_w_noise` | 62.37 | 3 | 4,258 | 25.83 | noise |
| hi-IN | `CleanedFiltered_WordCountMatch` | 15,877.05 | 1,970 | 3,975,655 | 6,905.74 | no audio |
| hi-IN | `IndicVoices` | 497.05 | 59 | 115,779 | 224.67 | included |
| hi-IN | `Normalized_Combined_Corrected` | 5,704.39 | 648 | 1,296,781 | 2,482.08 | included |
| kn-IN | `AI4Bharat` | 533.23 | 557 | 277,956 | 533.23 | included |
| kn-IN | `tn_transliterated_03012024_with_noise` | 780.45 | 620 | 1,231,098 | 780.45 | noise |
| ml-IN | `AI4Bharat` | 352.17 | 364 | 181,730 | 352.17 | included |
| ml-IN | `AI4BharatSimu` | 334.96 | 364 | 177,030 | 334.96 | included |
| ml-IN | `mlIN_KaryaVoice_B2_withwav` | 226.18 | 66 | 130,358 | 226.18 | included |
| ml-IN | `mlIN_KaryaVoice_withwav` | 300.69 | 87 | 173,190 | 300.69 | included |
| ml-IN | `mlIN_Kathbath_withwav` | 268.22 | 46 | 89,968 | 268.22 | included |
| ml-IN | `mlIN_Shruthilipi_withwav` | 529.14 | 203 | 405,170 | 529.14 | included |
| ml-IN | `tn_transliterated_111222_with_noise` | 481.27 | 479 | 952,225 | 481.27 | noise; no audio |
| ml-IN | `tn_transliterated_240323_with_noise` | 241.14 | 240 | 477,132 | 241.14 | noise |
| mr-IN | `AI4Bharat` | 560.53 | 573 | 286,073 | 560.53 | included |
| mr-IN | `AI4BharatSimu` | 532.97 | 573 | 278,628 | 532.97 | included |
| mr-IN | `tn_transliterated_alldata_28052024_with_noise` | 1,271.55 | 1,200 | 2,368,716 | 1,271.55 | noise |
| or-IN | `AI4Bharat` | 572.81 | 663 | 331,267 | 572.81 | included |
| or-IN | `AI4BharatSimu` | 544.65 | 663 | 322,523 | 544.65 | included |
| or-IN | `tn_transliterated_140323_with_noise` | 490.28 | 440 | 876,748 | 490.28 | noise |
| or-IN | `tn_transliterated_19072023_with_noise` | 307.10 | 240 | 478,382 | 307.10 | noise |
| pa-IN | `AI4Bharat` | 586.60 | 574 | 286,874 | 586.60 | included |
| pa-IN | `AI4BharatSimu` | 558.63 | 574 | 279,591 | 558.63 | included |
| pa-IN | `tn_transliterated_111023_with_noise` | 343.05 | 375 | 668,116 | 343.05 | noise |
| ta-IN | `AI4Bharat` | 780.11 | 821 | 409,609 | 780.11 | included |
| ta-IN | `tn_transliterated_all_03112023_with_noise` | 1,914.18 | 2,039 | 4,070,334 | 1,914.18 | noise |
| te-IN | `AI4Bharat` | 550.65 | 536 | 267,675 | 550.65 | included |
| te-IN | `AudioFiltered2` | 3,249.51 | 1,310 | 2,564,582 | 3,249.51 | included |
| ur-IN | `AI4Bharat` | 479.46 | 483 | 241,114 | 479.46 | included |
| ur-IN | `AI4BharatSimu` | 455.99 | 483 | 234,837 | 455.99 | included |
| ur-IN | `tn_transliterated_111023_with_noise` | 622.34 | 570 | 1,123,610 | 622.34 | noise |

## Missing-audio exclusions

All blobs under each of these chunk prefixes were listed. No `.audio` files were
found. Every manifest chunk in the other 41 source folders had a corresponding
nonempty audio blob; no unreferenced audio blobs were found in the original
inventory.

| Locale / folder | Original missing chunks | Original missing utterances | Original missing hours | S2 missing chunks | S2 missing utterances | S2 missing hours |
|---|---:|---:|---:|---:|---:|---:|
| as-IN / `tn_transliterated_111222_with_noise` | 219 | 442,636 | 168.44 | 219 | 442,636 | 168.44 |
| hi-IN / `CleanedFiltered_WordCountMatch` | 4,526 | 9,133,372 | 15,877.05 | 1,970 | 3,975,655 | 6,905.74 |
| ml-IN / `tn_transliterated_111222_with_noise` | 479 | 952,225 | 481.27 | 479 | 952,225 | 481.27 |
| **Total** | **5,224** | **10,528,233** | **16,526.76** | **2,668** | **5,370,516** | **7,555.45** |

The exclusions overlap: 16 folders are noise variants, and three have no audio.
Two folders meet both conditions. Consequently, 17 distinct folders are excluded
and 27 remain.

## Differences between final S2 and the current YAMLs

- **Physical chunk folders:** S2 references the same 44 original chunk folders;
  the current YAMLs retain only the 27 allowed folders.
- **Hindi file sets:** S2 substitutes sampled `file_set_<number>.json` manifests.
  Its complete selection removes 3,910 logical chunks relative to the original
  Hindi file sets. Shared chunk records retain their original counts/durations.
- **Hindi on our five retained sources:** original selection is 2,396 chunks,
  3,640,595 utterances and 7,008.69 hours. S2 selects 1,046 chunks, 1,587,946
  utterances and 3,062.42 hours; its 0.5 ratio gives 1,531.21 adjusted hours.
- **Other locales:** original file-set records are unchanged, even when the
  enclosing locale manifest has a `_wordratio` suffix.
- **Alternate info:** all locales except Urdu use `additional_data_path.info`
  under `/datablob1/users/v-wanhu/TaskForRui/Word_ratio/<language>/...`.
  These alternate info paths are not part of the current transfer inventory.
- **Urdu:** uses the original locale manifest without alternate word-ratio info.

The final S2 locale manifests are rooted at:

```text
/datablob1/users/v-wanhu/TaskForRui/Word_ratio/Json_sample/users/ruchaofan/DataSpecs/mlang_asr_data_2607/tier2-IN/
```

Their filenames are `tier2_<locale>_non_verbatim_lid_wordratio.json`, except Urdu,
whose filename is `tier2_ur-IN_non_verbatim_lid.json`.

## Runtime filters and limitations

The production Indian profile applies:

- `remove_utterance_by_word_ratio: true`, `max_other_ratio: 0.1`;
- `info_field_filter`: keep `lid_mix_label` in `Mix` or `Native`;
- `max_utterance_frames: 4000`, `min_utterance_frames: 0`;
- `max_utterance_labels: 800`, `min_utterance_labels: 1`;
- `mix_normal_sr: 0.94`;
- biasing with `buffer_size: 10000`, `max_piece_len: 5`, `max_num: 4000`,
  `hit_prob: 0.3` and `max_hit_ratio: 0.3`;
- `audio_frame_crop_prob: 0.1`.

These statistics do not apply those runtime filters, biasing, or cropping.
Durations are manifest-reported, not obtained by decoding every audio file.
The current locale YAMLs do not reproduce the final S2 runtime profile.
Audio availability was checked on the corporate source, not inferred from
whether a folder name exists in Orange. Dataset loading should wait for the
corresponding Orange transfer and verification to finish.

## Machine-readable reports

The local review directory is:

```text
/home/boren/data/asr_manifests/mlang_asr_data_2607/tier2-IN/
```

It contains:

- `indian_audio_backed_stats.json` and `indian_audio_backed_stats.csv`:
  original source statistics, audio availability and missing chunk names.
- `indian_final_s2_comparison.json`: original versus S2 file-set selections
  and exact alternate info paths.
- `indian_s2_final_locales.csv`: locale-level raw, audio-backed,
  sampling-adjusted and retained-source statistics.
- `indian_s2_final_datasets.csv`: all 44 dataset rows with source paths,
  noise/exclusion flags and counts.
- `indian_s2_final_statistics.json`: complete S2 statistics and filter settings.

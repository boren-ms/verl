# English Earnings Data Audit

Source: `stage2-s2s_verbatim_ft.yaml`

Audit date: 2026-09-12

## Summary

The source YAML contains three active English earnings-related dataset entries. Sample count means the sum of `fileInfo[].count` in the dataset's `file_set.json` manifest.

| Dataset | Variant | YAML lines | Samples | Manifest status | Audio status | Transcription status |
| --- | --- | ---: | ---: | --- | --- | --- |
| `EarningsCall_FY22Q2_L1_en-US_16k_Fixed` | R10 original | 405-408 | Unknown | Legacy `/datablob` path is not mounted; no corresponding object was found in Orange | Not verifiable | Not verifiable; no separate transcription path is declared |
| `EarningsCall_FY22Q2_L1_en-US_16k_Fixed` | R10 augmented | 488-491 | Unknown | Legacy `/datablob` path is not mounted; no corresponding object was found in Orange | Not verifiable | Not verifiable; no separate transcription path is declared |
| `en-US_Videoindex_PT_Earningscall_FY22_TRAIN` | R12 fixed-contraction | 564-568 | **343,726** | Exact legacy path is not mounted; migrated Orange manifest exists | **Yes**: 3,438/3,438 chunk objects | **Yes**: 3,438/3,438 chunk objects |

Verified total: **343,726 samples**. The total for all three entries cannot be computed until the two R10 manifests are accessible.

## Dataset Paths

### R10 Original

- Data path: `/datablob/am_data/en/labeled_data_prep_e2e/r10/MixedCaseData/EarningsCall_FY22Q2_L1_en-US_16k_Fixed/ChunkFiles/`
- Manifest: `/datablob/am_data/en/labeled_data_prep_e2e/r10/Display_Dpp/EarningsCall_FY22Q2_L1_en-US_16k_Fixed/sanity_checked_file_set/file_set.json`
- Info path: `/datablob/am_data/en/labeled_data_prep_e2e/r10/Display_Dpp/EarningsCall_FY22Q2_L1_en-US_16k_Fixed/ChunkFiles/`

### R10 Augmented

- Data path: `/datablob/am_data/en/labeled_data_prep_e2e/r10_augmented/MixedCaseData/EarningsCall_FY22Q2_L1_en-US_16k_Fixed/ChunkFiles/`
- Manifest: `/datablob/am_data/en/labeled_data_prep_e2e/r10_augmented/Display_Dpp/EarningsCall_FY22Q2_L1_en-US_16k_Fixed/sanity_checked_file_set/file_set.json`
- Info path: `/datablob/am_data/en/labeled_data_prep_e2e/r10/Display_Dpp/EarningsCall_FY22Q2_L1_en-US_16k_Fixed/ChunkFiles/`

### R12 Fixed-Contraction

Configured legacy paths:

- Data path: `/datablob/am_data/en/labeled_data_prep_e2e_fixed_contraction/r12/SentencePiece/en-US_Videoindex_PT_Earningscall_FY22_TRAIN/ChunkFiles/`
- Manifest: `/datablob/am_data/en/labeled_data_prep_e2e_fixed_contraction/r12/Display_Dpp/en-US_Videoindex_PT_Earningscall_FY22_TRAIN/sanity_checked_file_set/file_set.json`
- Transcription path: `/datablob/am_data/en/labeled_data_prep_e2e_fixed_contraction/r12/MixedCaseData/en-US_Videoindex_PT_Earningscall_FY22_TRAIN/ChunkFiles/`
- Info path: `/datablob/am_data/en/labeled_data_prep_e2e_fixed_contraction/r12/Display_Dpp/en-US_Videoindex_PT_Earningscall_FY22_TRAIN/ChunkFiles/`

Verified migrated paths:

- Manifest: `az://orngwus2cresco/data/speech/am_data/en/display_data_prep_v3/short_form_fix_disfluency_v2/r12/en-US_Videoindex_PT_Earningscall_FY22_TRAIN/sanity_checked_file_set/file_set.json`
- Audio and transcription chunks: `az://orngwus2cresco/data/speech/am_data/en/display_data_prep_v3/short_form_fix/r12/en-US_Videoindex_PT_Earningscall_FY22_TRAIN/`

Manifest details:

- Chunks: **3,438**
- Samples: **343,726**
- Duration: **2,071,703.44 seconds** (**575.47 hours**)
- Audio objects found: **3,438**; missing from manifest set: **0**
- Transcription objects found: **3,438**; missing from manifest set: **0**

## Verification Notes

- The configured `/datablob` files do not exist on the local machine or on ready Brix node `verl-n1-i7`; that node has no `/datablob`, `/datablob1`, or `/modelblob` mount.
- The guessed direct Orange mirrors of the two R10 manifests were also absent.
- Corp storage checks could not be completed because the current Azure identity was rejected by those storage accounts. Therefore, R10 data is reported as **not verifiable**, not as proven deleted.
- The R12 count and object checks use the migrated Orange manifest and require exact chunk-name matches for both `.audio` and `.transcription` objects.

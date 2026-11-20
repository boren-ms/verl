# dtest_enus_ter

The adjacent YAML selects exactly MoE's five
`disfluencytolerant_ter / default / Tier1 / en-US` rows from
`/home/boren/code/MoE/scripts/audio_eval/result_orders/2611_eval_rows.tsv`.
No other locale or dataset is selected. Three meeting sets are shared with EER.

All five use the existing `InhouseASR_2605_seg_presegment_30s/en-US` manifests.
Full recording references and ordered paths/selectors are preserved; no audio
is resegmented or sampled. Three short dictation recordings retain their existing
whole-recording paths without selectors. The loader guard is 60 seconds.

Normalized manifest root:
`az://orngwus2cresco/data/boren/data/Evaluation/dtest_enus_ter_v1`.
Original audio/reference root:
`az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US`.

MoE `run_SI_local.py` uses **DisplayTranscription** with
`getdfmetrics ter --locale en-us --ter_type disfluencytolerant`.
The scorer uses that Python backend from **ter 2.20.0**, shipped with
**speechinsight_tools_latest 2.28.4**, installed in an isolated runtime directory.
TER-only presets disable EER scoring. Primary score: equal-weight mean of the
five per-dataset rates. Corpus count-weighted TER is separately labeled.
The prompt uses the shared display task with the required English language hint:

```text
Transcribe the audio clip into text.<audio>
The language is English.
```

Use `inhouse_entity_report --language-hint English` to check every segment's saved prompt.

```bash
python -m recipe.phimm.data.inhouse_dtest_enus --suite ter --cache-dir /tmp/workbook_ter --publish
python -m recipe.phimm.main_asr_eval --config-name eval_enus_ter --cfg job --resolve
python -m recipe.phimm.tools.inhouse_entity_report --suite ter --language-hint English --result-root <val_data_gen> --output-dir <report_dir>
```


## Dataset statistics

| Dataset | Recordings | Segments | Display words |
|---|---:|---:|---:|
| Conversation_DTEST_FY21Q1_en-US | 32 | 731 | 44,154 |
| Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection | 10 | 740 | 50,953 |
| OnlineMeetings_CS_Product_FY22_en-US_DTEST | 5 | 596 | 40,231 |
| OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST | 5 | 734 | 46,007 |
| Dictation_Commonset_OfficeOffline_FY24Q3_en-US_DTEST_OfflineDataCollection | 113 | 774 | 39,292 |
| **Total** | **165** | **3,575** | **220,637** |

| Dataset | Segments / recording min / mean / max | Clip seconds min / mean / max | Words / recording min / mean / max |
|---|---:|---:|---:|
| Conversation_DTEST_FY21Q1_en-US | 3.00 / 22.84 / 183.00 | 0.27 / 24.82 / 29.99 | 121.00 / 1379.81 / 10535.00 |
| Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection | 53.00 / 74.00 / 82.00 | 0.64 / 24.60 / 29.99 | 3391.00 / 5095.30 / 6385.00 |
| OnlineMeetings_CS_Product_FY22_en-US_DTEST | 73.00 / 119.20 / 153.00 | 0.26 / 24.65 / 29.99 | 5228.00 / 8046.20 / 10378.00 |
| OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST | 129.00 / 146.80 / 164.00 | 0.74 / 25.03 / 29.99 | 8241.00 / 9201.40 / 9930.00 |
| Dictation_Commonset_OfficeOffline_FY24Q3_en-US_DTEST_OfflineDataCollection | 1.00 / 6.85 / 13.00 | 1.11 / 25.25 / 29.99 | 20.00 / 347.72 / 703.00 |

Display words are whitespace tokens containing letters/digits after removing
XML tags and timing markers; they are inventory counts, not official TER tokens.
Existing segment rows contain full-recording references, so aligned word counts
per segment are unavailable and are not estimated. Segment durations come from
the existing boundaries. The full machine-readable statistics and manifest hashes
are saved in `preparation.json` beside the published normalized manifests.


Prompts use the shared dataset transform:
```yaml
add_task_info:
  style: display
  language: English
  lang_hint: true
```
This generates the standard display task with an English language hint.
Full recording `DisplayTranscription` remains the scoring reference.

Each YAML contains five independent dataset entries with one manifest per entry.
Short data-source names route scores and label the saved results.

Data sources: `conversation`, `meetings`, `product`, `shiproom`, `dictation`.

The shared display-style prompt omits an explicit formatting instruction;
`style: display` still selects the standard `READABLE` reference wrapper.

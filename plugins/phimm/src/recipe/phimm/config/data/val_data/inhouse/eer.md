# dtest_enus_eer

The active EER suite contains exactly the five `Tier1 / en-US / eer / default` rows in
`/home/boren/code/MoE/scripts/audio_eval/result_orders/2611_eval_rows.tsv`.
No other dataset is selected by the adjacent YAML. All existing segment selectors
are retained; no audio is segmented, rewritten, or sampled.

| Workbook dataset | Recordings | Segments | Existing segment version | Workbook 2611 S2 EER |
|---|---:|---:|---|---:|
| CustomerSpeech Banking FY23Q4 | 36 | 626 | InhouseASR_2605_seg_presegment | 7.62% |
| CustomerSpeech Medical FY23Q4 | 69 | 1,193 | InhouseASR_2605_seg_presegment | 6.51% |
| OnlineMeetings Shiproom FY22 | 5 | 734 | InhouseASR_2605_seg_presegment_30s | 46.77% |
| OnlineMeetings Product FY22 | 5 | 596 | InhouseASR_2605_seg_presegment_30s | 31.07% |
| Conversation OnlineMeetings FY25Q3 | 10 | 740 | InhouseASR_2605_seg_presegment_30s | 23.77% |
| **Total** | **125** | **3,889** | | |

Banking/Medical existing segments reach 39.99 seconds; the three meeting sets
use the existing 30-second version. The evaluation retains complete clips with
a 60-second loader guard. One normalized row contains all ordered clips for an
original UUID. Both full original reference strings are preserved and checked
against every segment's copied references.

Reference root:
`az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US`

Normalized manifest root:
`az://orngwus2cresco/data/boren/data/Evaluation/dtest_enus_eer_v1`

Prepare from existing manifests:

```bash
python -m recipe.phimm.data.inhouse_dtest_enus --suite eer --cache-dir /tmp/workbook_eer --publish
```

MoE `run_SI_local.py` uses **DisplayTranscription** and
**GetMetrics --nondisfluency -l en-us** for `-m ewer`. EER is
`1 - NumTransEntsMatched / NumTransEnts`, distinct from entity word error rate.
The workbook's average gives each of the five dataset EERs equal weight.
The displayed S2 values average to 23.148%, reported as 23.15%.
Our reporter labels the count-weighted aggregate separately.

The YAML uses the shared `add_task_info` display task with an English
language hint:

```text
Transcribe the audio clip into text.<audio>
The language is English.
```

Use `inhouse_entity_report --language-hint English`
to verify the saved base prompt for every decoded segment.
Text h1 adds only the preceding generated transcript; context-OFF adds none.
Existing pre-segment boundaries and context controls are retained, so matching
the dataset list, prompt and scorer does not establish historical decoding parity.

Historical 31-subset `en-US-entity-v3` reports used tagged Transcription with
--verbatim; they are not results for this five-dataset workbook protocol.
Use `inhouse_entity_report --suite legacy_entity_v3` only to audit those old runs.

The scorer uses MoE's **2026-11-10** pinned bundle from source revision
`fa0fe75ffbb28397058a3fa96a2842de6540b889`. Its manifest is copied verbatim to
`recipe/phimm/reward/speechinsight_tools_manifest.json` and shipped with the plugin.
The verified mirror is
`az://orngwus2cresco/data/speech/users/boren/eval_packages/inhouse_moe_20261110/speechinsight_tools_latest/`.

EER uses **SpeechInsight 2.28.4** GetMetrics; its SHA-256 is unchanged:
`329cce87466c14b8d97a5bc05d665ebb36080601ac6057617b65aa260d3ad920`.
TER now selects **2.22.1**; its wheel SHA-256 is
`af2a5c37985beaf9df11baf86791d14210d6d4f46bca3c89fe86ea34f8314b84`.
Setup rejects either file if its hash differs and checks the imported TER version
and isolated installation path. Earlier evaluations used TER 2.20.0; their saved
provenance remains unchanged. Rescore saved generations with
`python -m recipe.phimm.tools.inhouse_entity_rescore --summary <original-summary.json> --output-dir <new-results>`;
this runs EER only and never loads audio or a model.

## Dataset statistics

| Dataset | Recordings | Segments | Display words |
|---|---:|---:|---:|
| CustomerSpeechDomainSet_DTEST_Banking_Entity_FY23Q4_en-US_DTEST | 36 | 626 | 62,193 |
| CustomerSpeechDomainSet_DTEST_Medical_Entity_FY23Q4_en-US_DTEST | 69 | 1,193 | 105,140 |
| OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST | 5 | 734 | 46,007 |
| OnlineMeetings_CS_Product_FY22_en-US_DTEST | 5 | 596 | 40,231 |
| Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection | 10 | 740 | 50,953 |
| **Total** | **125** | **3,889** | **304,524** |

| Dataset | Segments / recording min / mean / max | Clip seconds min / mean / max | Words / recording min / mean / max |
|---|---:|---:|---:|
| CustomerSpeechDomainSet_DTEST_Banking_Entity_FY23Q4_en-US_DTEST | 10.00 / 17.39 / 25.00 | 0.61 / 32.22 / 39.99 | 911.00 / 1727.58 / 2838.00 |
| CustomerSpeechDomainSet_DTEST_Medical_Entity_FY23Q4_en-US_DTEST | 6.00 / 17.29 / 23.00 | 0.33 / 31.94 / 39.99 | 482.00 / 1523.77 / 2262.00 |
| OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST | 129.00 / 146.80 / 164.00 | 0.74 / 25.03 / 29.99 | 8241.00 / 9201.40 / 9930.00 |
| OnlineMeetings_CS_Product_FY22_en-US_DTEST | 73.00 / 119.20 / 153.00 | 0.26 / 24.65 / 29.99 | 5228.00 / 8046.20 / 10378.00 |
| Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection | 53.00 / 74.00 / 82.00 | 0.64 / 24.60 / 29.99 | 3391.00 / 5095.30 / 6385.00 |

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

Data sources: `banking`, `medical`, `shiproom`, `product`, `meetings`.

The shared display-style prompt omits an explicit formatting instruction;
`style: display` still selects the standard `READABLE` reference wrapper.

## Verified English-hint results (2026-11-20)

| Model | Five-dataset mean EER |
|---|---:|
| 2611 V1 / S2 SFT @38000 | 23.5544% |
| Prompt @800 / text h1 | 23.1199% |
| Prompt @800 / context OFF | 22.8121% |

All three runs passed 125-recording / 3,889-segment / 8,584-entity audits,
including the display-style English-hint base prompt on every segment.
See the [full comparison](../../../../../../../../../docs/experiments/inhouse_entity_longform.md) for per-dataset rates and durable result paths.

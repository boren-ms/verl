# OnlineMeetings biasing validation

Last updated: 2026-11-27

`meetings_before.yaml`, `meetings_after.yaml` and `meetings_none.yaml` each
select the same two datasets with separate source names. They contain only
dataset configuration. Prompts use `style: display` with an English language
hint, matching the existing in-house display references. The none YAML omits
`biasing`, which defaults to `None`.

| Source | Canonical dataset | Recording rows | Ordered audio segments | Candidates per row | Matched keywords per row |
|---|---|---:|---:|---:|---:|
| shiproom | OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST | 5 | 734 | 1,000 | 15–47 |
| product | OnlineMeetings_CS_Product_FY22_en-US_DTEST | 5 | 596 | 1,000 | 12–43 |

JSONL manifests are stored at
`az://orngwus2cresco/data/speech/users/boren/data/online_meetings_biasing/validation_v1/`
as `shiproom.jsonl` and `product.jsonl`. Local copies are under
`/home/boren/data/online_meetings_biasing/validation_v1/`.

Each row retains one complete recording's `DisplayTranscription`, tagged
`Transcription` as `entity_transcription`, and the existing ordered 30-second
`audio_path` selectors. The reference scope is the complete recording;
these are not independent clips with aligned clip references. No audio is
resegmented or copied. Existing normalized sources are
`az://orngwus2cresco/data/boren/data/Evaluation/dtest_enus_eer_v1/<canonical dataset>.jsonl`;
original recording references are checked by UUID and exact transcription.

The original MoE `test.jsonl` contains complete recordings and references, with
no segment-time fields. The time ranges used here come from the existing
`InhouseASR_2605_seg_presegment_30s` preprocessing output. The earlier pipeline
uses `long_audio_vad_explode` / `svad_explode` with Smart VAD and
`--max-len-sec 30`, storing selectors such as `wav#0.0:24.71` rather than
writing separate WAV files. Its implementation is available in the existing
`verl-mirror/recipe/phimm/data/run_long_audio_vad_explode_inhouse_2605_all_seg30.sh`
and `long_audio_vad_explode.py`. This biasing preparation reuses those saved
boundaries and does not run segmentation again. The original source is MoE;
the 30-second segmentation is an existing preprocessing product.

Candidates come from each canonical dataset's original
`name_list_II_1000_new.tsv` under
`az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US/`.
Normalized duplicate spellings are removed while preserving the first surface
form and original ordering. Both lists retain 1,000 entries; no synthetic
sampling or truncation is applied. `keywords` contains actual candidate hits
in the recording's display reference, using the shared biasing phrase matcher
with markup and timing labels excluded from matching. Keywords are recording
level; they are not claimed to occur in every segment.

JSONL contains no prompt. `add_task_info` inserts the saved candidates before
or after `<audio>`. The source names are `shiproom_before`, `product_before`,
`shiproom_after`, `product_after`, `shiproom_none` and `product_none`.
Recording and biasing metadata are retained in `extra_info`; `keywords` is
included automatically by `verl_format`.

Import the selected data YAML at `data.val_data` in an evaluation recipe and
add these source names to that recipe's `reward_map` with its intended scorer.
The none condition reuses the same recording manifests and ground-truth
keywords while leaving candidates out of the prompt. `preparation.json`
beside the manifests records exact inputs, candidate counts and SHA-256 hashes.

Verification compared both published JSONLs byte for byte and by SHA-256,
checked the WAV headers of all ten source recordings and all segment bounds,
and exercised the actual JSONL loader and transforms for all six source/mode
pairs. Prompt positions, candidate lists, keywords, complete references and
recording metadata were verified. No decoding or model-quality evaluation
is implied by these data checks. Receipts are `verification.json` and
`loader_verification.json` beside the manifests.

# EN-US inhouse evaluation: dtest_enus_eer and dtest_enus_ter

The EER presets evaluate only the five en-US EER datasets selected by
MoE's `scripts/audio_eval/result_orders/2611_eval_rows.tsv`. The adjacent
[data inventory](../../plugins/phimm/src/recipe/phimm/config/data/val_data/inhouse/eer.md)
records all five names, original data roots and counts: **125 recordings,
3,889 existing segments**. The previous 31-domain entity inventory is a
historical suite and is no longer selected by the evaluation presets.

## Data and scoring contract

`recipe.phimm.data.inhouse_dtest_enus` loads only the five selected source
manifests. It verifies every original UUID, complete ordered segment indices,
source recording/bounds and both full reference forms. It retains all source
clips and selectors. No new segmentation or reference reconstruction is used.
The normalized manifest root is
`az://orngwus2cresco/data/boren/data/Evaluation/dtest_enus_eer_v1`.

MoE `scripts/audio_eval/run_SI_local.py` selects **DisplayTranscription** for
EWER/EER and calls **SpeechInsight GetMetrics --nondisfluency -l en-us**.
The active EER config now selects that reference and mode, using a dedicated
SpeechInsight package directory. The old --verbatim/Transcription measurements
are not comparable to the workbook protocol.

Each recording is scored once after joining its ordered generated clips.
Per-dataset EER is summed entity errors divided by summed reference entities.
The primary workbook summary is the **equal-weight mean of the five dataset
EERs**. A count-weighted corpus EER is supplementary and labeled separately.
DTER remains a diagnostic. Strict EER rejects failed or malformed backend results.

## Prepare and evaluate

```bash
python -m recipe.phimm.data.inhouse_dtest_enus \
  --suite eer --cache-dir /tmp/workbook_eer --publish

python -m recipe.phimm.main_asr_eval \
  --config-name eval_enus_eer --cfg job --resolve

python plugins/phimm/submit.py --evaluate \
  --submission-id <unique-workbook-five-dataset-id> \
  --config-name eval_enus_eer
```

The SFT preset uses 2611 V1 / S2 step38000. The Prompt @800 presets inherit
the same five-dataset list and scorer and differ only in context enablement:
`eval_enus_eer_prompt_h1` and
`eval_enus_eer_prompt_off`.
Both use a merged model with no new LoRA adapter and no training resume.
Both EN-US dataset YAMLs use the shared `add_task_info` display task with
the required English language hint:

```text
Transcribe the audio clip into text.<audio>
The language is English.
```

All SFT and Prompt context ON/OFF evaluations use this same base prompt. The reporter's `--language-hint English` option
checks the saved base prompt for every decoded segment.
Text h1 adds the preceding generated transcript only. Existing pre-segment
boundaries and the requested context settings remain separate from the workbook's
historical decoding, so identical scores are not guaranteed.

After exact Ray SUCCEEDED and durable metrics/details:

```bash
python -m recipe.phimm.tools.inhouse_entity_report \
  --result-root az://.../<unique-workbook-five-dataset-id>/val_data_gen \
  --suite eer --language-hint English --context-mode independent --output-dir /tmp/workbook_eer_results
```

Use `--context-mode text_h1` for the context run. The report reads only the
five selected manifests, audits complete recording/clip/reference coverage,
checks exact previous-output context chains and saved metric agreement, and
compares the five-dataset mean with the user-supplied workbook S2 column.
For historical 31-subset output audits, specify `--suite legacy_entity_v3`.

## EN-US TER suite

The adjacent [TER inventory](../../plugins/phimm/src/recipe/phimm/config/data/val_data/inhouse/ter.md)
and YAML select MoE's five `disfluencytolerant_ter / default / Tier1 / en-US` rows.
Three meeting datasets overlap with EER; Conversation FY21Q1 and Dictation
OfficeOffline FY24Q3 replace Banking/Medical. No other locale or dataset is loaded.
Both suites retain existing pre-segments and original full DisplayTranscription.
TER-only evaluation sets `compute_eer: false`, so it never calls entity scoring.

```bash
python -m recipe.phimm.data.inhouse_dtest_enus \
  --suite ter --cache-dir /tmp/workbook_ter --publish
python plugins/phimm/submit.py --evaluate \
  --submission-id <unique-en-us-ter-id> \
  --config-name eval_enus_ter
python -m recipe.phimm.tools.inhouse_entity_report \
  --suite ter --language-hint English --result-root az://.../<unique-en-us-ter-id>/val_data_gen \
  --output-dir /tmp/workbook_ter_results
```

MoE's workbook metric named TER is `disfluencytolerant_ter`; the scorer uses
`dfmetrics.ter.TER(locale="en-us", ter_type="disfluencytolerant")` from
**ter 2.22.1**, selected by MoE's 2026-11-10 manifest. The pinned wheel is installed
into an isolated node-local target under the dedicated package directory.
GetMetrics uses MoE's TSV layout (`id`, `hypothesis`, `reference`) and explicit
column indices and locale. Counts determine fractional rates, including rates
below 1.5%; no magnitude-based percentage inference is used.

Preparation saves per-dataset counts, display-word totals and distributions of
segments per recording, words per recording and segment duration in `preparation.json`.
Word counts remove tags/times and exclude punctuation-only tokens. They are
inventory counts, not official TER reference-token counts. Existing segment rows
copy the complete recording reference, so aligned words per segment are unavailable.
The reporter checks all selected recordings and ordered clips, full references,
context history and persisted metric counts. Primary suite scores are equal-weight
means of the five dataset scores; corpus count-weighted rates are supplementary.

## Current five-dataset EER results WITH English language hint (2026-11-20)

All three Ray jobs succeeded and passed the full recording/segment/reference,
context, metric-count, and every-segment prompt audits. Each uses 125 recordings,
3,889 existing segments and 8,584 reference entities. Data-source names are unprefixed.
The shared `add_task_info` transform selects display style and English language hint;
its prompt is `Transcribe the audio clip into text.<audio>` followed by
`The language is English.` on the next line.

Source snapshot: `0cb93228ccd7a4c8dcf76089fc542b897fa79c1d`. Tools: SpeechInsight 2.28.4 / TER 2.20.0.

The later local config-filename and scorer-default cleanups preserve these
effective settings; each durable audit retains the config and source revision
actually used for inference/scoring.

| Dataset | Workbook 2611 S2 | SFT 2611 S2 @38000 | Prompt @800 text h1 | Prompt @800 OFF |
|---|---:|---:|---:|---:|
| banking | 7.62% | 7.5110% | 7.6206% | 7.7851% |
| medical | 6.51% | 6.6789% | 6.4451% | 6.7290% |
| shiproom | 46.77% | 47.8495% | 45.1613% | 44.6237% |
| product | 31.07% | 32.2034% | 33.3333% | 31.6384% |
| meetings | 23.77% | 23.5294% | 23.0392% | 23.2843% |
| **Five-dataset mean** | **23.15%** | **23.5544%** | **23.1199%** | **22.8121%** |

The workbook column is the user-supplied **2611-s2 / shadow run** model,
identified with 2611 V1 here. Its five rounded row values average to 23.148%.
The primary mean gives each dataset equal weight; the corpus aggregate below
instead weights each reference entity equally.

| Model | Entity errors / references | Corpus-weighted EER |
|---|---:|---:|
| 2611 V1 / S2 SFT @38000 | 779 / 8584 | 9.0750% |
| Prompt @800 / text h1 | 762 / 8584 | 8.8770% |
| Prompt @800 / context OFF | 779 / 8584 | 9.0750% |

All reference and audio selectors match the previously prepared pre-segmented
sources; no resegmentation or sampling was performed. Matching the dataset/scorer
protocol does not establish parity with the workbook's historical decoding backend.
TER data/tooling were validated; no full TER model evaluation was requested.

Durable reports, effective configs, logs and provenance:

- 2611 V1 / S2 SFT @38000: `az://orngwus2cresco/data/boren/outputs/v2611_eval/dtest-enus-eer-langhint-2611s2-s38000-20261120-names/evaluation_audit`
- Prompt @800 / text h1: `az://orngwus2cresco/data/boren/outputs/v2611_eval/dtest-enus-eer-langhint-prompt-s800-text-h1-20261120-names/evaluation_audit`
- Prompt @800 / context OFF: `az://orngwus2cresco/data/boren/outputs/v2611_eval/dtest-enus-eer-langhint-prompt-s800-no-context-20261120-names/evaluation_audit`

The exact owned two-node Brix pool was verified paused after all three runs
and durable result collection completed. Local audit copies and comparison are
under `tmp/inhouse_eer_workbook_langhint/`.

## Historical five-dataset EER results WITHOUT language hint (2026-11-20)

These completed runs omitted the required English language hint. They are
retained as historical measurements and are superseded by the language-hint
reruns. Their prompts/configs/artifacts remain unchanged.

All three Ray jobs reached `SUCCEEDED`. Saved-detail audits passed for all
125 recordings and 3,889 existing segments in every run, with unchanged full
references, verified context histories and saved metric agreement. All runs
use the same source snapshot, `1ebbe4787a501352a636dd2bb6656e5c924eee35`.

| Dataset | Workbook 2611 S2 | SFT 2611 S2 @38000 | Prompt @800 text h1 | Prompt @800 OFF |
|---|---:|---:|---:|---:|
| Banking FY23Q4 | 7.62% | 8.2785% | 7.6754% | 7.8399% |
| Medical FY23Q4 | 6.51% | 7.1965% | 6.6455% | 6.9795% |
| OnlineMeetings Shiproom FY22 | 46.77% | 50.0000% | 46.2366% | 44.6237% |
| OnlineMeetings Product FY22 | 31.07% | 31.6384% | 32.7684% | 32.7684% |
| Conversation OnlineMeetings FY25Q3 | 23.77% | 24.0196% | 23.0392% | 23.0392% |
| **Five-dataset mean** | **23.15%** | **24.2266%** | **23.2730%** | **23.0501%** |

The workbook mean is its displayed 23.15% (23.148% from the five rounded
dataset values). The primary metric weights datasets equally. Corpus EER
uses summed counts and is supplementary:

| Run | Entity errors / entities | Corpus EER |
|---|---:|---:|
| sft | 829 / 8584 | 9.6575% |
| text_h1 | 776 / 8584 | 9.0401% |
| no_context | 796 / 8584 | 9.2731% |

Context OFF has the lower primary mean by 0.2229 percentage points. Text h1
has 20 fewer pooled entity errors: Banking improves by 3 and Medical by 20,
while Shiproom regresses by 3. Shiproom contains only 186 entities but receives
the same dataset weight as Medical's 5,989 entities, which explains why the
macro and corpus rankings differ.

Each run uses the matching MoE dataset list, default prompt and official
scoring mode. Existing pre-segment boundaries, context settings and this
inference implementation remain explicit; exact reproduction of all historical
workbook generation settings is not established. TER tooling/data were validated
but no full TER model evaluation was requested in this update.

Durable audit packages (metrics, reports, config, provenance and Ray log):

- 2611 V1 / S2 SFT @38000: `az://orngwus2cresco/data/boren/outputs/v2611_eval/inhouse-eer-workbook5-2611s2-s38000-20261120/evaluation_audit`
- Prompt @800 / text h1: `az://orngwus2cresco/data/boren/outputs/v2611_eval/inhouse-eer-workbook5-prompt-s800-text-h1-20261120/evaluation_audit`
- Prompt @800 / context OFF: `az://orngwus2cresco/data/boren/outputs/v2611_eval/inhouse-eer-workbook5-prompt-s800-no-context-20261120/evaluation_audit`

Each run root also retains `val_data_gen/0.metrics.json` and the complete
per-dataset generation details. Local comparison artifacts are under
`tmp/inhouse_eer_workbook/`. The owned two-worker Brix pool was verified paused
after all three audits and durable result collection.

Validation: **47 targeted tests passed**, Ruff and diff checks passed, all four
effective presets were composed, both complete dataset YAMLs were loaded, all
270 unique audio paths were accessible, and both workers passed native EER/TER
smoke checks using SpeechInsight 2.28.4 and ter 2.20.0.

## Historical 31-subset result (different protocol, 2026-11-20)

The following result used Transcription/--verbatim over 31 subsets; it is
retained for provenance and is superseded for workbook comparison.


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

The earlier evaluations used MoE `speechinsight_tools_latest`, version
**2.28.4**, mirrored unchanged (492 files, 771,543,885 bytes) at
`az://orngwus2cresco/data/speech/users/boren/eval_packages/inhouse_eer_workbook_20261120/speechinsight_tools_latest/`.
Both workers passed matched/mismatched entity smoke checks. The GetMetrics
SHA-256 is `329cce87466c14b8d97a5bc05d665ebb36080601ac6057617b65aa260d3ad920`.


The active data config names are `inhouse/eer` and `inhouse/ter`.
Their adjacent Markdown inventories describe the exact five datasets in each.
The current scorer supports only MoE's pinned **2026-11-10** tool bundle:
SpeechInsight **2.28.4** and TER **2.22.1**. The manifest is shipped as
`recipe/phimm/reward/speechinsight_tools_manifest.json`; setup validates both
GetMetrics and the TER wheel by SHA-256, dependency metadata, and the imported
TER version/location. There is no legacy installer or version selector.
The default cache is `/root/data/packages/SpeechInsight_moe_20261110_ter_2_22_1`,
and the mirror root is
`az://orngwus2cresco/data/speech/users/boren/eval_packages/inhouse_moe_20261110`.
Previously completed evaluations keep their original TER 2.20.0 provenance;
rescoring writes separate results and does not decode again.


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

The shared display-style prompt omits an explicit formatting instruction;
`style: display` still selects the standard `READABLE` reference wrapper.


`eval_score` defaults `pack_dir` to `DEFAULT_PACK_DIR` and
`speechinsight_package_root` to `DEFAULT_REMOTE_PACK_ROOT` in the function
signature. Inhouse reward configs omit those redundant values and the removed
`version` override. The shared response parser supplies its default format.

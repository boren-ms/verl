# OpenASR multilingual expansion — three-model evaluation

Evidence collected through 2026-11-07T05:43:20.948672+00:00.

All values below are percentages, computed as total errors / total reference units.
Mandarin uses character units; Vietnamese, French, and CORAAL use words.
All three models used two nodes with eight H100 GPUs each, greedy decoding, identical
prompts and scorer routes. Every Ray job succeeded; final logged metrics match
the independently summed counts from all 14 detail artifacts.

| Dataset | Metric | Recordings | 2609r2 | 2609r2b | 2611r2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Mandarin FLEURS | CER | 945 | 2.76 | 2.12 | 2.90 |
| Mandarin AISHELL-1 | CER | 7176 | 1.72 | 1.72 | 8.39 |
| Vietnamese FLEURS | WER | 857 | 4.60 | 4.84 | 4.14 |
| Vietnamese Common Voice | WER | 1367 | 7.51 | 7.35 | 8.16 |
| French ESLO telephony | WER | 36 | 24.61 | 20.98 | 28.27 |
| French Accueil UBS | WER | 41 | 28.11 | 25.36 | 28.88 |
| CORAAL, eight locations pooled | WER | 271 | 14.72 | 15.43 | 13.40 |

The six new rows stay visible but are excluded from established headline
averages. CORAAL is pooled by summing errors and reference words across all
locations, retaining one CORAAL component. This run does not reevaluate the
remaining established datasets or calculate a new full-suite headline average.

## CORAAL location results

| Location | Recordings | Segments | 2609r2 WER | 2609r2b WER | 2611r2 WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATL | 14 | 1199 | 16.84 | 17.04 | 15.60 |
| DCA | 74 | 4584 | 15.90 | 16.68 | 15.06 |
| DCB | 63 | 6260 | 11.22 | 11.07 | 10.64 |
| DTA | 40 | 3383 | 17.66 | 19.83 | 15.51 |
| LES | 15 | 1194 | 19.04 | 21.41 | 15.76 |
| PRV | 32 | 2030 | 22.77 | 24.41 | 20.28 |
| ROC | 19 | 1863 | 9.62 | 9.86 | 7.56 |
| VLD | 14 | 1528 | 11.17 | 10.85 | 10.93 |

## Exact model and job identities

| Label | Checkpoint | Ray submission | W&B |
| --- | --- | --- | --- |
| 2609r2 | FastLLM 2609 r2 step 54,000 | `raysubmit_rdJfADWZV3VE23Jx` | [run](https://msaip.wandb.io/genai/v2611_eval/runs/04a7a127012e0817bb4819571d83fa4e) |
| 2609r2b | ReMax r2b global_step_400 | `raysubmit_LEaHnjhiMuEsf9L3` | [run](https://msaip.wandb.io/genai/v2611_eval/runs/771114fb86161dda05bf34a9dde3a003) |
| 2611r2 | FastLLM 2611 shadow step 38,000 | `raysubmit_qx5V9RNLKjRQ1jda` | [run](https://msaip.wandb.io/genai/v2611_eval/runs/965449f251843c3cc6869c17f451e9ff) |

**2609r2 model export:** `az://orngwus2cresco/data/speech/projects/phi-fastllm-2609/amlt-results/fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.4-r2/54000/qwen_hf`


**2609r2b model export:** `az://orngwus2cresco/data/boren/outputs/v2609_asr/remax_2609r2_earning_ml0_ls_bad_verb_hint_e3_bs128_n8_r256_g16_ds1c_sfl_avg6_flr_rm0b_v0/global_step_400/qwen_hf`


**2611r2 model export:** `az://orngwus2cresco/data/speech/projects/phi-fastllm-2611/amlt-results/fast-llm-2611-qwen9b-flashenc-mtp-s2-data-v3-8-2-1-shadow/38000/qwen_hf`

## Reproducibility and limits

Source MOE commit: `18143a6e4d3393a8ef90761bb4b4fba7c379d1ea`.
Evaluation code revision: `db873221b3a7198821680e87be9739d0d814a2c3`.
Pinned Hugging Face revisions, prepared artifact hashes, complete CORAAL manifest
hashes, error counts, reference counts, result paths, and launch entrypoints are
retained in `comparison.json` and the per-model summary JSONs.

Mandarin uses OpenCC traditional-to-simplified conversion and WeText number
normalization before grapheme CER. Vietnamese and French use the imported MOE
normalizers with preserved diacritics and compound-aware WER. French and CORAAL
full recordings use the existing verl 30-second segmentation and grouped scoring.
These results use the verl decoder and segmentation runtime; they do not reproduce
the full MOE runtime. Checkpoint steps and training lineages differ, so the table
is a descriptive evaluation comparison rather than a controlled training ablation.

## Durable detail artifacts

- **2609r2:** `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2_multilingual_expansion_18143a6e_n2/evaluation_summary.json`; detail files at `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2_multilingual_expansion_18143a6e_n2/val_data_gen/<data_source>/0.jsonl`.
- **2609r2b:** `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2b_multilingual_expansion_18143a6e_n2/evaluation_summary.json`; detail files at `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2609r2b_multilingual_expansion_18143a6e_n2/val_data_gen/<data_source>/0.jsonl`.
- **2611r2:** `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2611r2_multilingual_expansion_18143a6e_n2/evaluation_summary.json`; detail files at `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2611r2_multilingual_expansion_18143a6e_n2/val_data_gen/<data_source>/0.jsonl`.

## Runtime recovery

2611r2 was retried after a pre-evaluation initialization failure in which vLLM
selected TileLang’s incomplete CUDA runtime stub. The successful attempt explicitly
pins VLLM_CUDART_SO_PATH through the existing verl hook to the verified real CUDA
runtime on both nodes. Model, data, decoding, scorer, and two-node topology are
unchanged. The failed attempt produced no completed dataset results; failure IDs
and runtime library hashes are retained in comparison.json.

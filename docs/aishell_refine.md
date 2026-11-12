# AISHELL-1 refinement with 2611r2

This recipe uses original AISHELL-1 human transcripts. Decode the full training
split with 2611r2, rank by **absolute Mandarin CER edit count `n_err`**, and keep
20,000 utterances. Independently keep the 200 highest-error utterances from the
previous 2611r2 test run. Ties use original `source_row`, then `id`, ascending.
No WER/CER threshold is applied; hypotheses are audit fields, never targets.

## Sources and artifacts

- External training root:
  `az://highperf01eus2/data/am_data/publicdataset_process/AIShell-1/train_data/feature_extraction/`
- Orange audio/transcription root:
  `az://orngwus2cresco/data/speech/am_data/publicdataset_process/AIShell-1/train_data/feature_extraction/`
- Artifact root:
  `az://orngwus2cresco/data/boren/data/aishell1_refine_2611r2_20261110/`
- Model:
  `az://orngwus2cresco/data/speech/projects/phi-fastllm-2611/amlt-results/fast-llm-2611-qwen9b-flashenc-mtp-s2-data-v3-8-2-1-shadow/38000/qwen_hf/`
- Previous test detail:
  `az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_2611r2_multilingual_expansion_18143a6e_n2/val_data_gen/zh_aishell1/0.jsonl`
- Previous test detail SHA-256:
  `81865255a717c8e759a9b96ebe93cb0a0ac273c2df2e29dc458a0609025c677e`

Training contains 120,098 unique utterances, 340 speakers and 150.852 source
hours in 301 chunks. Only the 301 `.audio` and 301 `.transcription` files were
copied through `orngtransfer`: 17,391,507,959 bytes. Every destination size and
available source MD5 was verified. Small `.info` sidecars were read locally to
recover IDs; `.info` and `.feature` files are not needed on Orange.

The full manifest retains the original source chunk transcription as its
`text`, and verifies it against `train_filenames.txt` after whitespace removal.
All training IDs are disjoint from all 7,176 test IDs. Source metadata hashes
are pinned in `scripts/prepare_aishell_refine.py`.

| Artifact | Purpose |
| --- | --- |
| `train_full.jsonl` | All original training references and Orange chunk selectors |
| `generation/part-*.jsonl` | Full greedy 2611r2 outputs with canonical CER counts |
| `train_top20000.jsonl` | Top 20,000 training rows, published after complete generation |
| `validation_top200.jsonl` | Fixed Top 200 test rows, reusing existing Orange ZIP offsets |
| `*.audit.jsonl` | Selected IDs, original references, scored hypotheses and edit counts |
| `*.meta.json` | Counts, hashes, selection rule and source/model provenance |

The fixed validation subset has selection errors 7–101 and was re-scored with
the canonical Mandarin scorer. It is a **hard-case diagnostic subset selected
on the baseline model**, not an unbiased replacement for full AISHELL-1 test CER.

## Reproduce preparation and selection

Use the repository's `transfer-to-orange` skill for verified two-stage copying,
restricting the recursive include pattern to `*.audio;*.transcription`. Never
put source SAS credentials in a YAML, report, command log or checked-in file.

Download the source `chunk_sanity_check/file_set.json`, training-root
`train_filenames.txt`, and pinned previous test detail to local scratch first.
Then:

```bash
python scripts/prepare_aishell_refine.py prepare \
  --file-set /scratch/aishell/file_set.json \
  --filenames /scratch/aishell/train_filenames.txt \
  --test-details /scratch/aishell/2611r2_zh_aishell1.jsonl \
  --cache /scratch/aishell/chunk_metadata \
  --output /scratch/aishell/artifacts/train_full.jsonl

python scripts/prepare_aishell_refine.py select --split validation \
  --inputs /scratch/aishell/2611r2_zh_aishell1.jsonl \
  --output /scratch/aishell/artifacts/validation_top200.jsonl --verify-scorer
```

Publish these JSONLs and metadata to the artifact root with `azcopy --put-md5`
and verify uploaded sizes/checksums. Three Orange audio chunk range reads were
also decoded as nonempty 16-kHz audio during initial verification.

## Full generation and final filtering

`recipe/phimm/config/v2611_dev/gen/gen_aishell1_train_2611r2.yaml` uses
2 × 8 GPUs, no LoRA, BF16, no quantization, temperature zero, language-hinted
verbatim prompts and all 120,098 rows in source order. The scorer is
`openasr_expansion_eval(version=2609)`: Chinese normalization followed by
grapheme CER, matching the prior test evaluation. Output parts are resumable.

On verified available two-node capacity, sync this worktree and run:

```bash
bash quick_run.sh recipe/phimm/config/v2611_dev/gen/gen_aishell1_train_2611r2.yaml
```

The launch must propagate the verified real CUDA runtime as
`VLLM_CUDART_SO_PATH` to Ray workers if the environment exposes a TileLang stub
runtime, and limit compilation threads as needed. Record the exact runtime
overrides and Ray submission ID in the task state ledger.

Mandarin scoring requires `opencc-python-reimplemented==0.1.7` and
`wetext==0.2.1`. They are pinned in both dependency lists and checked by the
installer's cached-environment validation. Before generation, exercise the
actual lazy-loaded scorer on each node, including re-scoring the fixed 200
previous test outputs; a model/audio loader check alone cannot verify it.

After the exact Ray submission succeeds and all outputs are durable, collect
all contiguous parts, then:

```bash
python scripts/prepare_aishell_refine.py select --split train \
  --inputs /scratch/aishell/generation/part-*.jsonl \
  --full-manifest /scratch/aishell/artifacts/train_full.jsonl \
  --output /scratch/aishell/artifacts/train_top20000.jsonl --verify-scorer
```

Selection rejects incomplete/duplicate results, missing parts, mismatched
IDs/references/chunk selectors/order metadata, invalid edit counts and missing
measured durations. Re-scoring runs before the output is written. Publish the
final manifest, audit and metadata only after these checks pass.

## Training config

`recipe/phimm/config/v2611_dev/aishell/remax_aishell_top20k.yaml`
has exactly one training source and one validation source, clears the previous
six-source `data.mix`, and enables training shuffle. Training execution has not
been requested; creating this configuration does not submit a training job.

The config inherits its own `aishell/base_aishell.yaml`, which reuses the shared
`v2611_dev/base.yaml` runtime and has no dependency on the `main/` configs.
The AISHELL base pins the original Base1a training settings without importing
its unrelated validation cohorts or training reward routes. Composed algorithm,
model, optimizer, rollout, active data and reward settings were checked against
the previous AISHELL config: ReMax, LoRA rank 320, learning rate 6e-7, rollout
`n=8`, batch 128, two epochs and two nodes, with the exact same 2611r2 model path.

The `aishell/remax_aishell_top20k_2611v1a.yaml` variant uses the user-specified
`az://orngwus2cresco/data/boren/outputs/v2611_eval/main2611_r2_finalbest_20261109/exports/main_base1a_2611r2_s900/qwen_hf`
checkpoint and experiment name `remax_aishell_top20k_2611v1a`. It keeps the
same fixed 2611r2-selected Top20k training and Top200 validation data,
reward, and training settings. Tokenizer and output paths follow the new
model and experiment name. Creating the variant does not launch training.

Training retains ASR-vote delta mechanics and the original character,
language and format terms, dropping word-level terms. AISHELL references have
segmentation spaces: the inherited word scorer counts two word errors for
`你好世界` against `你好 世界` although character recognition is perfect.
The new reward reduces over one primary character measure (`total: 1`) and
retains language/format beta 0.05 each. Training keeps the baseline character
normalizer; selection and diagnostic validation use the separate canonical
Mandarin CER scorer. No training has been launched.

Local task evidence and live state are under ignored
`tmp/aishell_refine/` and `tmp/run_openasr_exp/aishell_refine/state.json` in this
worktree. Compact verification reports are in `reports/aishell_refine_20261110/`.

## Completed data preparation (2026-11-10 UTC)

Generation `raysubmit_sQrFQY2LJMvuFU4Z` succeeded on 2 × 8 H100 GPUs using
code snapshot `983ec4cbf0545947a721656502c4f7ab24171099`. All 120,098 unique
utterances were saved in six contiguous parts. Aggregate Mandarin CER was
10.3023% (178,242 / 1,730,113). The console's generic `Overall wer` label refers
to Mandarin CER for this scorer.

| Manifest | Rows | Detail | SHA-256 |
| --- | ---: | --- | --- |
| `train_full.jsonl` | 120,098 | 150.852 measured hours | `ffeafcc13ec27b68341390a34ad482e63afad08e593710d372a93a4fa692cd52` |
| `train_top20000.jsonl` | 20,000 | 26.4474 hours; n_err 1–1,153; no zero-error rows | `a1295ba2a425d2ccd382b982fa2e39bc56330489e8f016e8afa4ba3a94ef0581` |
| `validation_top200.jsonl` | 200 | Original 2611r2 test outputs; n_err 7–101 | `753b4e38aefa202c564b5b0aff0e18258830bc2c5824db41a27752062f7fabb6` |

Every selected training and validation row was re-scored successfully. Two
selected training responses reached the 768-token generation limit; they were
retained under the requested absolute-error ranking. No CER-ratio filtering
was applied. The final train/validation ID overlap is zero.

All eight published manifest/audit/metadata files passed Orange size and MD5
verification. The actual configured dataset loader loaded 20,000 training and
200 validation rows, checked original reference targets and audio selectors,
and verified audio/token alignment including the longest training sample
(14.53125 seconds, 182 audio tokens). The training config is ready; no training
job was submitted. Final job lineage, publication and resource state are saved
in the report summary and the artifact root's `provenance/` directory.

## One-epoch ranked subset variants

The four `aishell/remax_aishell_top{1,2,5,10}k_2611v1a_e1.yaml` recipes use the
same actual 2611r2 Base1a step900 HF seed as the Top20k `2611v1a` variant.
Each corresponding training-data YAML reads the verified `train_top20000.jsonl`
and applies `post_process.input_egs_limit` before training shuffle. Thus the
subsets are nested prefixes of the fixed descending-absolute-error ranking,
with original human transcripts retained. Each subset has a separate cache
name and data-source reward route to avoid reusing the full Top20k cache.
No new manifests or audio copies are needed.

| Recipe suffix | Selected rows | Epochs | Expected optimizer steps | Dropped final-batch rows |
| --- | ---: | ---: | ---: | ---: |
| `top1k_2611v1a_e1` | 1,000 | 1 | 7 | 104 |
| `top2k_2611v1a_e1` | 2,000 | 1 | 15 | 80 |
| `top5k_2611v1a_e1` | 5,000 | 1 | 39 | 8 |
| `top10k_2611v1a_e1` | 10,000 | 1 | 78 | 16 |

`trainer.total_epochs=1` and `trainer.total_training_steps=null` derive the
budget from the runtime dataloader. The expected counts above follow the
inherited batch size128 and `drop_last=True`; confirm the runtime target
before accepting training progress. All other model/LoRA, optimizer, rollout,
character/language/format reward, and fixed Top200 diagnostic validation
settings match the Top20k recipe. Each recipe uses two nodes with eight GPUs
per node and a distinct experiment/output name.

Validation: all four Hydra configs compose and resolve; their model/actor,
reward definitions and validation data match the Top20k parent settings. The
source manifest SHA-256 matches the published digest above; all20,000 ranks
are consecutive and edit counts descend. The actual loader's `limit_ds`
function preserves each requested prefix's IDs and human references.

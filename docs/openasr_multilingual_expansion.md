# OpenASR multilingual expansion

Source: MOE merge `18143a6e4d3393a8ef90761bb4b4fba7c379d1ea`
(PR 45660, Expand ASR evaluation diagnostics and OpenASR coverage).
The checked-in TSV records repository, revision, test parquet, expected examples,
license, metric, and averaging policy for each new snapshot.

| Dataset | Language | Metric | Test recordings | Headline average |
| --- | --- | --- | ---: | --- |
| FLEURS | Mandarin | CER | 945 | excluded |
| AISHELL-1 | Mandarin | CER | 7,176 | excluded |
| FLEURS | Vietnamese | WER | 857 | excluded |
| Common Voice | Vietnamese | WER | 1,367 | excluded |
| ESLO telephony | French | WER | 36 | excluded |
| Accueil UBS | French | WER | 41 | excluded |

New short-audio rows are appended to the OpenASR-ML plain/verbatim/language-hint
manifests. The grouped ZIP language-hint manifest also includes the two French
full-recording datasets. Existing rows, scorer routes, and canonical workbook
headline row lists retain their previous behavior.

Mandarin uses MOE's OpenCC traditional-to-simplified and WeText number
normalization, then Unicode grapheme CER without spaces or compound merging.
Vietnamese preserves diacritics, harmonizes tone placement, and normalizes
decimal, thousands, and percentage forms before compound-aware WER.
These new normalizers are isolated from the established OpenASR scorer.
Install `opencc-python-reimplemented` and `wetext` on the evaluator in addition
to the existing `regex`, `num2words`, and `kaldialign` dependencies.

French full recordings use the repository's 30-second Smart VAD. Every segment
retains its complete parent reference and grouping metadata; validation uses
`val_batch_size: -1` to score each complete recording after concatenating its
segment hypotheses. This is a verl evaluation, with its existing decoder and
SVAD implementation; it is not a reproduction of MOE's entire runtime.

The separate CORAAL manifest exposes ATL, DCA, DCB, DTA, LES, PRV, ROC, and VLD
from the existing ZIP-backed segmented snapshot. Compute aggregate CORAAL WER
as `sum(n_err) / sum(n_ref)` over locations, then retain CORAAL as one component
of the established long-form headline average. Never average eight location
WERs as eight new headline components.

Prepare datasets without replacing existing outputs:

```bash
python -m scripts.prepare_openasr_multilingual_expansion \
  --output-root az://orngwus2cresco/data/boren/data/openasr_ml_expansion/18143a6e \
  --local-root tmp/run_openasr_exp/multilingual_expansion_20261106/datasets
```

The preparer verifies pinned test row counts, writes WAVs into ZIP_STORED
archives, and publishes JSONL only after each archive upload finishes. It saves
source-parquet, archive, and manifest SHA-256 values alongside provenance.
Use `--prepare-only` to separate conversion from transfer. To resume publication
of complete local artifacts, pass `--publish-only`; matching existing remote
artifacts are reused after size and MD5 checks. On machines with `boostedblob`
installed, `--uploader bbb` uses sixteen concurrent 1 MiB blocks to avoid large
request resets on the local uplink and verifies every published file's MD5.

The three `v2611_dev/eval/eval_*_multilingual_expansion.yaml` configs select:

- **2609r2:** fast-llm 2609 data-v3.5.4-r2, step 54,000.
- **2609r2b:** ReMax ds1c/flr/rm0b-v0, step 400, matching the established r2b export.
- **2611r2:** fast-llm 2611 data-v3-8-2-1-shadow, step 38,000.

Each config evaluates only the six supplementary rows and eight CORAAL
locations, with 2 nodes × 8 GPUs, merged-model LoRA disabled, and training resume
disabled. Saved effective configs and job state belong under
`tmp/run_openasr_exp/multilingual_expansion_20261106/`. Dataset metrics must retain
their CER/WER labels and individual model/checkpoint identities.

Completed three-model results, the Excel workbook, and regeneration instructions
are checked in under
[`reports/openasr_multilingual_expansion_20261107`](../reports/openasr_multilingual_expansion_20261107/README.md).
The workbook opens on the supplementary comparison, preserving the preceding
canonical metrics and headline averages.

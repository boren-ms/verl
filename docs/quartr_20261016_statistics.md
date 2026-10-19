# Quartr 20261016 dataset statistics

Analysis date: 2026-10-19 UTC.

Source:

```text
az://orngwus2cresco/data/speech/users/xiaoych/quartr_earliest2_reviewed_bitbank_20261016/data_zip.jsonl.gz
```

SHA-256 of the compressed source, verified against the remote blob:

```text
bc7cab886bd50f1b41d47fc172e75fa2312de5613df0133c308e50b6a79a1f28
```

The complete dataset contains **19,728 unique records**, **213 events**, and
**182.39 hours** of audio. It reuses 32 native BitBank ZIPs through direct
`archive.zip!offset:size` references.

## Metric definitions and methodology

- **WER:** the stored `wer_maiears` ratio, multiplied by 100 for percentage
  displays. Every record was checked against a fresh word-level Levenshtein
  calculation on `wer_normalized_ref` and `wer_normalized_hyp`; all matched.
- **Word error count:** word substitutions + deletions + insertions. Counts
  use one optimal RapidFuzz alignment; equally optimal alignments can assign
  individual edit types differently without changing the total error count.
- **Normalized reference word count:** whitespace-split `wer_normalized_ref`.
  This is the denominator used for nonempty-reference WER.
- **Original transcript word count:** whitespace-split `text`, before WER
  normalization.
- **Duration:** `end - start`, in seconds. These are original parent-event
  timestamps, not durations from a fresh decode of every audio record.
- **Percentiles:** equal weight per record, using NumPy's `linear` method
  (Hyndman-Fan type 7). p0 is the minimum, p50 the median, p90 the 90th
  percentile, and p99 the 99th percentile. Interpolation can produce fractional
  percentiles for integer-valued counts. Tables are rounded to two decimals.
- **Mean WER:** arithmetic mean of per-record WER values, not corpus WER.
- **Corpus WER:** total word errors divided by total normalized reference words.
- **Empty normalized references:** the stored convention uses the hypothesis
  word count as WER when the reference is empty, or zero when both are empty.
  All 72 empty-reference records here also have empty normalized hypotheses.

Percentiles and means were independently cross-checked using sorted-value
interpolation and Python standard-library arithmetic. Record counts, error
totals, and saved JSON/CSV artifacts were also checked for consistency.

## Complete dataset: no WER filtering

| Metric | p0 / minimum | p50 / median | p90 | p99 | Mean |
|---|---:|---:|---:|---:|---:|
| WER (%) | 0.00% | 5.00% | 14.52% | 22.38% | 6.18% |
| Word error count | 0.00 | 4.00 | 13.00 | 22.00 | 5.29 |
| Normalized reference word count | 0.00 | 88.00 | 136.00 | 150.00 | 78.59 |
| Original transcript word count | 1.00 | 86.00 | 133.00 | 145.00 | 76.41 |
| Duration (seconds) | 0.10 | 34.90 | 61.34 | 80.13 | 33.28 |

### Totals and extremes

| Measure | Value |
|---|---:|
| Records | 19,728 |
| Events | 213 |
| Timestamp-based duration | 656,592.95 seconds / 182.39 hours |
| Normalized reference words | 1,550,396 |
| Original transcript words | 1,507,363 |
| Word errors | 104,415 |
| Mean per-record WER | 6.18% |
| Corpus WER | 6.73% |
| Zero-WER records | 4,381 / 22.21% |
| Maximum WER | 24.82% |
| Maximum error count | 34 |
| Maximum normalized reference word count | 181 |
| Maximum original transcript word count | 177 |
| Longest segment | 89.70 seconds |

The maximum-WER record is `event_35304_p0000.flac`: 34 errors over 137
normalized reference words, or 24.82% WER, with duration 79.44 seconds.
The longest record is `event_351131_p0076.flac`, at 89.70 seconds.

### Error composition

| Error type | Count | Share of errors |
|---|---:|---:|
| Insertions | 73,503 | 70.40% |
| Substitutions | 23,000 | 22.03% |
| Deletions | 7,912 | 7.58% |
| Total | 104,415 | 100.00% |

Insertions dominate the errors. Percentages may not sum to exactly 100% after
rounding. The mean per-record WER is 6.18%, while corpus WER is 6.73%; the latter
weights errors against the total reference-word denominator rather than giving
every record equal weight.

### WER bands

| WER band | Records | Share of records |
|---|---:|---:|
| Exactly 0% | 4,381 | 22.21% |
| Greater than 0%, at most 5% | 5,512 | 27.94% |
| Greater than 5%, at most 10% | 5,336 | 27.05% |
| Greater than 10%, at most 20% | 3,979 | 20.17% |
| Greater than 20% | 520 | 2.64% |
| Total | 19,728 | 100.00% |

No record has WER above 100%; the observed maximum is 24.82%.

### Normalization caveat

There are **72 records** whose normalized reference and hypothesis are both
empty. Their original reference texts include fillers such as `Um.`, `Mm-hmm.`,
and `Mm.`, or the label `[Foreign language]`. These receive zero normalized
WER even when the original reference and hypothesis differ.

Thus, a zero WER is not proof that the original text strings match. Excluding
empty-normalized-reference records changes mean per-record WER from 6.18% to
6.21%.

### Timestamp versus decoded duration

The manifest reports **656,592.9340625 decoded seconds**. The timestamp-based
sum is **656,592.95 seconds**, a difference of **0.0159375 seconds**. This is
fully explained by the three retained original timestamp mismatches:

| Record | Decoded frames minus timestamp-derived frames |
|---|---:|
| `event_131862_p0026.flac` | -63 |
| `event_349560_p0139.flac` | -32 |
| `event_349556_p0070.flac` | -160 |

All records have sample rate 16,000 Hz; the combined 255-frame discrepancy
equals 0.0159375 seconds.

## Training subset: WER at most 5%

The [WER-filtered training YAML](../recipe/phimm/config/data/train_data/quartr_20261016_wer05.yaml)
keeps `0.0 <= wer_maiears <= 0.05`. It retains **9,893 records (50.15%)**
from all **213 events**, totaling **82.02 hours (44.97% of the original audio)**.
All **81 records at exactly 5% WER** are retained.

| Metric | p0 / minimum | p50 / median | p90 | p99 | Mean |
|---|---:|---:|---:|---:|---:|
| WER (%) | 0.00% | 1.17% | 4.26% | 4.96% | 1.61% |
| Word error count | 0.00 | 1.00 | 5.00 | 6.00 | 1.64 |
| Normalized reference word count | 0.00 | 68.00 | 140.00 | 151.00 | 70.72 |
| Original transcript word count | 1.00 | 66.00 | 136.00 | 146.00 | 68.82 |
| Duration (seconds) | 0.10 | 26.52 | 62.34 | 79.90 | 29.85 |

Subset totals:

- Normalized reference words: **699,604**.
- Original transcript words: **680,875**.
- Word errors: **16,263**.
- Corpus WER: **2.32%**, versus 1.61% mean per-record WER.
- Longest segment: **89.38 seconds**.
- The 72 empty-normalized-reference records remain in this subset.

This filtering reduces mean WER but does not remove the long-duration tail:
the subset's p90 duration remains 62.34 seconds.

## Training risk: inherited 40-second audio cap

At analysis time, the
[earnings-only recipe](../recipe/phimm/config/v2611_dev/earning/remax_earning_base0a_r0.yaml)
at commit `015c2f2efe` inherited `data.max_audio_dur: 40` from the
[shared base](../recipe/phimm/config/v2611_dev/base.yaml).
The [audio loader](../recipe/phimm/utils/audio.py) truncates audio beyond this
cap, while retaining the full reference text.

| Dataset | Records longer than 40 seconds | Share |
|---|---:|---:|
| Complete dataset | 8,734 | 44.27% |
| WER at most 5% subset | 4,000 | 40.43% |

Consequently, a substantial fraction of training examples can have cropped
audio paired with a full-segment transcript. Before training, either raise the
duration cap to accommodate complete segments, exclude long records, or
resegment audio and references together. Raising the cap also requires checking
model token limits and GPU memory capacity.

This analysis did **not** change the training configuration, source JSONL, or
audio ZIPs.

## Related configurations and artifacts

- [Unfiltered training YAML](../recipe/phimm/config/data/train_data/quartr_20261016.yaml).
- [WER-filtered training YAML](../recipe/phimm/config/data/train_data/quartr_20261016_wer05.yaml).
- [Earnings-only recipe](../recipe/phimm/config/v2611_dev/earning/remax_earning_base0a_r0.yaml).
- [ZIP/BitBank loading documentation](audio_zip.md).
- Detailed machine-readable results were saved locally as
  `dataset_statistics.json` and `record_statistics.csv` alongside the cached
  dataset. These local artifacts are not tracked in this repository.

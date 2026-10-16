# OpenLong segmented ZIP dataset statistics

Audit completed: 2026-10-16 18:56:01 UTC.

Configuration: [openlong_seg_verb_langhint_zip.yaml](../recipe/phimm/config/data/val_data/openlong_seg_verb_langhint_zip.yaml).

The audit covers every configured manifest, parent recording, and segment:
**145 manifests, 446 parent recordings/audio files, and 44,965 segments**.
Dataset files were not changed.

## Definitions and methodology

- **Segment length:** `seg_end - seg_start`, in seconds. The manifest's
  `duration` and `audio_length_s` fields describe the full parent recording,
  not the segment.
- **Word count:** whitespace-separated tokens using Python `str.split()`,
  without normalization.
- **Reference scope:** every segment retains the full parent transcript.
  Per-segment-row reference word counts are therefore parent transcript lengths,
  weighted by the number of child segments. They are **not the number of words
  spoken within each segment**. True segment-level word counts require
  transcript alignment.
- **Parent-level statistics:** each original recording is counted once.
- **Percentiles:** NumPy's default linear interpolation. Values in this report
  are rounded to two decimal places; full precision is retained in the
  [summary JSON](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/summary.json)
  and [statistics CSV](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/statistics.csv).
- **Audio checks:** every ZIP was downloaded and every member read fully for
  CRC validation. Exact uncompressed byte offsets and lengths were checked,
  media headers inspected with libsndfile, and every segment end compared
  with actual audio duration.
- **Other checks:** original and unpacked manifest correspondence, unchanged
  references and metadata, complete parent groups, segment indices/counts,
  nonempty references, time selectors, and segment duration bounds.
- **Limits:** media headers were inspected, but full PCM decoding, ASR, and
  transcript alignment were not performed. Original standalone audio files
  were not re-downloaded; evaluation uses the ZIP members.

## Dataset coverage and audio availability

| Dataset | Manifests | Parent recordings | Segments | Parent audio hours | Missing audio segments | Unusable audio segments |
|---|---:|---:|---:|---:|---:|---:|
| Earnings21 | 10 | 44 | 5,552 | 39.26 | 0 | 0 |
| Earnings22 | 28 | 125 | 17,099 | 119.89 | 0 | 0 |
| CORAAL | 106 | 271 | 22,041 | 160.94 | 0 | 0 |
| Earnings22-Cleaned-AA | 1 | 6 | 273 | 1.92 | 0 | 0 |

All expected parent records are present, and all segment groups are complete.
There are no missing archives/members, CRC failures, invalid byte ranges,
empty references, or segments exceeding 30 seconds.

## Segment length statistics

Unit: seconds. Each segment is counted once.

| Dataset | Min | p50 | p90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| Earnings21 | 0.23 | 27.13 | 29.66 | 29.99 | 25.63 |
| Earnings22 | 0.10 | 26.98 | 29.66 | 29.99 | 25.30 |
| CORAAL | 0.25 | 27.92 | 29.72 | 29.99 | 26.51 |
| Earnings22-Cleaned-AA | 1.96 | 27.02 | 29.68 | 29.99 | 25.15 |

## Manifest reference words per segment row

These are full parent transcript word counts repeated on child rows, not
segment-aligned reference lengths.

| Dataset | Min | p50 | p90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| Earnings21 | 2,715.00 | 9,597.00 | 12,460.00 | 14,593.00 | 9,283.80 |
| Earnings22 | 1,747.00 | 9,163.00 | 13,854.00 | 17,122.00 | 9,169.76 |
| CORAAL | 237.00 | 7,674.00 | 12,354.00 | 17,817.00 | 7,873.67 |
| Earnings22-Cleaned-AA | 1,738.00 | 2,797.00 | 3,412.00 | 3,412.00 | 2,703.62 |

## Reference words per unique parent recording

Each original recording is counted once, without weighting by segment count.

| Dataset | Min | p50 | p90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| Earnings21 | 2,715.00 | 8,813.50 | 11,669.20 | 14,593.00 | 8,235.25 |
| Earnings22 | 1,747.00 | 7,811.00 | 11,959.80 | 17,122.00 | 8,076.94 |
| CORAAL | 237.00 | 5,901.00 | 10,820.00 | 17,817.00 | 6,176.54 |
| Earnings22-Cleaned-AA | 1,738.00 | 2,789.00 | 3,186.50 | 3,412.00 | 2,635.67 |

## Actual parent audio length statistics

Unit: seconds. Durations come from the audio headers, not manifest metadata.

| Dataset | Min | p50 | p90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| Earnings21 | 1,097.10 | 3,280.62 | 4,499.72 | 5,740.61 | 3,212.41 |
| Earnings22 | 874.73 | 3,552.12 | 5,097.35 | 7,407.05 | 3,452.71 |
| CORAAL | 58.75 | 2,173.40 | 3,535.90 | 4,911.80 | 2,137.94 |
| Earnings22-Cleaned-AA | 874.73 | 1,218.83 | 1,301.78 | 1,343.92 | 1,150.16 |

## Number of segments per parent recording

| Dataset | Min | p50 | p90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| Earnings21 | 41.00 | 130.50 | 178.00 | 217.00 | 126.18 |
| Earnings22 | 35.00 | 139.00 | 198.60 | 298.00 | 136.79 |
| CORAAL | 3.00 | 81.00 | 134.00 | 182.00 | 81.33 |
| Earnings22-Cleaned-AA | 35.00 | 48.50 | 50.50 | 51.00 | 45.50 |

## Missing-audio statistics

For each dataset, both the per-segment missing-audio indicator
(`1 = missing`, `0 = available`) and the number of missing-audio segments
per parent recording have the following statistics:

| Dataset | Min | p50 | p90 | Max | Mean | Missing segment rate |
|---|---:|---:|---:|---:|---:|---:|
| Earnings21 | 0 | 0 | 0 | 0 | 0 | 0% |
| Earnings22 | 0 | 0 | 0 | 0 | 0 | 0% |
| CORAAL | 0 | 0 | 0 | 0 | 0 | 0% |
| Earnings22-Cleaned-AA | 0 | 0 | 0 | 0 | 0 | 0% |

## Timing and coverage observations

SVAD allows up to one second of collar overlap between adjacent segments.
Overlap is not missing audio. Time outside the union of selected segment
intervals is reported separately; SVAD can exclude silence/noise.

| Dataset | Overlapping adjacent pairs | Backward-start pairs | Audio hours outside segment union |
|---|---:|---:|---:|
| Earnings21 | 5,034 | 0 | 0.28 |
| Earnings22 | 15,160 | 1 | 1.30 |
| CORAAL | 19,505 | 0 | 1.15 |
| Earnings22-Cleaned-AA | 226 | 0 | 0.03 |

The single Earnings22 backward-start pair remains within the existing
one-second overlap allowance. No overlap exceeds that allowance.

## Metadata warnings

Two Earnings22-Cleaned-AA segment ends exceed the parent duration stored in
the manifest. Both remain within the actual audio duration, so neither is
missing or unusable audio. Segment indices are zero-based.

| Record ID | Segment index | Segment end (s) | Metadata duration (s) | Actual audio duration (s) |
|---|---:|---:|---:|---:|
| 4471586 | 47 | 1,259.50 | 1,259.00 | 1,259.64 |
| 4483046 | 48 | 1,192.32 | 1,187.00 | 1,193.01 |

These metadata fields were reported, not modified.

## Detailed artifacts

Detailed artifacts are archived in Azure Blob Storage and require authorized
access to the storage account.

- [Summary JSON](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/summary.json): full-precision statistics, coverage, and methodology.
- [Statistics CSV](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/statistics.csv): machine-readable min/p50/p90/max/mean tables.
- [All segments](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/segments.csv): every segment's word count, length, timing,
  parent, audio reference, availability, and validation status.
- [All parent records](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/parents.csv): per-parent child-segment statistics,
  coverage, overlaps, and missing-audio counts.
- [Manifest coverage](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/manifests.csv): every manifest and its source correspondence.
- [Audio file checks](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/audio_files.csv): every member's CRC, byte-range,
  duration, sample rate, and format checks.
- [Issue details](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/issues.jsonl): the two metadata-duration warnings.
- [HTML report](https://orngwus2cresco.blob.core.windows.net/data/boren/data/verl/openlong_audit_20261016T184457Z/report.html): browser-readable audit summary.

All requested percentile statistics were independently recomputed from the
exported CSV files and checked against the summary JSON.

# Bad OSS WER Cached Datasets

Filtered generation outputs cached as JSONL on Azure blob.

## WER-based filters

| Config | Filter | Rows | Output Path |
|--------|--------|------|-------------|
| `gen_bad_oss_en_wer05_30` | EN, WER [0.05, 0.3] | 837,442 | `az://orngwus2cresco/data/boren/data/verl/gen_cached/bad_oss_en_wer05_30/data.jsonl` |
| `gen_bad_oss_en_wer30_100` | EN, WER [0.3, 1.0] | 182,091 | `az://orngwus2cresco/data/boren/data/verl/gen_cached/bad_oss_en_wer30_100/data.jsonl` |
| `gen_bad_oss_noen_wer05_30` | non-EN, WER [0.05, 0.3] | 1,553,880 | `az://orngwus2cresco/data/boren/data/verl/gen_cached/bad_oss_noen_wer05_30/data.jsonl` |
| `gen_bad_oss_noen_wer30_100` | non-EN, WER [0.3, 1.0] | 209,276 | `az://orngwus2cresco/data/boren/data/verl/gen_cached/bad_oss_noen_wer30_100/data.jsonl` |

## Error-count-based filters

| Config | Filter | Rows | Output Path |
|--------|--------|------|-------------|
| `gen_bad_oss_en_nerr2_10` | EN, n_err [2, 10] | 746,141 | `az://orngwus2cresco/data/boren/data/verl/gen_cached/bad_oss_en_nerr2_10/data.jsonl` |
| `gen_bad_oss_noen_nerr2_10` | non-EN, n_err [2, 10] | 1,216,950 | `az://orngwus2cresco/data/boren/data/verl/gen_cached/bad_oss_noen_nerr2_10/data.jsonl` |

## English sources

- `ls_raw_rp`, `oss_cv15_en_wer01`, `oss_earning_wer01`, `oss_gigaspeech`, `oss_mls_en`, `oss_ami`, `oss_voxpopuli_en_wer01`, `oss_wavllm_wer01`

## Non-English sources

- `oss_no_en_v1`, `oss_no_en_v0`

# LibriSpeech bad-case verbatim + language-hint dataset statistics

Generated (UTC): 2026-10-16T01:00:06+00:00

## Source and methodology

- Config: [ls_rare_bad_verb_langhint_v0.yaml](../recipe/phimm/config/data/train_data/ls_rare_bad_verb_langhint_v0.yaml).
- Source JSONL: `az://orngwus2cresco/data/boren/data/verl/gen_qwen/2607v1a_earning_step650/ls_rare_nonempty_verb_g32_nonzero_n_err.jsonl`.
- Source SHA-256: `de4052b865ead04daf3b5872fa99f94981739d1f0392ac76b235d510856060d7`.
- Analyze every JSONL record once, before training sampling or repetition.
- Use stored `wer`, `n_err` (word-edit error count), and `n_ref` (scored reference-word count); do not re-normalize or re-score text.
- Verified every row has finite, nonnegative integer error counts, positive integer reference counts, and `wer = n_err / n_ref` within numerical tolerance.
- Corpus WER = sum(n_err) / sum(n_ref); mean utterance WER = mean(wer). These weight utterances differently.
- Word count means stored scoring denominator `n_ref`, not a fresh whitespace split of `text`.
- The config removes reference brackets and requests verbatim transcription with an English language hint. This report describes the original stored scores, not recomputed scores after preprocessing.
- This is an already selected bad-case dataset, not an unbiased LibriSpeech evaluation. High error rates here do not estimate full-corpus performance.
- Source records do not provide separate substitution/deletion/insertion counts, so those components are not inferred.

## Overall summary

| Metric | Value |
| --- | --- |
| Utterances | 71,707 |
| Total word errors | 116,161 |
| Total reference words | 2,560,318 |
| Corpus (word-weighted) WER | 4.537% |
| Mean utterance WER | 5.430% |
| Median utterance WER | 3.448% |
| Mean errors per utterance | 1.620 |
| Mean reference words per utterance | 35.705 |
| Zero-error utterances | 0 |
| Utterances with WER > 100% | 6 |
| Duplicate IDs beyond first occurrence | 0 |
| Blank source lines ignored | 0 |

## Metric distribution and conventional percentiles

Pxx is the conventional ascending percentile (P90 has 90% of values at or below it). Percentiles use linear interpolation at index `(N - 1) * q`; fractional counts are interpolated cutoffs, not actual utterance counts. Standard deviation is population SD.

| Metric | Min | Mean | SD | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Utterance WER | 1.389% | 5.430% | 6.692% | 1.786% | 2.000% | 2.174% | 2.500% | 3.448% | 6.061% | 10.256% | 14.286% | 29.032% | 900.000% |
| Errors per utterance | 1.000 | 1.620 | 1.199 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.000 | 3.000 | 4.000 | 6.000 | 26.000 |
| Reference words per utterance | 1.000 | 35.705 | 11.002 | 7.000 | 14.000 | 20.000 | 30.000 | 37.000 | 43.000 | 49.000 | 52.000 | 57.000 | 77.000 |

## WER distribution

| Bucket | Utterances | Utterance share | Errors | Error share | Reference words | Corpus WER |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | 0 | 0.000% | 0 | 0.000% | 0 | - |
| (0%, 5%] | 48,179 | 67.189% | 54,704 | 47.093% | 1,885,204 | 2.902% |
| (5%, 10%] | 16,260 | 22.676% | 34,962 | 30.098% | 510,688 | 6.846% |
| (10%, 20%] | 5,605 | 7.817% | 18,340 | 15.788% | 136,489 | 13.437% |
| (20%, 30%] | 1,000 | 1.395% | 4,719 | 4.062% | 19,529 | 24.164% |
| (30%, 50%] | 539 | 0.752% | 2,731 | 2.351% | 7,376 | 37.025% |
| (50%, 100%] | 118 | 0.165% | 661 | 0.569% | 1,005 | 65.771% |
| >100% | 6 | 0.008% | 44 | 0.038% | 27 | 162.963% |

## Error-count distribution (largest first)

| Bucket | Utterances | Utterance share | Errors | Error share | Reference words | Corpus WER |
| --- | --- | --- | --- | --- | --- | --- |
| 26 | 1 | 0.001% | 26 | 0.022% | 36 | 72.222% |
| 23 | 1 | 0.001% | 23 | 0.020% | 55 | 41.818% |
| 21 | 2 | 0.003% | 42 | 0.036% | 76 | 55.263% |
| 20 | 1 | 0.001% | 20 | 0.017% | 47 | 42.553% |
| 19 | 3 | 0.004% | 57 | 0.049% | 87 | 65.517% |
| 18 | 3 | 0.004% | 54 | 0.046% | 152 | 35.526% |
| 17 | 3 | 0.004% | 51 | 0.044% | 148 | 34.459% |
| 16 | 5 | 0.007% | 80 | 0.069% | 222 | 36.036% |
| 15 | 13 | 0.018% | 195 | 0.168% | 569 | 34.271% |
| 14 | 8 | 0.011% | 112 | 0.096% | 333 | 33.634% |
| 13 | 23 | 0.032% | 299 | 0.257% | 852 | 35.094% |
| 12 | 25 | 0.035% | 300 | 0.258% | 1,004 | 29.880% |
| 11 | 36 | 0.050% | 396 | 0.341% | 1,442 | 27.462% |
| 10 | 61 | 0.085% | 610 | 0.525% | 2,415 | 25.259% |
| 9 | 91 | 0.127% | 819 | 0.705% | 3,307 | 24.766% |
| 8 | 161 | 0.225% | 1,288 | 1.109% | 6,086 | 21.163% |
| 7 | 257 | 0.358% | 1,799 | 1.549% | 9,210 | 19.533% |
| 6 | 491 | 0.685% | 2,946 | 2.536% | 17,651 | 16.690% |
| 5 | 855 | 1.192% | 4,275 | 3.680% | 30,842 | 13.861% |
| 4 | 2,238 | 3.121% | 8,952 | 7.707% | 79,792 | 11.219% |
| 3 | 5,200 | 7.252% | 15,600 | 13.430% | 187,157 | 8.335% |
| 2 | 15,988 | 22.296% | 31,976 | 27.527% | 572,447 | 5.586% |
| 1 | 46,241 | 64.486% | 46,241 | 39.808% | 1,646,388 | 2.809% |

## Reference-word-count distribution

| Bucket | Utterances | Utterance share | Errors | Error share | Reference words | Corpus WER |
| --- | --- | --- | --- | --- | --- | --- |
| 1-5 | 346 | 0.483% | 505 | 0.435% | 1,440 | 35.069% |
| 6-10 | 1,719 | 2.397% | 2,617 | 2.253% | 14,070 | 18.600% |
| 11-15 | 2,428 | 3.386% | 3,781 | 3.255% | 32,001 | 11.815% |
| 16-20 | 3,072 | 4.284% | 5,018 | 4.320% | 55,497 | 9.042% |
| 21-30 | 11,850 | 16.526% | 19,339 | 16.648% | 312,137 | 6.196% |
| 31-40 | 26,707 | 37.245% | 42,894 | 36.926% | 957,310 | 4.481% |
| 41-50 | 20,924 | 29.180% | 34,018 | 29.285% | 935,415 | 3.637% |
| 51-75 | 4,660 | 6.499% | 7,987 | 6.876% | 252,371 | 3.165% |
| 76-100 | 1 | 0.001% | 2 | 0.002% | 77 | 2.597% |
| >=101 | 0 | 0.000% | 0 | 0.000% | 0 | - |

## Descending-error percentiles: cumulative top slices

Rank by `n_err` from largest to smallest. Top X% contains `ceil(N * X / 100)` records. Ties retain original JSONL order and may be split at the boundary; error totals are unaffected by tie order, but selected word totals/WER can vary. The cutoff is the smallest error count inside the selected slice. Top 10% corresponds approximately to the upper P90 tail, not the lowest P10.

| Slice | Utterances | Utterance share | Errors | Error share | Reference words | Corpus WER | Error cutoff | Mean errors | Mean words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Top 1% | 718 | 1.001% | 6,315 | 5.436% | 26,946 | 23.436% | 6 | 8.795 | 37.529 |
| Top 5% | 3,586 | 5.001% | 19,576 | 16.852% | 129,781 | 15.084% | 4 | 5.459 | 36.191 |
| Top 10% | 7,171 | 10.000% | 31,023 | 26.707% | 259,886 | 11.937% | 3 | 4.326 | 36.241 |
| Top 20% | 14,342 | 20.001% | 47,672 | 41.040% | 521,671 | 9.138% | 2 | 3.324 | 36.374 |
| Top 25% | 17,927 | 25.000% | 54,842 | 47.212% | 648,981 | 8.450% | 2 | 3.059 | 36.201 |
| Top 30% | 21,513 | 30.001% | 62,014 | 53.386% | 774,529 | 8.007% | 2 | 2.883 | 36.003 |
| Top 40% | 28,683 | 40.000% | 73,137 | 62.962% | 1,033,413 | 7.077% | 1 | 2.550 | 36.029 |
| Top 50% | 35,854 | 50.001% | 80,308 | 69.135% | 1,293,515 | 6.209% | 1 | 2.240 | 36.077 |
| Top 60% | 43,025 | 60.001% | 87,479 | 75.308% | 1,555,691 | 5.623% | 1 | 2.033 | 36.158 |
| Top 70% | 50,195 | 70.000% | 94,649 | 81.481% | 1,810,342 | 5.228% | 1 | 1.886 | 36.066 |
| Top 75% | 53,781 | 75.001% | 98,235 | 84.568% | 1,934,796 | 5.077% | 1 | 1.827 | 35.975 |
| Top 80% | 57,366 | 80.001% | 101,820 | 87.654% | 2,058,823 | 4.946% | 1 | 1.775 | 35.889 |
| Top 90% | 64,537 | 90.001% | 108,991 | 93.828% | 2,313,107 | 4.712% | 1 | 1.689 | 35.842 |
| Top 95% | 68,122 | 95.000% | 112,576 | 96.914% | 2,436,398 | 4.621% | 1 | 1.653 | 35.765 |
| Top 99% | 70,990 | 99.000% | 115,444 | 99.383% | 2,534,798 | 4.554% | 1 | 1.626 | 35.706 |
| Top 100% | 71,707 | 100.000% | 116,161 | 100.000% | 2,560,318 | 4.537% | 1 | 1.620 | 35.705 |

## Descending-error deciles: non-overlapping bands

Each record occurs in exactly one band. The first band is the highest-error 10%; the last band is the lowest-error 10%.

| Rank band | Utterances | Utterance share | Errors | Error share | Reference words | Corpus WER | Error range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-10% | 7,171 | 10.000% | 31,023 | 26.707% | 259,886 | 11.937% | 3-26 |
| 10-20% | 7,171 | 10.000% | 16,649 | 14.333% | 261,785 | 6.360% | 2-3 |
| 20-30% | 7,171 | 10.000% | 14,342 | 12.347% | 252,858 | 5.672% | 2-2 |
| 30-40% | 7,170 | 9.999% | 11,123 | 9.576% | 258,884 | 4.297% | 1-2 |
| 40-50% | 7,171 | 10.000% | 7,171 | 6.173% | 260,102 | 2.757% | 1-1 |
| 50-60% | 7,171 | 10.000% | 7,171 | 6.173% | 262,176 | 2.735% | 1-1 |
| 60-70% | 7,170 | 9.999% | 7,170 | 6.172% | 254,651 | 2.816% | 1-1 |
| 70-80% | 7,171 | 10.000% | 7,171 | 6.173% | 248,481 | 2.886% | 1-1 |
| 80-90% | 7,171 | 10.000% | 7,171 | 6.173% | 254,284 | 2.820% | 1-1 |
| 90-100% | 7,170 | 9.999% | 7,170 | 6.172% | 247,211 | 2.900% | 1-1 |

## Top 20 utterances by absolute error count

| Rank | ID | Errors | Reference words | WER |
| --- | --- | --- | --- | --- |
| 1 | 4423-29084-0030.flac | 26 | 36 | 72.222% |
| 2 | 6687-119386-0004.flac | 23 | 55 | 41.818% |
| 3 | 2946-163375-0086.flac | 21 | 43 | 48.837% |
| 4 | 6709-74047-0001.flac | 21 | 33 | 63.636% |
| 5 | 5456-62014-0017.flac | 20 | 47 | 42.553% |
| 6 | 5456-62014-0004.flac | 19 | 26 | 73.077% |
| 7 | 1401-174511-0002.flac | 19 | 30 | 63.333% |
| 8 | 4910-14125-0045.flac | 19 | 31 | 61.290% |
| 9 | 2946-163375-0105.flac | 18 | 45 | 40.000% |
| 10 | 4841-26659-0065.flac | 18 | 54 | 33.333% |
| 11 | 6535-66371-0025.flac | 18 | 53 | 33.962% |
| 12 | 5456-62014-0006.flac | 17 | 33 | 51.515% |
| 13 | 4910-14125-0030.flac | 17 | 67 | 25.373% |
| 14 | 6535-66371-0029.flac | 17 | 48 | 35.417% |
| 15 | 5192-19397-0046.flac | 16 | 40 | 40.000% |
| 16 | 6378-61780-0033.flac | 16 | 46 | 34.783% |
| 17 | 6378-61781-0004.flac | 16 | 61 | 26.230% |
| 18 | 2946-163375-0076.flac | 16 | 38 | 42.105% |
| 19 | 6687-119386-0032.flac | 16 | 37 | 43.243% |
| 20 | 5456-62014-0008.flac | 15 | 47 | 31.915% |

## Takeaways

- The highest-error 10% of utterances contain **26.707%** of all word errors; their corpus WER is **11.937%**.
- The highest-error 20% of utterances contain **41.040%** of all word errors; their corpus WER is **9.138%**.
- The highest-error 50% of utterances contain **69.135%** of all word errors; their corpus WER is **6.209%**.
- **64.486%** of utterances have exactly one word error.
- Absolute error count prioritizes total error volume and can favor longer utterances; WER prioritizes error density. Use both when choosing training subsets.

## Generated p50 and p80 training subsets

**Definition:** p50 and p80 retain the highest-error 50% and 80% of records, respectively, not records accounting for 50%/80% of total errors, and not the lowest-error conventional P50/P80 tails. Rank by descending `n_err`; take `ceil(N * fraction)`. Boundary ties retain source JSONL order and are split to give exact target sizes. Output records are in descending error order, with every original field and value preserved.

Both YAMLs preserve the original English verbatim task, language hint, bracket removal, and `data_source: ls_rare_verb`. Their `jsonl_paths` point to the uploaded subsets. P50 is a subset of P80; remote bytes were SHA-256 verified against local files.

| Subset | Records | Minimum errors | Total errors | Share of source errors | Reference words | Corpus WER | Boundary ties retained / total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p50 | 35,854 | 1 | 80,308 | 69.135% | 1,293,515 | 6.209% | 10,388 / 46,241 |
| p80 | 57,366 | 1 | 101,820 | 87.654% | 2,058,823 | 4.946% | 31,900 / 46,241 |

### p50 artifacts

- Dataset YAML: [ls_rare_bad_verb_langhint_v0_p50.yaml](../recipe/phimm/config/data/train_data/ls_rare_bad_verb_langhint_v0_p50.yaml).
- Local JSONL: [ls_rare_nonempty_verb_g32_nonzero_n_err_p50.jsonl](../tmp/ls_bad_percentiles/ls_rare_nonempty_verb_g32_nonzero_n_err_p50.jsonl).
- Blob JSONL: `az://orngwus2cresco/data/boren/data/verl/gen_qwen/2607v1a_earning_step650/ls_rare_nonempty_verb_g32_nonzero_n_err_p50.jsonl`.
- SHA-256: `b19bf3a76691a13dd7dfb38421f2b8b2c6867e14d0f30c63ab3d5e87ac18b6eb`.

### p80 artifacts

- Dataset YAML: [ls_rare_bad_verb_langhint_v0_p80.yaml](../recipe/phimm/config/data/train_data/ls_rare_bad_verb_langhint_v0_p80.yaml).
- Local JSONL: [ls_rare_nonempty_verb_g32_nonzero_n_err_p80.jsonl](../tmp/ls_bad_percentiles/ls_rare_nonempty_verb_g32_nonzero_n_err_p80.jsonl).
- Blob JSONL: `az://orngwus2cresco/data/boren/data/verl/gen_qwen/2607v1a_earning_step650/ls_rare_nonempty_verb_g32_nonzero_n_err_p80.jsonl`.
- SHA-256: `7947a9dbc0361f605be9c4aa753bded974aff77c1caa33e51662675b5eef0cbb`.

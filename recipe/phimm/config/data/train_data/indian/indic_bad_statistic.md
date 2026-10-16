# IndicVoices Hindi bad-sample dataset statistics

Analysis date: 2026-10-16. Full-source analysis and subset publication ran remotely on `verl-n2-i0`; only compact reports and ten sample records were retrieved locally.

## Source and interpretation

- Source: `az://orngwus2cresco/data/boren/data/verl/cached_qwen/v2611/2609r2/indic_voice_wer0001_200/data.jsonl`
- Source SHA-256: `dfa19c6e61b09003cd070ffc8236152b33a6191f46432e256b37d8f0be17b1e6`
- Full training configuration: [indic_bad.yaml](./indic_bad.yaml).
- Cache configuration: [cache_indic_voice__2609r2_wer_gt0.yaml](../../../v2611_dev/gen/cache_indic_voice__2609r2_wer_gt0.yaml).

- Cached ratio, verified per row as n_err/n_ref; corpus WER = sum(n_err)/sum(n_ref).
- Hindi OpenASR multilingual normalization with compound normalization; first 100 rows independently rescored.
- Raw whitespace-separated words in text (reference) and response (hypothesis); n_ref is the distinct normalized scoring denominator.
- Already filtered to cached WER [0.001, 200] inclusive, i.e. [0.1%, 20000%]. No further filtering.
- Error-only selected subset, not an unbiased benchmark; no duration field, so audio hours cannot be inferred. Cached counts do not include substitution/deletion/insertion breakdown.
- The id field identifies 431 shared chunk containers, not utterances. All 286066 full audio_chunk references are unique; no duplicate audio/text pairs.
- All rates below are percentages unless explicitly labeled as ratios. Raw word counts and normalized scoring units are intentionally reported separately.
- WER buckets use (lower, upper] boundaries; the first includes zero. Integer count buckets include both named endpoints. Overflow buckets have no upper bound.
- Corpus WER is reference-length weighted; mean utterance WER gives each sample equal weight. WER may exceed 100% when insertions outnumber the reference units.

## Dataset overview

| Metric | Value |
| --- | --- |
| Samples | 286,066 |
| Language | Hindi (100%) |
| Corpus WER | 12.9372% |
| Mean utterance WER | 20.6913% |
| Total errors (n_err) | 1,261,624 |
| Raw reference words | 6,149,088 |
| Raw hypothesis words | 6,025,674 |
| Normalized reference units (n_ref) | 9,751,908 |
| WER > 100% samples | 1,455 (0.5086%) |
| Hypothesis > 2x raw reference length | 258 (0.0902%) |
| Independently rescored samples | 100 |

## Summary statistics

| Metric | Count | Sum | Mean | Population std. deviation |
| --- | --- | --- | --- | --- |
| Utterance WER (%) | 286,066 | 5,919,070.909175 | 20.691277 | 34.029075 |
| Raw reference words | 286,066 | 6,149,088.000000 | 21.495347 | 14.591187 |
| Raw hypothesis words | 286,066 | 6,025,674.000000 | 21.063929 | 14.570931 |
| Normalized reference units | 286,066 | 9,751,908.000000 | 34.089714 | 22.859824 |
| Error count | 286,066 | 1,261,624.000000 | 4.410255 | 5.229612 |

The sum of utterance WER percentages is included for completeness, not as an evaluation metric.

## Quantiles

| Quantile | Utterance WER (%) | Raw reference words | Raw hypothesis words | Normalized reference units | Error count |
| --- | --- | --- | --- | --- | --- |
| P0 | 0.724638 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| P1 | 1.562500 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| P5 | 2.500000 | 2.000000 | 2.000000 | 3.000000 | 1.000000 |
| P10 | 3.333333 | 4.000000 | 4.000000 | 7.000000 | 1.000000 |
| P25 | 5.882353 | 10.000000 | 10.000000 | 16.000000 | 2.000000 |
| P50 | 11.290323 | 19.000000 | 19.000000 | 31.000000 | 3.000000 |
| P75 | 23.076923 | 30.000000 | 29.000000 | 48.000000 | 5.000000 |
| P90 | 50.000000 | 42.000000 | 41.000000 | 66.000000 | 9.000000 |
| P95 | 84.210526 | 49.000000 | 49.000000 | 77.000000 | 13.000000 |
| P99 | 100.000000 | 64.000000 | 64.000000 | 99.000000 | 25.000000 |
| P100 | 6,750.000000 | 168.000000 | 251.000000 | 252.000000 | 617.000000 |

## WER distribution: 10 buckets

| Bucket | Samples | Samples (%) | Cumulative samples (%) |
| --- | --- | --- | --- |
| 0-2.5% | 15,152 | 5.2967% | 5.2967% |
| 2.5-5% | 43,303 | 15.1374% | 20.4341% |
| 5-10% | 72,003 | 25.1701% | 45.6042% |
| 10-15% | 44,442 | 15.5356% | 61.1397% |
| 15-20% | 29,973 | 10.4777% | 71.6174% |
| 20-30% | 28,119 | 9.8295% | 81.4469% |
| 30-50% | 27,826 | 9.7271% | 91.1741% |
| 50-100% | 23,793 | 8.3173% | 99.4914% |
| 100-200% | 1,258 | 0.4398% | 99.9311% |
| >200% | 197 | 0.0689% | 100.0000% |

## Raw reference word counts: 10 buckets

| Bucket | Samples | Samples (%) | Cumulative samples (%) |
| --- | --- | --- | --- |
| 0-5 | 38,036 | 13.2962% | 13.2962% |
| 6-10 | 36,463 | 12.7464% | 26.0426% |
| 11-15 | 37,486 | 13.1040% | 39.1466% |
| 16-20 | 40,590 | 14.1890% | 53.3356% |
| 21-25 | 35,892 | 12.5468% | 65.8823% |
| 26-30 | 28,645 | 10.0134% | 75.8958% |
| 31-40 | 37,489 | 13.1050% | 89.0008% |
| 41-50 | 18,818 | 6.5782% | 95.5790% |
| 51-80 | 12,387 | 4.3301% | 99.9091% |
| 81+ | 260 | 0.0909% | 100.0000% |

## Raw hypothesis word counts: 10 buckets

| Bucket | Samples | Samples (%) | Cumulative samples (%) |
| --- | --- | --- | --- |
| 0-5 | 39,996 | 13.9814% | 13.9814% |
| 6-10 | 37,010 | 12.9376% | 26.9190% |
| 11-15 | 39,109 | 13.6713% | 40.5903% |
| 16-20 | 40,983 | 14.3264% | 54.9167% |
| 21-25 | 35,160 | 12.2909% | 67.2076% |
| 26-30 | 27,783 | 9.7121% | 76.9197% |
| 31-40 | 35,736 | 12.4922% | 89.4119% |
| 41-50 | 18,074 | 6.3181% | 95.7300% |
| 51-80 | 11,957 | 4.1798% | 99.9098% |
| 81+ | 258 | 0.0902% | 100.0000% |

## Normalized reference units: 10 buckets

| Bucket | Samples | Samples (%) | Cumulative samples (%) |
| --- | --- | --- | --- |
| 0-5 | 22,307 | 7.7979% | 7.7979% |
| 6-10 | 22,914 | 8.0100% | 15.8079% |
| 11-15 | 22,958 | 8.0254% | 23.8333% |
| 16-20 | 22,513 | 7.8699% | 31.7032% |
| 21-25 | 25,029 | 8.7494% | 40.4526% |
| 26-30 | 26,181 | 9.1521% | 49.6046% |
| 31-40 | 46,505 | 16.2567% | 65.8614% |
| 41-50 | 35,199 | 12.3045% | 78.1659% |
| 51-80 | 50,738 | 17.7365% | 95.9023% |
| 81+ | 11,722 | 4.0977% | 100.0000% |

## Error-count distribution: 10 buckets

| Bucket | Samples | Samples (%) | Cumulative samples (%) |
| --- | --- | --- | --- |
| 0-1 | 68,365 | 23.8983% | 23.8983% |
| 2 | 56,105 | 19.6126% | 43.5109% |
| 3 | 40,246 | 14.0688% | 57.5797% |
| 4 | 30,008 | 10.4899% | 68.0696% |
| 5 | 22,023 | 7.6986% | 75.7682% |
| 6-7 | 28,243 | 9.8729% | 85.6411% |
| 8-10 | 20,064 | 7.0138% | 92.6548% |
| 11-15 | 11,807 | 4.1274% | 96.7822% |
| 16-25 | 6,366 | 2.2254% | 99.0076% |
| 26+ | 2,839 | 0.9924% | 100.0000% |

## WER by raw reference length

| Reference words | Samples | Mean WER (%) | Median WER (%) | Errors | Normalized reference units | Corpus WER (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 0-5 | 38,036 | 62.0062% | 50.0000% | 91,062 | 191,579 | 47.5323% |
| 6-10 | 36,463 | 24.9110% | 18.1818% | 114,465 | 476,509 | 24.0216% |
| 11-15 | 37,486 | 16.7582% | 12.0000% | 131,831 | 802,549 | 16.4265% |
| 16-20 | 40,590 | 13.2660% | 9.6774% | 154,496 | 1,171,990 | 13.1824% |
| 21-25 | 35,892 | 12.6126% | 8.5714% | 168,368 | 1,313,438 | 12.8189% |
| 26-30 | 28,645 | 11.6565% | 8.1633% | 148,150 | 1,266,435 | 11.6982% |
| 31-40 | 37,489 | 10.8452% | 7.6923% | 224,472 | 2,062,965 | 10.8810% |
| 41-50 | 18,818 | 9.5925% | 7.1429% | 126,932 | 1,316,660 | 9.6405% |
| 51-80 | 12,387 | 8.8416% | 6.8966% | 98,638 | 1,116,208 | 8.8369% |
| 81+ | 260 | 9.2677% | 7.8938% | 3,210 | 33,575 | 9.5607% |

## Error-count quantiles with exact sample coverage

Integer ties mean that the actual coverage at a quantile threshold can exceed its nominal percentile.

| Quantile | Error count | Samples below | Samples equal | Samples at or below | Coverage (%) | Samples above |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | 1 | 0 | 68,365 | 68,365 | 23.8983% | 217,701 |
| P5 | 1 | 0 | 68,365 | 68,365 | 23.8983% | 217,701 |
| P10 | 1 | 0 | 68,365 | 68,365 | 23.8983% | 217,701 |
| P25 | 2 | 68,365 | 56,105 | 124,470 | 43.5109% | 161,596 |
| P50 | 3 | 124,470 | 40,246 | 164,716 | 57.5797% | 121,350 |
| P75 | 5 | 194,724 | 22,023 | 216,747 | 75.7682% | 69,319 |
| P90 | 9 | 253,772 | 6,464 | 260,236 | 90.9706% | 25,830 |
| P95 | 13 | 271,545 | 2,188 | 273,733 | 95.6888% | 12,333 |
| P99 | 25 | 282,960 | 267 | 283,227 | 99.0076% | 2,839 |
| P100 | 617 | 286,065 | 1 | 286,066 | 100.0000% | 0 |

Descending accumulation includes all samples with `n_err >= threshold`. Sample share and total-error share are different denominators.

## Descending error accumulation: 10 thresholds

| Minimum errors | Cumulative samples | Samples (%) | Cumulative errors | Total errors (%) |
| --- | --- | --- | --- | --- |
| >= 26 | 2,839 | 0.9924% | 102,171 | 8.0984% |
| >= 16 | 9,205 | 3.2178% | 224,383 | 17.7853% |
| >= 11 | 21,012 | 7.3452% | 372,190 | 29.5009% |
| >= 8 | 41,076 | 14.3589% | 548,802 | 43.4996% |
| >= 6 | 69,319 | 24.2318% | 730,164 | 57.8749% |
| >= 5 | 91,342 | 31.9304% | 840,279 | 66.6030% |
| >= 4 | 121,350 | 42.4203% | 960,311 | 76.1171% |
| >= 3 | 161,596 | 56.4891% | 1,081,049 | 85.6871% |
| >= 2 | 217,701 | 76.1017% | 1,193,259 | 94.5812% |
| >= 1 | 286,066 | 100.0000% | 1,261,624 | 100.0000% |

## Exact error histogram and descending accumulation: all 91 observed counts

| Minimum errors | Samples with exactly this error count | Cumulative samples | Samples (%) | Cumulative errors | Total errors (%) |
| --- | --- | --- | --- | --- | --- |
| >= 617 | 1 | 1 | 0.0003% | 617 | 0.0489% |
| >= 408 | 1 | 2 | 0.0007% | 1,025 | 0.0812% |
| >= 405 | 1 | 3 | 0.0010% | 1,430 | 0.1133% |
| >= 302 | 1 | 4 | 0.0014% | 1,732 | 0.1373% |
| >= 251 | 1 | 5 | 0.0017% | 1,983 | 0.1572% |
| >= 196 | 1 | 6 | 0.0021% | 2,179 | 0.1727% |
| >= 142 | 1 | 7 | 0.0024% | 2,321 | 0.1840% |
| >= 111 | 1 | 8 | 0.0028% | 2,432 | 0.1928% |
| >= 108 | 1 | 9 | 0.0031% | 2,540 | 0.2013% |
| >= 94 | 1 | 10 | 0.0035% | 2,634 | 0.2088% |
| >= 84 | 2 | 12 | 0.0042% | 2,802 | 0.2221% |
| >= 83 | 2 | 14 | 0.0049% | 2,968 | 0.2353% |
| >= 82 | 2 | 16 | 0.0056% | 3,132 | 0.2483% |
| >= 80 | 1 | 17 | 0.0059% | 3,212 | 0.2546% |
| >= 79 | 2 | 19 | 0.0066% | 3,370 | 0.2671% |
| >= 78 | 2 | 21 | 0.0073% | 3,526 | 0.2795% |
| >= 77 | 3 | 24 | 0.0084% | 3,757 | 0.2978% |
| >= 76 | 1 | 25 | 0.0087% | 3,833 | 0.3038% |
| >= 74 | 2 | 27 | 0.0094% | 3,981 | 0.3155% |
| >= 73 | 3 | 30 | 0.0105% | 4,200 | 0.3329% |
| >= 71 | 1 | 31 | 0.0108% | 4,271 | 0.3385% |
| >= 70 | 3 | 34 | 0.0119% | 4,481 | 0.3552% |
| >= 69 | 3 | 37 | 0.0129% | 4,688 | 0.3716% |
| >= 68 | 1 | 38 | 0.0133% | 4,756 | 0.3770% |
| >= 67 | 5 | 43 | 0.0150% | 5,091 | 0.4035% |
| >= 66 | 5 | 48 | 0.0168% | 5,421 | 0.4297% |
| >= 65 | 4 | 52 | 0.0182% | 5,681 | 0.4503% |
| >= 64 | 5 | 57 | 0.0199% | 6,001 | 0.4757% |
| >= 63 | 4 | 61 | 0.0213% | 6,253 | 0.4956% |
| >= 62 | 6 | 67 | 0.0234% | 6,625 | 0.5251% |
| >= 61 | 11 | 78 | 0.0273% | 7,296 | 0.5783% |
| >= 60 | 9 | 87 | 0.0304% | 7,836 | 0.6211% |
| >= 59 | 16 | 103 | 0.0360% | 8,780 | 0.6959% |
| >= 58 | 4 | 107 | 0.0374% | 9,012 | 0.7143% |
| >= 57 | 12 | 119 | 0.0416% | 9,696 | 0.7685% |
| >= 56 | 12 | 131 | 0.0458% | 10,368 | 0.8218% |
| >= 55 | 6 | 137 | 0.0479% | 10,698 | 0.8480% |
| >= 54 | 12 | 149 | 0.0521% | 11,346 | 0.8993% |
| >= 53 | 13 | 162 | 0.0566% | 12,035 | 0.9539% |
| >= 52 | 16 | 178 | 0.0622% | 12,867 | 1.0199% |
| >= 51 | 13 | 191 | 0.0668% | 13,530 | 1.0724% |
| >= 50 | 16 | 207 | 0.0724% | 14,330 | 1.1358% |
| >= 49 | 21 | 228 | 0.0797% | 15,359 | 1.2174% |
| >= 48 | 37 | 265 | 0.0926% | 17,135 | 1.3582% |
| >= 47 | 36 | 301 | 0.1052% | 18,827 | 1.4923% |
| >= 46 | 37 | 338 | 0.1182% | 20,529 | 1.6272% |
| >= 45 | 44 | 382 | 0.1335% | 22,509 | 1.7841% |
| >= 44 | 53 | 435 | 0.1521% | 24,841 | 1.9690% |
| >= 43 | 64 | 499 | 0.1744% | 27,593 | 2.1871% |
| >= 42 | 54 | 553 | 0.1933% | 29,861 | 2.3669% |
| >= 41 | 78 | 631 | 0.2206% | 33,059 | 2.6204% |
| >= 40 | 73 | 704 | 0.2461% | 35,979 | 2.8518% |
| >= 39 | 99 | 803 | 0.2807% | 39,840 | 3.1578% |
| >= 38 | 98 | 901 | 0.3150% | 43,564 | 3.4530% |
| >= 37 | 103 | 1,004 | 0.3510% | 47,375 | 3.7551% |
| >= 36 | 108 | 1,112 | 0.3887% | 51,263 | 4.0633% |
| >= 35 | 107 | 1,219 | 0.4261% | 55,008 | 4.3601% |
| >= 34 | 106 | 1,325 | 0.4632% | 58,612 | 4.6458% |
| >= 33 | 112 | 1,437 | 0.5023% | 62,308 | 4.9387% |
| >= 32 | 117 | 1,554 | 0.5432% | 66,052 | 5.2355% |
| >= 31 | 155 | 1,709 | 0.5974% | 70,857 | 5.6163% |
| >= 30 | 160 | 1,869 | 0.6533% | 75,657 | 5.9968% |
| >= 29 | 190 | 2,059 | 0.7198% | 81,167 | 6.4335% |
| >= 28 | 227 | 2,286 | 0.7991% | 87,523 | 6.9373% |
| >= 27 | 270 | 2,556 | 0.8935% | 94,813 | 7.5152% |
| >= 26 | 283 | 2,839 | 0.9924% | 102,171 | 8.0984% |
| >= 25 | 267 | 3,106 | 1.0858% | 108,846 | 8.6275% |
| >= 24 | 333 | 3,439 | 1.2022% | 116,838 | 9.2609% |
| >= 23 | 353 | 3,792 | 1.3256% | 124,957 | 9.9045% |
| >= 22 | 445 | 4,237 | 1.4811% | 134,747 | 10.6804% |
| >= 21 | 518 | 4,755 | 1.6622% | 145,625 | 11.5427% |
| >= 20 | 694 | 5,449 | 1.9048% | 159,505 | 12.6428% |
| >= 19 | 688 | 6,137 | 2.1453% | 172,577 | 13.6790% |
| >= 18 | 851 | 6,988 | 2.4428% | 187,895 | 14.8931% |
| >= 17 | 1,016 | 8,004 | 2.7980% | 205,167 | 16.2621% |
| >= 16 | 1,201 | 9,205 | 3.2178% | 224,383 | 17.7853% |
| >= 15 | 1,353 | 10,558 | 3.6908% | 244,678 | 19.3939% |
| >= 14 | 1,775 | 12,333 | 4.3112% | 269,528 | 21.3636% |
| >= 13 | 2,188 | 14,521 | 5.0761% | 297,972 | 23.6181% |
| >= 12 | 2,817 | 17,338 | 6.0608% | 331,776 | 26.2975% |
| >= 11 | 3,674 | 21,012 | 7.3452% | 372,190 | 29.5009% |
| >= 10 | 4,818 | 25,830 | 9.0294% | 420,370 | 33.3198% |
| >= 9 | 6,464 | 32,294 | 11.2890% | 478,546 | 37.9310% |
| >= 8 | 8,782 | 41,076 | 14.3589% | 548,802 | 43.4996% |
| >= 7 | 11,904 | 52,980 | 18.5202% | 632,130 | 50.1045% |
| >= 6 | 16,339 | 69,319 | 24.2318% | 730,164 | 57.8749% |
| >= 5 | 22,023 | 91,342 | 31.9304% | 840,279 | 66.6030% |
| >= 4 | 30,008 | 121,350 | 42.4203% | 960,311 | 76.1171% |
| >= 3 | 40,246 | 161,596 | 56.4891% | 1,081,049 | 85.6871% |
| >= 2 | 56,105 | 217,701 | 76.1017% | 1,193,259 | 94.5812% |
| >= 1 | 68,365 | 286,066 | 100.0000% | 1,261,624 | 100.0000% |

## Published training subsets

All three JSONLs are beside the original `data.jsonl`. Full source records and source order are preserved. These subsets overlap and must not be treated as disjoint splits.

| Subset | Selection | Samples | Samples (%) | Errors | Total errors (%) |
| --- | --- | --- | --- | --- | --- |
| P50 | n_err >= 7 | 52,980 | 18.5202% | 632,130 | 50.1045% |
| P80 | n_err >= 3 | 161,596 | 56.4891% | 1,081,049 | 85.6871% |
| Errors <= 3 | n_err <= 3 | 164,716 | 57.5797% | 301,313 | 23.8829% |

P50/P80 refer to cumulative **total-error mass**, not sample-count percentiles. All threshold ties are included: P50 covers 50.1045%; P80 covers 85.6871% (the stricter `n_err >= 4` covers only 76.1171%).

### P50 total-error subset

- JSONL: `az://orngwus2cresco/data/boren/data/verl/cached_qwen/v2611/2609r2/indic_voice_wer0001_200/data_total_errors_p50.jsonl`
- YAML: [indic_bad_total_errors_p50.yaml](./indic_bad_total_errors_p50.yaml).
- SHA-256: `e07fac7159c12d8428e7beff6b530ae027fba69b3ed2024f7951b692a1a1c3c1`
- Verified by a complete remote reread, row/error counts, and SHA-256 comparison.

### P80 total-error subset

- JSONL: `az://orngwus2cresco/data/boren/data/verl/cached_qwen/v2611/2609r2/indic_voice_wer0001_200/data_total_errors_p80.jsonl`
- YAML: [indic_bad_total_errors_p80.yaml](./indic_bad_total_errors_p80.yaml).
- SHA-256: `a44ecf40d8aed2e83b086f60ea9fa897d176f80b492d2b3c4590abd67a7f7a7f`
- Verified by a complete remote reread, row/error counts, and SHA-256 comparison.

### Error count <= 3 subset

- JSONL: `az://orngwus2cresco/data/boren/data/verl/cached_qwen/v2611/2609r2/indic_voice_wer0001_200/data_errors_le3.jsonl`
- YAML: [indic_bad_errors_le3.yaml](./indic_bad_errors_le3.yaml).
- SHA-256: `176ac6d545d18645a9c922fd25a4893064f40e7c2c88154db49a97b34a157eec`
- Normalized reference units: 4,614,260.
- Corpus WER: 6.530039%.
- Verified by a complete remote reread, row/error counts, and SHA-256 comparison.

| Error count | Samples |
| --- | --- |
| 1 | 68,365 |
| 2 | 56,105 |
| 3 | 40,246 |

No zero-error samples are present because the original source is already error-filtered. The `n_err >= 3` P80 subset overlaps the `n_err <= 3` subset on all 40,246 samples with exactly 3 errors.

## Quality and identity checks

| Check | Value |
| --- | --- |
| empty_reference | 0 |
| empty_hypothesis | 0 |
| audio_chunk_rows | 286,066 |
| audio_path_rows | 0 |
| repeated_container_id_rows | 285,635 |
| unique_container_ids | 431 |
| unique_audio_references | 286,066 |
| duplicate_audio_reference_rows | 0 |
| duplicate_audio_text_pair_rows | 0 |

Repeated IDs identify chunk containers, not duplicate utterances. Use the full `audio_chunk` reference, including its record index, as the unique audio identity.

| Most common container ID | Samples |
| --- | --- |
| chunk_ivhi_c96f9088_000398 | 755 |
| chunk_ivhi_c96f9088_000349 | 755 |
| chunk_ivhi_c96f9088_000265 | 752 |
| chunk_ivhi_c96f9088_000281 | 752 |
| chunk_ivhi_c96f9088_000305 | 751 |

All rows were checked for finite WER, selected WER range, valid nonnegative integer error counts, positive scoring denominator, agreement between cached WER and `n_err/n_ref` within 1e-8, and an audio reference. The first 100 rows were independently rescored with matching error counts and denominators. The training-data pipeline was exercised on ten full records, checking ground truth, audio references, Hindi hints, and verbatim targets.

## Highest-WER sample statistics

Top 30 by cached WER. Full reference/hypothesis text is available in the linked review artifacts; line numbers are 1-based positions in the original JSONL.

| Source line | Container ID | WER (%) | Errors | Normalized reference units | Raw reference words | Raw hypothesis words |
| --- | --- | --- | --- | --- | --- | --- |
| 240992 | chunk_ivhi_c96f9088_000096 | 6750.0000% | 405 | 6 | 3 | 236 |
| 233115 | chunk_ivhi_c96f9088_000005 | 5033.3333% | 302 | 6 | 3 | 151 |
| 183562 | chunk_ivhi_c96f9088_000022 | 5000.0000% | 50 | 1 | 1 | 33 |
| 224932 | chunk_ivhi_c96f9088_000168 | 3629.4118% | 617 | 17 | 14 | 2 |
| 157988 | chunk_ivhi_c96f9088_000160 | 3137.5000% | 251 | 8 | 5 | 251 |
| 174140 | chunk_ivhi_c96f9088_000301 | 2000.0000% | 20 | 1 | 1 | 13 |
| 23922 | chunk_ivhi_c96f9088_000295 | 1942.8571% | 408 | 21 | 13 | 212 |
| 82719 | chunk_ivhi_c96f9088_000119 | 1700.0000% | 17 | 1 | 1 | 10 |
| 66573 | chunk_ivhi_c96f9088_000429 | 1550.0000% | 31 | 2 | 1 | 22 |
| 44605 | chunk_ivhi_c96f9088_000320 | 1166.6667% | 35 | 3 | 2 | 23 |
| 141157 | chunk_ivhi_c96f9088_000249 | 1009.0909% | 111 | 11 | 4 | 54 |
| 93708 | chunk_ivhi_c96f9088_000402 | 875.0000% | 35 | 4 | 2 | 35 |
| 82721 | chunk_ivhi_c96f9088_000119 | 733.3333% | 22 | 3 | 2 | 13 |
| 269582 | chunk_ivhi_c96f9088_000122 | 700.0000% | 14 | 2 | 1 | 8 |
| 255018 | chunk_ivhi_c96f9088_000169 | 700.0000% | 7 | 1 | 1 | 5 |
| 144098 | chunk_ivhi_c96f9088_000089 | 700.0000% | 21 | 3 | 1 | 16 |
| 1457 | chunk_ivhi_c96f9088_000422 | 700.0000% | 21 | 3 | 2 | 15 |
| 96924 | chunk_ivhi_c96f9088_000112 | 675.0000% | 27 | 4 | 2 | 19 |
| 32168 | chunk_ivhi_c96f9088_000218 | 666.6667% | 20 | 3 | 2 | 12 |
| 120647 | chunk_ivhi_c96f9088_000103 | 650.0000% | 13 | 2 | 2 | 13 |
| 5942 | chunk_ivhi_c96f9088_000280 | 650.0000% | 13 | 2 | 1 | 11 |
| 282037 | chunk_ivhi_c96f9088_000281 | 633.3333% | 19 | 3 | 2 | 13 |
| 97934 | chunk_ivhi_c96f9088_000414 | 600.0000% | 6 | 1 | 1 | 4 |
| 94014 | chunk_ivhi_c96f9088_000396 | 600.0000% | 6 | 1 | 1 | 3 |
| 93721 | chunk_ivhi_c96f9088_000402 | 600.0000% | 12 | 2 | 2 | 8 |
| 19586 | chunk_ivhi_c96f9088_000079 | 600.0000% | 12 | 2 | 2 | 9 |
| 108143 | chunk_ivhi_c96f9088_000054 | 550.0000% | 11 | 2 | 1 | 7 |
| 232083 | chunk_ivhi_c96f9088_000072 | 540.0000% | 108 | 20 | 8 | 47 |
| 260873 | chunk_ivhi_c96f9088_000057 | 500.0000% | 5 | 1 | 1 | 5 |
| 249920 | chunk_ivhi_c96f9088_000011 | 500.0000% | 5 | 1 | 1 | 3 |

## Detailed artifacts

These relative links point to local analysis artifacts under `tmp/`; those files may not be present in a fresh checkout. The aggregate tables above are self-contained.

- [data_errors_le3.manifest.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/data_errors_le3.manifest.json).
- [error_count_descending_10_buckets.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/error_count_descending_10_buckets.csv).
- [error_count_descending_cumulative.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/error_count_descending_cumulative.csv).
- [error_count_distribution.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/error_count_distribution.csv).
- [error_count_histogram.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/error_count_histogram.json).
- [error_count_quantiles.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/error_count_quantiles.csv).
- [highest_wer_samples.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/highest_wer_samples.csv).
- [identity_checks.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/identity_checks.json).
- [report.html](../../../../../../tmp/indic_voice_2609r2_wer0001_200/report.html).
- [sample_10.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/sample_10.json).
- [sample_10_total_errors_p50.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/sample_10_total_errors_p50.json).
- [summary.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/summary.json).
- [total_errors_subsets.json](../../../../../../tmp/indic_voice_2609r2_wer0001_200/total_errors_subsets.json).
- [wer_by_reference_length.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/wer_by_reference_length.csv).
- [wer_distribution.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/wer_distribution.csv).
- [word_count_distribution.csv](../../../../../../tmp/indic_voice_2609r2_wer0001_200/word_count_distribution.csv).

Initial summary, distributions, identity checks, and HTML were also published under `az://orngwus2cresco/data/boren/data/verl/analysis/indic_voice_2609r2_wer0001_200/`. The local artifact inventory additionally contains later descending-error statistics and subset manifests.

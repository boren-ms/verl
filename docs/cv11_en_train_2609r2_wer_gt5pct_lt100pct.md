# CV11 English TRAIN: strict 5% < WER < 100%

The augmented CV11 English training split is filtered using original 2609r2 step 54000 predictions, English normalized compound-aware WER, scoring version 2609. Retain exactly `100*n_err > 5*n_ref and n_err < n_ref`; both boundaries are excluded. Original transcripts, IDs, speed/RIR/noise provenance and Blob WAV pointers are preserved. Responses are audit evidence, not replacement training labels.

| Statistic | Full augmented source | Strict filtered subset |
|---|---:|---:|
| Clips | 948,736 | 438,730 |
| Retention | 100% | 46.243634% |
| Audio hours | 1042.252720 | 502.104074 |
| Original transcript words | 9,722,696 | 4,688,995 |
| Mean words/clip | 10.248052 | 10.687655 |
| Mean duration, seconds | 3.954851 | 4.120016 |
| Median duration, seconds | 3.755000 | 3.924531 |
| Corpus WER, percent | 55.859186 | 42.628631 |

Words mean whitespace-separated tokens in the original preserved transcripts. The normalized scoring reference counts are separate. The filtered WER range is 5.263158%–96.774194%. The 2 retained responses at the original 768-token limit remain flagged.

## Published artifacts

Root: `az://orngwus2cresco/data/boren/data/verl/gen_filtered/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1_wer_gt5pct_lt100pct/`

- `data.jsonl`: preserved augmented audio and ground-truth labels plus baseline error measurements.
- `training.yaml`: data loader config with statistics comments.
- `remax_training.yaml`: ReMax config matching the original URGENT E4 settings.
- `summary.json`: full counts, distributions, generation paths and pinned source/model identity.
- `COMPLETE.json`: full-source join and uploaded-manifest hash verification.

Filtered JSONL SHA256: `f83ee0828c524370686ddcab781033345aa4c30e6a0595fff048b7aded56e956`.

Canonical configs:

- `recipe/phimm/config/data/train_data/noisy/cv11_en_train_rir_snr0_12_2609r2_wer_gt5pct_lt100pct_verb_langhint.yaml`
- `recipe/phimm/config/v2611_dev/urgent/remax_cv11_wer_gt5pct_lt100pct_base0b_v0_e4.yaml`

The composed ReMax recipe differs from the existing CV11 WER ≥1% config only in training data/source dispatch and experiment/output names. It retains original 2609r2 step 54000, rank320/alpha640, batch128, rollout n16, 4 epochs, rewards and URGENT2024 delta validation. The strict-range ReMax variant has not been launched.

## Reproduction

```bash
python scripts/filter_cv11_augmented_wer.py \
  --filter-mode wer_gt5pct_lt100pct \
  --generation-dir az://orngwus2cresco/data/boren/data/verl/gen_qwen/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1 \
  --source-jsonl az://orngwus2cresco/data/boren/data/verl/gen_cached/cv11_en_train_speed_rir_noise_snr0_12_but_v1/data.jsonl \
  --output-dir az://orngwus2cresco/data/boren/data/verl/gen_filtered/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1_wer_gt5pct_lt100pct \
  --work-dir /root/data/cv11_wer_gt5pct_lt100pct_20261107 \
  --expected-rows 948736 \
  --training-yaml recipe/phimm/config/data/train_data/noisy/cv11_en_train_rir_snr0_12_2609r2_wer_gt5pct_lt100pct_verb_langhint.yaml \
  --remax-yaml recipe/phimm/config/v2611_dev/urgent/remax_cv11_wer_gt5pct_lt100pct_base0b_v0_e4.yaml
```

Use a fresh output root to rebuild; the filter rejects an existing completion marker.

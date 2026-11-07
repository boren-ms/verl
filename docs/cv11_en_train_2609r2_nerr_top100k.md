# CV11 English TRAIN: top 100,000 by baseline word-error count

Select the first 100,000 rows after sorting baseline `n_err` descending, breaking ties by the original candidate row index ascending. Candidates are the completed 438,730-row augmented CV11 English TRAIN subset with strict **5% < WER < 100%**, scored by original 2609r2 step54000 and English normalized compound-aware WER version 2609. The ranking uses absolute word errors, not the WER ratio.

| Statistic | Selected top 100,000 |
|---|---:|
| Clips | 100,000 |
| Audio hours | 110.235445 |
| Original transcript words | 1,207,376 |
| Mean words/clip | 12.073760 |
| Mean duration, seconds | 3.968476 |
| Median duration, seconds | 3.755813 |
| Total baseline word errors | 1,000,364 |
| Mean baseline word errors/clip | 10.003640 |
| Baseline word-error count range | 7–36 |
| Corpus baseline WER, percent | 81.191325 |

The cutoff is **n_err = 7**. There are 26,068 candidates at that cutoff, of which 1,270 are retained in source order. Original ground-truth labels, audio pointers, and speed/RIR/noise metadata remain unchanged. The JSONL remains sorted; the training recipe retains its existing shuffle setting.

## Artifacts

Root: `az://orngwus2cresco/data/boren/data/verl/gen_filtered/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1_wer_gt5pct_lt100pct_nerr_top100k/`

Contains `data.jsonl`, `training.yaml`, `remax_training.yaml`, `summary.json`, `COMPLETE.json`, `validation_loader.json`, `provenance.json`, and `README.md`. Full candidate-manifest SHA256 and row count are verified before publication; the entire uploaded JSONL is read back and hashed before completion evidence.

Candidate JSONL SHA256: `f83ee0828c524370686ddcab781033345aa4c30e6a0595fff048b7aded56e956`.

Top-100k JSONL SHA256: `311c349a70c1309c3ab20211efd27282dca89d9bf923a43b59f7e98be1366cfc`.

Canonical configs:

- `recipe/phimm/config/data/train_data/noisy/cv11_en_train_rir_snr0_12_2609r2_wer_gt5pct_lt100pct_nerr_top100k_verb_langhint.yaml`
- `recipe/phimm/config/v2611_dev/urgent/remax_cv11_nerr_top100k_base0b_v0_e4.yaml`

The ReMax recipe retains original 2609r2 step54000, rank320/alpha640, batch128, rollout n16, four epochs, rewards, and URGENT validation. Only training data/source dispatch and experiment/output names differ. The expected absolute target is `floor(100000/128)*4 = 3124` optimizer updates, subject to runtime dataloader confirmation.

192 evenly spaced retained rows passed the actual training loader, original labels/audio/prompts/reward-source checks, recomputed baseline WER and inherited strict threshold, and original WAV SHA256/header/signal checks.

## Execution

Dataset selection ran on the CPU of `verl-new-n2-i3-0`, with the existing CV11 training kept running. Source/runtime training code was unchanged by the auxiliary script/config sync. A slow development-machine manifest selection was terminated after matching its exact process, before any output publication; it was replaced by the in-region remote build.

The independent `team-critical` pool `verl-new-n2-i5` in `prod-westus2-cw-6`, namespace `boren`, is resumed but currently awaiting capacity. New training has not been submitted yet. Existing CV11 training `raysubmit_hx3KNjWS2DJi7cVE` remains tracked separately on `verl-new-n2-i3` with target 22812.

Execution and validation evidence: `tmp/run_openasr_exp/remax_cv11_nerr_top100k_base0b_v0_e4_20261107/`.

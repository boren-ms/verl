# Augmented CV11 English train filtered by 2609r2 WER >= 1%

## Completed: 2026-11-06 (America/Los_Angeles)

Inference completed successfully as Ray submission `raysubmit_idCe44XeuqbM5v4j`:
**948,736 responses**, all **1,853 batches**, and **47** published JSONL parts.
The full source/generation join verified every ID, original transcript, audio
pointer, decoded duration, measurement, and the pinned source SHA256.

The inclusive WER filter retains **730,071 clips (76.95197%)** and removes
**218,665 clips**. Original labels and all augmentation metadata are preserved.

| Statistic | Full augmented source | Retained WER >= 1% |
| --- | ---: | ---: |
| Clips | 948,736 | 730,071 |
| Audio hours | 1,042.252720 | 771.917163 |
| Original transcript words | 9,722,696 | 7,481,486 |
| Mean words per clip | 10.248052 | 10.247614 |
| Median words per clip | 11 | 11 |
| Mean duration per clip | 3.954851 s | 3.806345 s |
| Median duration per clip | 3.755000 s | 3.601500 s |
| Corpus WER of the fixed 2609r2 baseline | 55.85919% | 72.54963% |

The retained corpus WER describes selection of harder examples from the same
baseline responses. All 5,519,678 word errors remain; normalized reference words
decrease from 9,881,415 to 7,608,141. The 20 responses that hit the 768-token limit
remain flagged in the manifest; no upper WER or length-based filtering is added.

`validation_loader.json` records **192 evenly spaced retained samples** checked
through the actual training loader. Audio paths, original labels, English prompts,
empty prefixes, reward source, formatted references, re-computed baseline WER,
inclusive threshold, and cached original WAV SHA256/header/signal checks passed.
The full uploaded filtered manifest SHA256 was independently read back before
publishing `COMPLETE.json`:
`1a5679e97f622b8ce1e5d6d8eec73285aa79bf3fe3bf7c7f7a789be367c8e4aa`.

The Blob root below contains the completed manifest, data/training YAMLs,
summary, loader audit, README, and provenance. Completion cleanup is recorded
in pipeline state after durable result collection and final resource checks.

The inference/filtering stage uses the complete 948,736-row augmented CV11
English training manifest. It retains original transcripts and audio pointers,
including speed/RIR/noise provenance. The filtered count is established only
after generation finishes and every source row is verified.

Source:
`az://orngwus2cresco/data/boren/data/verl/gen_cached/cv11_en_train_speed_rir_noise_snr0_12_but_v1/data.jsonl`

Source SHA256:
`9243d564a8b1f97c52a22858fa95d6709a32b5cbb18c0c7d679ab2bb0a16767f`

Model: original **2609r2, step 54000**, matching URGENT Base0b E4:
`az://orngwus2cresco/data/speech/projects/phi-fastllm-2609/amlt-results/fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.4-r2/54000/qwen_hf`

Generation config:
`recipe/phimm/config/v2611_dev/gen/gen_cv11_en_train_rir_snr0_12_2609r2.yaml`.
One greedy response per example, temperature 0, BF16, no new LoRA adapter or
weight quantization, English language hint, no assistant prefix, no audio-duration
cutoff. The response length is 768 tokens; limit hits remain flagged.

Scoring uses `openasr_en_eval`, version 2609: parsed model text, HF English
normalization, and compound-aware word-error counts. The inclusive threshold is
**`100 * n_err >= n_ref`**, with no upper WER cap. Reference and hypothesis word
counts in reports use whitespace splitting; the WER denominator uses normalized
reference words, with the scorer's floor of one for an empty normalized reference.

Raw inference outputs:
`az://orngwus2cresco/data/boren/data/verl/gen_qwen/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1/part-*.jsonl`

Filtered dataset root:
`az://orngwus2cresco/data/boren/data/verl/gen_filtered/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1_wer_ge1pct/`

The completed root contains `data.jsonl`, `training.yaml`, `remax_training.yaml`,
`summary.json`, and `COMPLETE.json`. The completion marker is written after the
uploaded manifest's full SHA256 matches its local producer output.

Data YAML:
`recipe/phimm/config/data/train_data/noisy/cv11_en_train_rir_snr0_12_2609r2_wer_ge1pct_verb_langhint.yaml`.

ReMax YAML, under the requested URGENT folder:
`recipe/phimm/config/v2611_dev/urgent/remax_cv11_base0b_v0_e4.yaml`.
It retains the original E4 model, rank 320/alpha 640, batch 128, 16 rollouts,
four epochs, reward weights, and URGENT delta validation. Its training source
and reward dispatch map use `noisy_cv11_en_train_2609r2_wer_ge1pct`.

The filter validates contiguous generation parts, complete ordered source
coverage, unique IDs, unchanged transcripts/audio pointers, decoded audio
duration, measurement validity, and the pinned source checksum before publishing.

After generation succeeds:

```bash
python scripts/filter_cv11_augmented_wer.py \
  --generation-dir az://orngwus2cresco/data/boren/data/verl/gen_qwen/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1 \
  --source-jsonl az://orngwus2cresco/data/boren/data/verl/gen_cached/cv11_en_train_speed_rir_noise_snr0_12_but_v1/data.jsonl \
  --output-dir az://orngwus2cresco/data/boren/data/verl/gen_filtered/v2611/2609r2/cv11_en_train_speed_rir_noise_snr0_12_but_v1_wer_ge1pct \
  --work-dir /root/data/cv11_2609r2_wer_ge1pct_20261106 \
  --expected-rows 948736 \
  --training-yaml recipe/phimm/config/data/train_data/noisy/cv11_en_train_rir_snr0_12_2609r2_wer_ge1pct_verb_langhint.yaml \
  --remax-yaml recipe/phimm/config/v2611_dev/urgent/remax_cv11_base0b_v0_e4.yaml
```

Execution state, exact Ray submission, effective configs and completion evidence
are retained under `tmp/run_openasr_exp/cv11_2609r2_wer_ge1pct_20261106/`.
The dedicated pool is `verl-new-n2-i3`, cluster `prod-westus2-cw-6`, namespace
`boren`, two H100 nodes with eight GPUs each. The ReMax config is prepared for
handoff; this request does not submit a training job.

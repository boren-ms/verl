# Augmented CV11 English train filtered by 2609r2 WER >= 1%

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

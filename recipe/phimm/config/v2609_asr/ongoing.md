# Ongoing ASR experiments

## Pending training

The four variants created on 2026-10-06 are listed below as untrained.
This is a tracking list only; it does not submit training jobs.

| Recipe | Advantage mask | Actor loss aggregation | Status |
| --- | --- | --- | --- |
| [avg4_1stdiff](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff.yaml) | `first_diff` | `token-mean` | Pending training |
| [avg4_1stdiff_smts](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff_smts.yaml) | `first_diff` | `seq-mean-token-sum` | Pending training |
| [avg4_1stdiff_smtm](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff_smtm.yaml) | `first_diff` | `seq-mean-token-mean` | Pending training |
| [avg4_edit_smts](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_edit_smts.yaml) | `edit_boundary` | `seq-mean-token-sum` | Pending training |

All filenames share the prefix
`remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_`.
`smts` means sequence-mean token-sum; `smtm` means sequence-mean token-mean.

## Shared settings

- Algorithm: ReMax.
- Training steps: 400.
- Prompt batch size: 128.
- Sampled responses per prompt: 8.
- LoRA rank: 256.
- Resources: 4 nodes, 8 GPUs per node.
- Data schedule: four earnings epochs followed by one OpenML epoch per grouped pass.
- Reward reduction: component sum divided by 4 for both data sources.
- Actor-side KL: enabled, coefficient 0.001.

The original [avg4 source recipe](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4.yaml)
is the parent configuration, not an additional pending variant in this list.
See the [advantage-to-loss explanation](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff.md)
for mask behavior, aggregation formulas, and numerical examples.

Update each status when training is submitted or completed, and record the job
identifier and results when available.

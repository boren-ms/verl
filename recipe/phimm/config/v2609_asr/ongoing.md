# Ongoing 2609r2 ASR experiments

Updated: 2026-10-06. Tracks all five `2609r2` recipe YAMLs.

| Recipe | Advantage mask | Loss aggregation | Status |
| --- | --- | --- | --- |
| [avg4](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4.yaml) | None (`null`) | `token-mean` | No local job record |
| [avg4_1stdiff](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff.yaml) | `first_diff` | `token-mean` | Pending training |
| [avg4_1stdiff_smts](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff_smts.yaml) | `first_diff` | `seq-mean-token-sum` | Pending training |
| [avg4_1stdiff_smtm](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff_smtm.yaml) | `first_diff` | `seq-mean-token-mean` | Pending training |
| [avg4_edit_smts](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_edit_smts.yaml) | `edit_boundary` | `seq-mean-token-sum` | Pending training |

## Shared settings

- ReMax; 400 steps; prompt batch 128; 8 sampled responses per prompt.
- LoRA rank 256; 4 nodes with 8 GPUs each.
- Data: four earnings epochs, then one OpenML epoch per grouped pass.
- Reward: component sum divided by 4; actor-side KL coefficient 0.001.

## Notes

- `smts`: sequence-mean token-sum; `smtm`: sequence-mean token-mean.
- The original `avg4` has no entry in the local [job list](../verl_job.txt).
  Live training status has not been verified.
- Update statuses and add job IDs/results when runs are submitted or completed.
- See the [advantage-to-loss explanation](./remax_2609r2_earning_ml_verb_hint_s400_bs128_n8_r256_g32_smp8_avg4_1stdiff.md)
  for mask behavior, aggregation formulas, and examples.

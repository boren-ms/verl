#!/bin/bash
# python ./wandb_result.py --metric val-aux search 'eval_phimm'
# python ./wandb_result.py --metric val-aux search 'grpo_prod_fy22_7b_bs64_rep_gt_t12_tis5_2k|dapo_prod_fy22_7b_rare05_gt_t12_tis5_n8_wer_20_2k|grpo_tts_entity_7b_gt_t12_tis5_n16_2k_mw10|grpo_tts_entity_sr_fy22_7b_gt_t12_tis5_n16_2k_mw10'
# python ./wandb_result.py --metric val-aux search '[dapo|grpo]_prod_fy22_7b_rare01' 2198
python ./wandb_result.py --metric val-aux search 'entity' 2198


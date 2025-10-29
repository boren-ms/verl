#!/bin/bash

# bash submit_job.sh l-n2-hpe2 eval_phimm_3b
# bash submit_job.sh l-n2-hpe2 eval_phimm_7b
# bash submit_job.sh l-n2-hpe2 eval_phimm_rlbr
# bash submit_job.sh l-n1-uks7 eval_phimm_bias_sft

# bash submit_job.sh h-n2-wus2 grpo_ls_bias_bs64_rep_2k
# bash submit_job.sh h1-n2-hpe4 dapo_prod_fy22_bs64_rep_me10_2k
# bash submit_job.sh h1-n2-hpe4  dapo_ls_bias_bs64_rep_2k
# bash submit_job.sh h1-n2-wus2 dapo_prod_fy22_bs64_rep_2k
# bash submit_job.sh h1-n2-wus2  dapo_ls_bias_bs64_rep_e2


# bash submit_job.sh dev-n1-uks7 dapo_prod_fy22_bs64_2k
# bash submit_job.sh h-n1-uks7 dapo_rare_fy23q2_bs64_rep

# bash submit_job.sh h-n2-wus2  grpo_ls_bias_bs64_rep_2k true # dryrun, will cleanup
# bash submit_job.sh h-n2-wus2  dapo_ls_bias_bs64_rep_2k_n16
# bash submit_job.sh h-n2-uks7  dapo_ls_bs64_2k_v1 true
# bash submit_job.sh h-n2-uks7  dapo_ls_bs64_2k_w_cache
# bash submit_job.sh h1-n2-hpe4 dapo_ls_bias_bs64_rep_2k true
# bash submit_job.sh h1-n2-hpe4 grpo_ls_bias_bs64_rep_2k

# bash submit_job.sh l-n1-uks7 grpo_ls_bias_bs64_rep_2k_t12
# bash submit_job.sh l1-n1-hpe2 grpo_ls_bias_bs64_rep_2k_gt_t12
# bash submit_job.sh h-n2-uks7 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5
# bash submit_job.sh l-n1-uks7 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_raw
# # bash submit_job.sh l-n1-uks7 grpo_ls_bias_bs64_rep_2k_gt_t12_b16
# # bash submit_job.sh h1-n2-wus2 dapo_ls_bias_bs64_rep_e2 true
# bash submit_job.sh h1-n2-wus2 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_n16


# bash submit_job.sh h-n2-uks7 dapo_ls_bs64_2k_eb true
# bash submit_job.sh h-n2-uks7 grpo_ls_bs64_2k_eb
# bash submit_job.sh h1-n2-hpe4 grpo_ls_bias_bs64_rep_2k true
# bash submit_job.sh h-n2-wus2 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_dr

# bash submit_job.sh h-n1-uks7 dapo_rare_fy23q2_bs64_rep  true
# bash submit_job.sh h-n1-uks7 grpo_prod_fy22_bs64_rep_gt_tis5_2k
# bash submit_job.sh dev-n1-uks7 dapo_prod_fy22_bs64_2k true
# bash submit_job.sh dev-n1-uks7 grpo_prod_fy22_bs64_gt_tis5_2k

# bash submit_job.sh l-n1-uks7 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_raw true
# bash submit_job.sh h1-n2-hpe4 grpo_ls_eb_bs64_rep_2k_gt_t12_tis5_n8
# bash submit_job.sh l-n1-hpe2 grpo_ls_bs64_gt_t12_tis5_2k

# bash submit_job.sh dev1-n1-wus2 grpo_prod_fy22_7b_bs64_rep_gt_t12_tis5_2k
# bash submit_job.sh dev-n1-wus2 grpo_ls_bias_bs64_rep_2k_gt_t12_b16
# bash submit_job.sh dev-n1-wus2 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_sm

# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_bs64_gt_t12_tis5_eb_n8_2k
# bash submit_job.sh h-n2-hpe4 grpo_ls_eb_bs64_t12_tis5_n2_2k
# bash submit_job.sh h-n2-hpe4 grpo_ls_eb_bs64_t12_tis5_n2_4k


# bash submit_job.sh dev-n1-uks7 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_n32
# bash submit_job.sh l1-n1-hpe2 grpo_bias_7b_prod_fy22_rep_gt_t12_tis5_n16_2k
# bash submit_job.sh h-n2-hpe4 grpo_ls_bias_bs64_rep_gt_t12_tis5_n16_e2
# bash submit_job.sh l-n1-hpe2 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_n2x4
# bash submit_job.sh h1-n2-wus2 dapo_prod_fy22_bs64_rep_2k true
# bash submit_job.sh h1-n2-wus2 dapo_prod_fy22_bs64_gt_tis5_me10_n16_2k
# bash submit_job.sh l-n1-uks7 grpo_ls_eb_bs64_rep_2k_gt_t12_tis5_n2
# bash submit_job.sh h-n2-hpe4 grpo_ls_eb_bs64_t12_tis5_n2_4k true
# bash submit_job.sh h-n2-hpe4 grpo_ls_rare05_gt_t12_tis5_n8_2k

# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_me10_2k
# bash submit_job.sh l1-n1-hpe2 grpo_ls_bias_bs64_rep_gt_t12_tis5_n16_e2 true
# bash submit_job.sh h-n1-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_2k_v1
# bash submit_job.sh l1-n1-hpe2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_2k_v1 true
# bash submit_job.sh h-n2-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_2k
# bash submit_job.sh h-n2-hpe4 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_n2x4

# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_8k

# bash submit_job.sh h-n2-wus2 dapo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4

# bash submit_job.sh dev-n1-hpe2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_2k
# bash submit_job.sh h1-n2-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_2k_simple
# bash submit_job.sh h-n2-hpe4 grpo_ls_bias_bs64_rep_2k_gt_t12_tis5_n2x4 true
# bash submit_job.sh h1-n2-wus2 gen_prod_fy22_phi4_7b_wer_20
# bash submit_job.sh h-n2-hpe4 gen_prod_fy22_phi4_7b_wer_01_20
# bash submit_job.sh h1-n2-hpe4 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_a0_2k
# bash submit_job.sh h-n2-uks7 grpo_bias_7b_prod_fy22_rep_gt_t12_tis5_n2x4_2k
# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_8k 
# bash submit_job.sh dev-n1-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_wer_20_2k
# bash submit_job.sh h-n2-hpe4 gen_prod_fy22_phi4_7b_wer_01_20 true
# bash submit_job.sh h1-n2-hpe4 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_a0_2k
# bash submit_job.sh h1-n2-hpe4 dapo_prod_fy22_bs64_rep_me10_2k true
# bash submit_job.sh h1-n2-hpe4 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_wer_20_2k
# bash submit_job.sh h-n2-hpe4 gen_sr_fy23q2_phi4_7b_wer_20
# bash submit_job.sh h-n2-hpe4 dapo_prod_fy22_7b_rare05_gt_t12_tis5_n8_wer_20_2k
# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_8k

# bash submit_job.sh dev-n1-hpe2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_2k
# bash submit_job.sh h-n2-wus2 grpo_bias_7b_prod_fy22_wer_20_rep_gt_t12_tis5_2k

# bash submit_job.sh dev1-n1-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_2k
# bash submit_job.sh l-n1-uks7 grpo_entity_7b_gt_t12_tis5_n32_1k

# bash submit_job.sh h1-n2-wus2 grpo_tts_entity_7b_gt_t12_tis5_n16_2k_mw10
# bash submit_job.sh h1-n2-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_2k_mw05
# bash submit_job.sh l-n1-uks7 grpo_tts_entity_7b_t12_tis5_n16_s100

# bash submit_job.sh h-n2-uks7 dapo_prod_fy22_7b_rare05_n16_mw05_2k_pe3 true
# bash submit_job.sh h-n2-uks7 dapo_prod_fy22_7b_rare02_n4_mw05_8k
# bash submit_job.sh dev-n1-wus2 dapo_prod_fy22_7b_rare05_n8_mw05_2k
# bash submit_job.sh h-n2-wus2 gen_sr_fy23q2_phi4_7b_wer_01_20 true
# bash submit_job.sh h-n2-wus2 dapo_prod_fy22_7b_rare02_n4_mw05_8k_lr01
# bash submit_job.sh h1-n2-hpe4 dapo_prod_fy22_7b_rare05_n16_mw02_2k
# bash submit_job.sh dev-n1-uks7 dapo_prod_fy22_7b_rare02_n16_mw05_2k
# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_8k true
# bash submit_job.sh h-n2-uks7 grpo_tts_entity_sr_fy22_7b_gt_t12_tis5_n16_2k_mw10_tts80

# bash submit_job.sh h-n2-uks7 grpo_bias_7b_prod_fy22_rep_gt_t12_tis5_n2x4_2k true
# bash submit_job.sh h-n2-uks7 dapo_tts_entity_7b_gt_t12_tis5_n16_2k_mw10
# bash submit_job.sh h1-n2-hpe4 grpo_tts_entity_sr_fy22_7b_gt_t12_tis5_n32_4k_mw10 true
# bash submit_job.sh h1-n2-hpe4 dapo_tts_entity_7b_gt_t12_tis5_n16_2k_mw10_pe3
# bash submit_job.sh l-n1-hpe2 grpo_tts_entity_sr_fy22_7b_gt_t12_tis5_n16_2k_mw10

# bash submit_job.sh l-n1-hpe2     dapo_prod_fy22_7b_rare05_n8_mw05_2k_gmpo
# bash submit_job.sh l1-n1-hpe2     dapo_prod_fy22_7b_rare05_n8_mw05_2k_kl001
# bash submit_job.sh h1-n2-wus2     grpo_prod_fy22_7b_rare05_gt_t12_tis5_n4_me10_4k
# bash submit_job.sh h-n1-uks7     dapo_prod_fy22_7b_rare01_n16_mw05_2k
# bash submit_job.sh h-n2-hpe4     grpo_prod_fy22_7b_rare01_n8_mw05_2k
# bash submit_job.sh h-n2-hpe4     grpo_prod_fy22_7b_rare01_n8_mw05_2k_remax

# bash submit_job.sh h1-n2-uks7     grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_me10_8k
# bash submit_job.sh h1-n2-uks7     grpo_prod_fy22_7b_rare01a_n8_mw05_2k
bash submit_job.sh h1-n2-hpe4     grpo_prod_fy22_7b_rare05a_gt_t12_tis5_n8_2k
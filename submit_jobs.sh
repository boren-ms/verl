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
# bash submit_job.sh h-n2-wus2 gen_sr_fy23q2_phi4_7b_wer_01_20
# bash submit_job.sh h-n2-uks7 grpo_bias_7b_prod_fy22_rep_gt_t12_tis5_n2x4_2k 
# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_8k 
# bash submit_job.sh h-n1-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_wer_01_20_2k
# bash submit_job.sh dev-n1-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_wer_20_2k
# bash submit_job.sh h-n2-hpe4 gen_prod_fy22_phi4_7b_wer_01_20 true
# bash submit_job.sh h-n2-hpe4 gen_sr_fy23q2_phi4_7b_wer_20
# bash submit_job.sh h1-n2-hpe4 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_a0_2k
# bash submit_job.sh h1-n2-hpe4 dapo_prod_fy22_bs64_rep_me10_2k true
# bash submit_job.sh h1-n2-hpe4 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_wer_20_2k
# bash submit_job.sh h-n2-hpe4 dapo_prod_fy22_7b_rare05_gt_t12_tis5_n8_wer_20_2k
# bash submit_job.sh h-n2-uks7 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2_8k

bash submit_job.sh dev-n1-hpe2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n2x4_2k
bash submit_job.sh dev1-n1-wus2 grpo_prod_fy22_7b_rare05_gt_t12_tis5_n8_2k

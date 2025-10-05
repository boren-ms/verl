#!/bin/bash

# bash submit_job.sh l-n1-uks7 recipe/phimm/config/dapo_ls_bs64.yaml & # background job
# bash submit_job.sh h-n2-uks7 recipe/phimm/config/dapo_ls_bias_bs64_2k.yaml & # background job
# bash submit_job.sh l-n1-hpe2 recipe/phimm/config/dapo_prod_fy22_bs64_2k.yaml &
bash submit_job.sh h1-n2-hpe4 recipe/phimm/config/dapo_ls_bias_bs64_rep_2k.yaml &
bash submit_job.sh h1-n2-hpe4 recipe/phimm/config/dapo_prod_fy22_bs64_rep_me10_2k.yaml &
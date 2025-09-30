#!/bin/bash
set -xe

# rel_path=librispeech_biasing/sunit_filter/20250919
rel_path=ckp/hf_models/phi4_mm_bias_merged
src_host=h100_03
data_path="/home/boren/data/${rel_path}"

mkdir -p $data_path
# rsync -avz $src_host:$data_path/*.* $data_path/

bbb sync $data_path/ az://orngwus2cresco/data/boren/data/${rel_path}
bbb sync $data_path/ az://orngcresco/data/boren/data/${rel_path}
bbb sync $data_path/ az://orngscuscresco/data/boren/data/${rel_path}
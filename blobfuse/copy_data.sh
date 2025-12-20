#!/bin/bash
set -x

get_sas() {
    local cfg_file=/home/boren/blobfuse/$1.cfg
    local yaml_file=/home/boren/blobfuse/$1.yaml
    if [[ -f "$cfg_file" ]]; then
        grep sasToken "$cfg_file" | awk '{print $2}'
        elif [[ -f "$yaml_file" ]]; then
        grep "sas: " "$yaml_file" | awk '{print $2}'
    else
        echo "File not found: $cfg_file"
    fi
}

# src_storage=stdstoragetts01wus2
# src_storage=tsstd01wus2
src_storage=tsstd01wus2
src_blob=data

# dst_storage=tsstd01safn
dst_storage=highperf01safn
dst_blob=data
src_sas=$(get_sas ${src_storage}_${src_blob})
dst_sas=$(get_sas ${dst_storage}_${dst_blob})

# dst_storage=speechdatacrawlrgwusd353
# dst_blob=ingestioninput
# dst_sas="sp=racwdli&st=2025-04-01T18:01:35Z&se=2025-04-07T02:01:35Z&skoid=976202ce-0885-45ee-a2d6-acf916be0381&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-04-01T18:01:35Z&ske=2025-04-07T02:01:35Z&sks=b&skv=2024-11-04&spr=https&sv=2024-11-04&sr=c&sig=w0yDlKkzHDI6cwfpBBCWPrumsEkvKhTRkWzdlK4WmKY%3D"
# src_rel_path=projects/boren/amlt-results/7260938103.41953-6eaca5c5-2c3d-4dc0-b6f3-0b8221e798d4/25000/mp_rank_00_model_states.pt
# dst_rel_path=projects/boren/amlt-results/PhiS-SqaV0/25000/mp_rank_00_model_states.pt
# # export src_rel_path=v-litfen/customer/extractdata
# # export dst_rel_path=users/boren/text/tts/extractdata

# https://tsstd01uks.blob.core.windows.net/amulet/projects/phimm/amlt-results/7261045517.23695-d19a1453-dea3-4d4b-9e82-a60db2f8ce1d/540000/

# tag=Gaming_43k
# # tag=K12_Higher_Education_36k
# # tag=Retail_26k
# # tag=Science_And_Tech_48k
# # tag=Insurance_16k
# src_rel_path="/home/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3"
# dst_rel_path="users/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3"
# src_rel_path="/home/boren/data/librispeech_biasing"
# dst_rel_path="users/boren/data/librispeech_biasing"
# src_rel_path="/home/boren/data/inhouse/entity_data/entity_chunk"
# dst_rel_path="users/boren/data/inhouse/entity_data/entity_chunk"
src_rel_path="/home/boren/data/ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf"
dst_rel_path="users/boren/data/ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf"
# src_rel_path="/home/boren/data/LibriSpeech/asr_train_transcribe.tsv"
# dst_rel_path="users/boren/data/LibriSpeech/asr_train_transcribe.tsv"
# src_rel_path="/home/boren/data/ckp/hf_models/phi4-7b-fast-api-s2-final-v4"
# dst_rel_path="users/boren/data/ckp/hf_models/phi4-7b-fast-api-s2-final-v4"
# dst_rel_path=/home/boren/data/inhouse/entity_data/entity_chunk/${tag}/base_chunk/


azcopy cp --recursive=true --overwrite=false \
"${src_rel_path}/*" \
"https://${dst_storage}.blob.core.windows.net/${dst_blob}/${dst_rel_path}?${dst_sas}"

# azcopy cp --recursive=true --overwrite=false --include-pattern "*.audio;*.transcription;*.json"  \
# "https://${src_storage}.blob.core.windows.net/${src_blob}/${src_rel_path}?${src_sas}" \
# ${dst_rel_path}


# /home/boren/
# src_rel_path="users/v-benzhao/generated_tts_data/20251121/wave2_chunk"
# dst_rel_path="users/v-benzhao/generated_tts_data/20251121/wave2_chunk"
# azcopy cp --recursive=true --overwrite=true \
#     "https://${src_storage}.blob.core.windows.net/${src_blob}/${src_rel_path}/*?${src_sas}" \
#     "https://${dst_storage}.blob.core.windows.net/${dst_blob}/${dst_rel_path}?${dst_sas}"


# dst_dir=/home/boren/data/tts_gen/egs/
# mkdir -p $dst_dir

# src_rel_path=run_metadata/1e3c6b0a-bd47-4ad1-8513-855c3fdba054/split_filenames_result/
# azcopy cp --recursive=true --overwrite=true \
#     "https://${src_storage}.blob.core.windows.net/${src_blob}/${src_rel_path}/*?${src_sas}" $dst_dir

# src_path=/home/boren/Phi-4-multimodal-instruct
# dst_rel_path=users/boren/data/hf_models
# azcopy cp --recursive=true --overwrite=true $src_path/ \
#     "https://${dst_storage}.blob.core.windows.net/${dst_blob}/${dst_rel_path}/?${dst_sas}"

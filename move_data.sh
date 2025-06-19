
DATA_DIR=/root/data

bbb sync --concurrency 64  az://orngscuscresco/data/boren/data/gsm8k ${DATA_DIR}/gsm8k
# bbb sync --concurrency 64  az://orngscuscresco/data/boren/data/ckp/Qwen ${DATA_DIR}/ckp/Qwen
bbb sync --concurrency 64  az://orngscuscresco/data/boren/data/ckp/hf_models ${DATA_DIR}/ckp/hf_models

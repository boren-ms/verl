#!/bin/bash
echo "Syncing data from Azure Blob Storage to local machine"
bbb sync az://orngwus2cresco/data/boren/data/gsm8k/ ~/data/gsm8k/
bbb sync az://orngwus2cresco/data/boren/data/ckp/hf_models/Qwen2.5-Math-7B/ ~/data/ckp/hf_models/Qwen2.5-Math-7B/
bbb sync az://orngwus2cresco/data/boren/data/ckp/hf_models/Qwen2.5-0.5B-Instruct/ ~/data/ckp/hf_models/Qwen2.5-0.5B-Instruct/
bbb sync az://orngwus2cresco/data/boren/data/ckp/hf_models/Qwen3-0.6B/ ~/data/ckp/hf_models/Qwen3-0.6B/

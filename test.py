# %%

from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk

output_path = Path("/home/boren/data/gsm8k/test.parquet")
# %%
# df = pd.read_parquet(data_path)
# Load local Parquet file as HuggingFace Dataset
ds = Dataset.from_parquet(str(output_path))
ds = ds.take(100)
new_data_path = output_path.with_stem(output_path.stem + "_h100")
ds.to_parquet(new_data_path)
df = pd.read_parquet(new_data_path)
# %%

output_path = Path("/home/boren/data/outputs/r1_distill_qwen_aime24/test-output-k1.parquet")
ds1 = Dataset.from_parquet(str(output_path))
print(ds1[0])


# %%
ds_path = Path("/home/boren/data/cache_datasets/ls_sc1k_fn1/")


def update_path(x):
    audio_path = x["audio_path"]
    if audio_path.startswith("/root/"):
        audio_path = audio_path.replace("/root/", "/home/boren/")
    return {"audio_path": audio_path}


ds = load_from_disk(ds_path)
ds = ds.map(update_path)

# %%

parquet_path = Path("/home/boren/data/parquet/ls_sc1k_fn1.parquet")
ds.to_parquet(parquet_path)
# %%
n = 100
top_data_path = parquet_path.with_name(parquet_path.stem + f"_h{n}.parquet")
ds_top = ds.take(n)
print(f"Saving {n} elements to ", top_data_path)
ds_top.to_parquet(top_data_path)

# %%
parquet_path = Path("/home/boren/data/parquet/ls_sc1k_fn1.parquet")
ds = Dataset.from_parquet(str(parquet_path))


def update_prompt(x):
    prompt = x["prompt"]
    for token in ["<|user|>", "<|end|>", "<|assistant|>"]:
        prompt = prompt.replace(token, "").strip()
    return {"prompt": prompt}


output_path = parquet_path.with_name(parquet_path.stem + f"_h{n}.parquet")
# %%
n = 100
ds = ds.take(n)
ds = ds.map(update_prompt)
print("Updating ", output_path)
ds.to_parquet(output_path)
# %%
# %%

from pathlib import Path
from pprint import pprint

from datasets import Dataset

output_path = Path("/home/boren/data/parquet/dapo-math-17k.parquet")
output_path = Path("/home/boren/data/parquet/aime-2024.parquet")
output_path = Path("/home/boren/data/parquet/gsm8k/train.parquet")
# output_path = Path("/root/data/parquet/ls_sc1k_fn1.parquet")
output_path = Path("/root/data/parquet/ls_clean_sc1k_fn1_h100.parquet")
# output_path = Path("/root/data/parquet/ls_sc1k_fn1_h100_root.parquet")

ds = Dataset.from_parquet(str(output_path))

print(ds)
egs = ds[10]
pprint(egs)

# %%
src_path = "/home/boren/"
dst_path = "az://orngcresco/data/boren/"


def map_path(egs):
    return {"audio_path": egs["audio_path"].replace(src_path, dst_path)}


ds = ds.map(map_path)
print(ds)
egs = ds[10]
pprint(egs)
ds.to_parquet(output_path.with_stem(output_path.stem + "_remote"))

# %%


# %%
import transformers
from transformers import AutoModelForCausalLM
import vllm
from pathlib import Path

print("vLLM:", getattr(vllm, "__version__", "?"))
print("Transformers:", transformers.__version__)

# %%# %%
# %%
model_id = Path("/home/boren/data/ckp/hf_models/phi4_mm_bias_merged/")
model = AutoModelForCausalLM.from_pretrained(
    str(model_id),
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)


# %%
# Convert config to regular Python types before creating PEFT model
from peft import LoraConfig, get_peft_model

lora_config = {
    "task_type": "CAUSAL_LM",
    "r": 32,
    "lora_alpha": 32,
    # "target_modules": "all-linear",
    "target_modules": "model.layers",
    # "target_modules": ".*qkv_proj.*",
    # "target_modules": "(qkv_proj|o_proj|gate_up_proj|down_proj)",
    # "target_modules": [
    #     "qkv_proj",
    #     "o_proj",
    #     "gate_up_proj",
    #     "down_proj",
    # ],
    "bias": "none",
}
peft_model = get_peft_model(model, LoraConfig(**lora_config))

# %%
with open("peft_v1.log", "w") as f:
    for name, param in peft_model.named_parameters():
        print(f"{name}: {param.shape}", file=f)

# %%

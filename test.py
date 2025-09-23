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
output_path = Path("/home/boren/data/outputs/ls_sc1k_fn1_h100/ls_sc1k_fn1_h100.parquet")
ds = Dataset.from_parquet(str(output_path))
from pprint import pprint

print(ds)
egs = ds[10]
pprint(egs)


# %%

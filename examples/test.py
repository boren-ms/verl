#%%
from datasets import load_dataset

# Replace 'path/to/your/file.parquet' with your actual Parquet file path
ds = load_dataset("parquet", split="train", data_files="/home/boren/data/gsm8k/train.parquet")

print(ds)
print(ds[0])

# %%

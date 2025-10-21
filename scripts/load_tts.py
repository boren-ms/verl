#%%
from pathlib import Path

#%%

data_dir = Path("/home/boren/data/inhouse/GeneratedAudio/Entity/RetailEntity")

for audio_file in data_dir.glob("*.wav"):
    text_file = audio_file.with_suffix(".txt")
    trans = text_file.read_text().strip() if text_file.exists() else None
    print(f"{audio_file}: {trans}")
#%%


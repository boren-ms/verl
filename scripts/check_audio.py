#%%
from pathlib import Path
import soundfile as sf

data_dir= Path("/home/boren/data/inhouse/GeneratedAudio/Entity/AudioEntityV2")

failed_files = []
for wav_path in data_dir.glob("*.wav"):
    try:
        data, samplerate = sf.read(wav_path)
    except Exception as e:
        failed_files.append(wav_path)
        print(e)
        continue
    # print(f"{wav_path}: {len(data)/samplerate}s, samplerate={samplerate}")
print(f"Failed files ({len(failed_files)}):")
for wav_path in sorted(failed_files):
    print(wav_path)

# %%

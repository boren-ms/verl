#%%
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
#%%
from pathlib import Path
data_dir = Path("/home/boren/data/ckp/hf_models/")
model_path = data_dir / "whisper-turbo-v3" # whisper-turbo-v3-fp8
llm = LLM(
    model=str(model_path),
    max_model_len=448,
    max_num_seqs=256,
    gpu_memory_utilization=0.5,
    limit_mm_per_prompt={"audio": 1},
)
#%%
audio_path = "/home/boren/data/LibriSpeech/train-clean-100/103/1240/103-1240-0002.flac"

import soundfile as sf
data, sr = sf.read(audio_path)

inputs = {
    "encoder_prompt": {
        "prompt": "",                          # (optional) text to condition the encoder
        "multi_modal_data": {
            "audio": [(data, sr)],  # list of (audio_data: np.ndarray, sample_rate: int)
        },
    },
    "decoder_prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",  # pick target lang token(s)
}


out = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=128))
print(out[0].outputs[0].text)
# trans: for not even a brook could run past mrs rachel linsdore without due regard for decency and decorum it probably was conscious that mrs rachel was sitting at her window keeping a sharp eye on everything that passed from brooks and children up

# %%

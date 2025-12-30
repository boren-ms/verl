# %%
from vllm.lora.request import LoRARequest
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from soundfile import read as sf_read


# %%


model_path = Path("/home/boren/data/ckp/hf_models/Phi-4-multimodal-instruct")
# model_path = Path("/home/boren/data/ckp/hf_models/phi4_mm_bias")

# %%
# HuggingFace model name

speech_lora_path = model_path / "speech-lora"
# Create the LLM object with LoRA adaptation
llm = LLM(
    model=str(model_path),
    trust_remote_code=True,
    max_model_len=8192,
    max_num_seqs=2,
    enable_lora=True,
    tensor_parallel_size=2,
    limit_mm_per_prompt={"audio": 1},
    gpu_memory_utilization=0.5,
    max_lora_rank=320,
)
# %%
# prompt = "What is the capital of France?"
# outputs = llm.generate([prompt], SamplingParams(temperature=1, max_tokens=640))
# for output in outputs:
#     print(output.outputs[0].text)
# %%
sampling_params = SamplingParams(temperature=1, max_tokens=8192)
prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
audios = ["/home/boren/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
inputs = [{"prompt": prompt, "multi_modal_data": {"audio": [sf_read(audio_path)]}} for prompt, audio_path in zip(prompts, audios)]

outputs = llm.generate(inputs, sampling_params=sampling_params)
texts = [output.outputs[0].text for output in outputs]
text = texts[0]
print(text)
# %%
# params = BeamSearchParams(max_tokens=8192, beam_width=5)

# prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
# audios = ["/home/boren/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
# inputs = [{"prompt": prompt, "multi_modal_data": {"audio": [sf_read(audio_path)]}} for prompt, audio_path in zip(prompts, audios)]

# outputs = llm.beam_search(inputs, params=params)
# texts = [output.outputs[0].text for output in outputs]
# text = texts[0]
# print(text)
# %%
# text_prompt = "Please capture the text and output it in <result> <text></text>"
prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
text_prompt = "Transcribe the audio clip into text."
inputs[0]["prompt"] = f"<|user|><|audio_1|>{text_prompt}<|end|><|assistant|>"
sampling_params = SamplingParams(temperature=1, max_tokens=8192)
lora_request = [LoRARequest("speech", 1, str(speech_lora_path))]
outputs = llm.generate(inputs, sampling_params=sampling_params, lora_request=lora_request)
texts = [output.outputs[0].text for output in outputs]
text = texts[0]
print(text)
# %%
if __name__ == "__main__":
    # vLLM is compatible with spawn; this avoids the RuntimeError
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

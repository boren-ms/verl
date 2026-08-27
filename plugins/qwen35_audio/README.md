# vLLM Qwen3.5-Audio Plugin

Out-of-tree vLLM plugin for running **Qwen3.5-Audio** through vLLM's official model implementation.

## Verified Stack

The working stack verified on `verl-n1-i11` is:

| Package | Version |
| --- | --- |
| vLLM | `0.17.0` |
| PyTorch | `2.10.0+cu128` |
| CUDA runtime | `12.8` |
| Transformers | `5.7.0` |
| flashinfer-python | `0.6.4` |
| flashinfer-cubin | `0.6.4` |

Hardware used for verification: 8x NVIDIA H100 80GB HBM3 with tensor parallel size 8.

`ray_tool.py prepare_env` installs the full environment including vLLM, flashinfer,
Transformers and the qwen35_audio plugin with `--no-deps` overrides so Verl HF
workers can load checkpoints with native `qwen3_5_text` support.

## Installation

```bash
cd plugins/qwen35_audio
uv pip install -r requirements-vllm-0.17.txt --torch-backend=auto
uv pip install -e .
```

The plugin is registered through the `vllm.general_plugins` entry point as `qwen35_audio`.

## Smoke Test

The vLLM smoke test defaults to the verified raw HuggingFace `az://` bundle for fresh-node reproduction:

- model: `az://orngwus2cresco/data/speech/projects/phi-fastllm-2605/amlt-results/fast-llm-2605-qwen3-5-9b-s2-st-example/90000/qwen_hf/`
- audio: `az://orngwus2cresco/data/boren/data/LibriSpeech/train-clean-360/115/122944/115-122944-0036.flac`

The script stages `az://` inputs into `/root/data/qwen35_audio_test/` with `bbb` before loading vLLM.

```bash
cd plugins/qwen35_audio
QWEN35_AUDIO_DISABLE_CUDNN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_PLUGINS=qwen35_audio \
PYTHONPATH=src \
python scripts/run_qwen35_audio_vllm.py
```

Use `--model` and repeatable `--audio` arguments to point at different `az://` or local paths. Use `--audio-folder` to recursively decode supported audio files from a local directory while loading the model only once. Use `--local-cache-root` to change the staging directory.

```bash
# vLLM: submits all discovered files in one generate call.
python scripts/run_qwen35_audio_vllm.py \
    --model <model-path> \
    --audio-folder <audio-directory>

# Hugging Face: loads once, then generates sequentially for each file.
python scripts/run_qwen35_audio_hf.py \
    --model <model-path> \
    --audio-folder <audio-directory>
```

Both scripts emit one `AUDIO_RESULT_START`/`AUDIO_RESULT_END` block per file and finish with `BATCH_DONE count=<N>`.
Shared argument parsing, input staging, folder discovery, cache naming, and audio loading live in `scripts/qwen35_audio_utils.py`.

The verified run produced:

```text
Audio Language: English
<ASR><lang=English><TXT>The advocates of a criminal are seldom artists enough to turn the beautiful terribleness of the deed to the advantage of the doer.</TXT></ASR>
```

## Basic Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="path/to/qwen35-audio-hf",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=1,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.15,
    limit_mm_per_prompt={"audio": 1},
)

prompt = (
    "<|im_start|>user\n<|audio_start|>\n"
    "Detect the language and transcribe the audio into text.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

outputs = llm.generate(
    [{"prompt": prompt, "multi_modal_data": {"audio": [(waveform, sample_rate)]}}],
    sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop_token_ids=[248044, 248046],
    ),
)
print(outputs[0].outputs[0].text)
```

## Model Checkpoint Format

The raw checkpoint must be HuggingFace-compatible with:

```json
{
    "architectures": ["Qwen3_5AudioForCausalLM"],
    "model_type": "qwen3_5_audio"
}
```

The vLLM script passes the architecture through `hf_overrides`, and the plugin normalizes the language backbone to `qwen3_5_text` internally before constructing vLLM's Qwen3.5 decoder.

## Compatibility Shims

vLLM `0.17.0` includes `vllm.model_executor.models.qwen3_5` and `vllm.model_executor.models.qwen3_5_audio`, but the shipped Qwen3.5-Audio code expects a few internal symbols or signatures that are not exported in that release. The plugin installs narrow compatibility shims before importing the official class:

- `vllm.multimodal.profiling.BaseDummyInputsBuilder`
- `vllm.multimodal.inputs.MultiModalDataDict`
- selected exports from `vllm.multimodal.processing.processor`
- mamba state classmethods on `Qwen3_5ForCausalLM`
- optional `mm_options` support for `Qwen3_5AudioDummyInputsBuilder.get_dummy_mm_data`

`QWEN35_AUDIO_DISABLE_CUDNN=1` disables cuDNN before registration. This avoided a cuDNN initialization failure in the audio encoder path on the verified H100 node.

## Troubleshooting

### `ModuleNotFoundError: No module named 'vllm.multimodal.profiling'`

Use this plugin with `VLLM_PLUGINS=qwen35_audio`. vLLM `0.17.0` through `0.20.2` were checked and did not ship `vllm.multimodal.profiling`; the plugin supplies a local module alias for the official Qwen3.5-Audio import.

### `flashinfer-cubin version ... does not match flashinfer version ...`

Install matching FlashInfer packages. The verified pair is:

```bash
uv pip install flashinfer-python==0.6.4 flashinfer-cubin==0.6.4
```

### `undefined symbol` from `flash_attn_2_cuda`

Remove ABI-incompatible `flash-attn` builds from the environment. The verified run did not require `flash-attn`.

### `libtorch_cuda.so: undefined symbol: ncclDevCommDestroy`

This was seen with newer vLLM/PyTorch stacks on the node. The verified workaround is to use `vllm==0.17.0` with the CUDA 12.8 stack.

## References

- [vLLM Multimodal Guide](https://docs.vllm.ai/en/latest/models/multimodal.html)
- [vLLM Plugin System](https://docs.vllm.ai/en/latest/plugins/)

# Qwen3.5-Audio model plugin

This package supplies the model implementation used by the Base3a ASR recipe:

- `hf_qwen35_audio` registers the audio config, causal LM and processor with
  Transformers. The text backbone uses Main's regular model engine; audio
  encoding is attached at the embedding boundary.
- `vllm_qwen35_audio` registers the model and multimodal processor through
  vLLM's `vllm.general_plugins` entry point, named `qwen35_audio`.
- Both runtimes retain the flash/cascade audio encoder implementations and
  checkpoint configuration support. These are model dependencies, even though
  Base3a's checkpoint selects the flash frontend.

Install into the matching GPU environment without replacing its dependencies:

```bash
uv pip install --no-deps -e plugins/qwen35_audio
```

The ASR recipe sets `model.external_lib=hf_qwen35_audio` and forwards
`VERL_USE_EXTERNAL_MODULES=hf_qwen35_audio` and `VLLM_PLUGINS=qwen35_audio`
through Main's Ray runtime configuration. It selects the installed NVIDIA CUDA
runtime matching PyTorch's CUDA major. The package initializes optional
FlashInfer IPC before the model imports TileLang, avoiding selection of an
incomplete CUDA stub during spawned-worker config unpickling.

Training and validation use the entry points documented in
[the ASR recipe](../phimm/README.md). Local config checks do not establish
compatibility with a particular GPU, vLLM or Transformers installation.

## Checkpoints

`src/hf_qwen35_audio/convert_verl_to_pt.py` remains available for the documented
post-training export/LoRA merge workflow. It reads all FSDP rank shards and
supports Azure inputs. Conversion of the original model and standalone
transcription verification scripts are outside this training package's scope.

## Validation

```bash
python -m pytest -q plugins/qwen35_audio/tests/test_audio_frontend.py
python -m pytest -q plugins/qwen35_audio/tests/test_cuda_import.py
```

The frontend test checks ragged feature padding, masks and contiguous tensor
layout. The CUDA test checks spawned config imports and both CUDA consumers
against the installed runtime. Both require the matching model/GPU dependency
stack; the Base3a refactor's CPU-only validation is recorded in
[REFACTOR.md](../phimm/REFACTOR.md).

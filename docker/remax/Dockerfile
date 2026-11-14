# syntax=docker/dockerfile:1.7
# Context: source/ (verl and plugin), packages/ (internal wheels).
ARG BASE_IMAGE=iridiumsdc.azurecr.io/rcall:crow-1676263
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
# The curated ReMax stack is checked below and on a fresh GPU pool.
ENV SKIP_TORCHFLOW_SETUP_CHECK=1
ENV PATH=/root/.pyenv/versions/3.12.9/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}

# Pin Orange-compatible Ray; update protobuf within 5.x for vLLM 0.32.
RUN <<'BASH'
mkdir -p /opt/verl-env
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
assert version("protobuf") == "5.29.5", "Unexpected base-image protobuf"
Path("/opt/verl-env/constraints.txt").write_text(
    "ray==2.54.0\nprotobuf==5.29.6\n"
    "torch==2.13.0+cu130\ntorchaudio==2.11.0+cu130\n"
    "torchvision==0.28.0+cu130\nnumpy==2.2.6\n"
)
Path("/opt/verl-env/overrides.txt").write_text(
    # Match the upstream cu130 x86_64 resolver overrides in pyproject.toml.
    "tilelang==0.1.15\nnvidia-cusparselt-cu13==0.8.0\n"
)
Path("/opt/verl-env/base-ray-version.txt").write_text(version("ray"))
PY
BASH

# Mount wheels only during installation; keep credentials outside the image.
RUN --mount=type=bind,source=packages,target=/root/packages <<'BASH'
python -m pip install uv
# Resolve the complete vLLM dependency graph, including its newer tokenizer,
# FlashInfer and serving dependencies. Constraints preserve Orange's Ray.
# TorchAudio 2.11 is its final release, also pinned by upstream Main/vLLM.
export UV_CONSTRAINT=/opt/verl-env/constraints.txt
export UV_OVERRIDE=/opt/verl-env/overrides.txt
tar -C / -xzf /root/packages/cuda-compat-13-0-580.95.05.tgz
printf '%s\n' /usr/local/cuda-13.0/compat > /etc/ld.so.conf.d/00-oai-cuda-compat.conf
ldconfig
uv pip install --system \
    'https://download.pytorch.org/whl/cu130/torch-2.13.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl' \
    'https://download.pytorch.org/whl/cu130/torchvision-0.28.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl' \
    'https://download.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl'
uv pip uninstall --system flashinfer-cubin flashinfer-jit-cache
uv pip install --system \
    vllm==0.32.0 'ray[default]==2.54.0' protobuf==5.29.6 \
    transformers==5.16.1 tokenizers==0.23.2 \
    torchdata==0.11.0 accelerate==1.14.0 datasets==4.0.0 peft==0.19.1 \
    liger-kernel==0.8.4 tensordict==0.10.0 numpy==2.2.6 \
    flashinfer-python==0.7.0.post1 apache-tvm-ffi==0.1.11 \
    'nvidia-cutlass-dsl[cu13]==4.8.0' tilelang==0.1.15 \
    flash-linear-attention==0.5.2 fla-core==0.5.2 ml-dtypes==0.5.4 \
    TransferQueue==0.1.11 pyzmq msgspec \
    cupy-cuda13x==14.0.1 trl==0.27.0 kernels==0.14.1 \
    sentencepiece==0.2.1 kaldi-native-fbank rignore hydra-core==1.3.3 \
    pandas 'pyarrow>=19.0.0,<26.0.0' pybind11 pylatexenc \
    codetiming dill wandb 'packaging>=20.0' \
    latex2sympy2_extended math_verify mathruler qwen-vl-utils blobfile scipy soundfile \
    more-itertools whisper_normalizer opencc-python-reimplemented==0.1.7 wetext==0.2.1 \
    pytest pytest-asyncio pytest-rerunfailures 'jiwer>=4.0.0' kaldialign voi-oiwer==0.1.4 editdistance \
    num2words evaluate tensorboard bitsandbytes==0.50.2

# FlashInfer JIT requires a coherent CUDA compiler/runtime generation.
python -m pip install \
    nvidia-cuda-runtime==13.3.29 nvidia-cuda-nvcc==13.3.73 \
    nvidia-cuda-crt==13.3.73 nvidia-nvvm==13.3.73 nvidia-cuda-nvdisasm==13.3.73 \
    nvidia-cuda-tileiras==13.3.36
cuda_dir=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')
ln -sfn lib "$cuda_dir/lib64"
ln -sfn libcudart.so.13 "$cuda_dir/lib/libcudart.so"
# nvcc depends on unbounded nvidia-nvvm: a newer frontend can emit PTX that
# this assembler cannot read. Check the pinned pair before the full build.
printf '__global__ void verl_cuda_probe() {}\n' > /tmp/verl_cuda_probe.cu
"$cuda_dir/bin/nvcc" -arch=sm_90 -c /tmp/verl_cuda_probe.cu -o /tmp/verl_cuda_probe.o
rm /tmp/verl_cuda_probe.cu /tmp/verl_cuda_probe.o
uv pip install --system --no-deps --reinstall-package flash-attn \
    'https://github.com/verl-project/verl-wheelhouse/releases/download/flash-attention-v2.8.3-py3.12-torch2.13.0-cu130/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl'
uv pip install --system --no-index --find-links /root/packages/bitbank bitbank==0.2.0

# The old internal extension wheel targets Torch 2.11. Compile against this
# image's Torch/CUDA ABI instead of carrying that binary into Torch 2.13.
CUDA_HOME="$cuda_dir" PATH="$cuda_dir/bin:$PATH" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 \
    uv pip install --system --no-cache --reinstall-package causal-conv1d --no-deps --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.7.0
apt-get update
apt-get install -y --no-install-recommends lsof ffmpeg
rm -rf /var/lib/apt/lists/* /root/.cache/pip
BASH

COPY source/ /opt/verl-source/
WORKDIR /opt/verl-source
RUN <<'BASH'
python -m pip install --no-deps -e . -e plugins/phimm -e plugins/qwen35_audio
# Validate versions and extension loading; GPU/NCCL checks run on a fresh pool.
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
from packaging.version import Version
expected = {
    "torch": "2.13.0+cu130", "torchvision": "0.28.0+cu130", "torchaudio": "2.11.0+cu130", "vllm": "0.32.0", "transformers": "5.16.1",
    "flash-attn": "2.8.3", "flashinfer-python": "0.7.0.post1",
    "flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "causal-conv1d": "1.7.0",
    "tilelang": "0.1.15", "apache-tvm-ffi": "0.1.11", "ml-dtypes": "0.5.4",
    "nvidia-cutlass-dsl": "4.8.0",
    "numpy": "2.2.6", "tensordict": "0.10.0", "bitbank": "0.2.0",
    "TransferQueue": "0.1.11", "protobuf": "5.29.6", "bitsandbytes": "0.50.2",
    "voi-oiwer": "0.1.4", "nvidia-cuda-runtime": "13.3.29",
    "nvidia-cuda-nvcc": "13.3.73", "nvidia-cuda-crt": "13.3.73", "nvidia-nvvm": "13.3.73",
    "nvidia-cuda-nvdisasm": "13.3.73", "nvidia-cuda-tileiras": "13.3.36",
    "ray": "2.54.0", "opencc-python-reimplemented": "0.1.7", "wetext": "0.2.1",
    "liger-kernel": "0.8.4",
}
for package, pin in expected.items():
    assert Version(version(package)) == Version(pin), (package, version(package), pin)
import torch
import causal_conv1d_cuda
import flash_attn_2_cuda
from torchaudio.functional import resample
assert resample(torch.randn(1, 160), 16000, 8000).shape == (1, 80)
import hf_qwen35_audio
import vllm
from transfer_queue import BatchMeta, KVBatchMeta
# Candidate marker only: job startup must still perform its GPU checks.
Path("/opt/verl-env/image-ray-version.txt").write_text(version("ray"))
Path("/opt/verl-env/.env_done_torch2.13.0_vllm0.32.0_flashattn2.8.3_cu130_v1").touch()
PY
python -m pip freeze > /opt/verl-env/requirements.freeze.txt
rm -rf /root/.cache/pip
BASH

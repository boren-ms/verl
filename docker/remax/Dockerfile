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

# Pin Orange-compatible Ray before cluster startup; preserve protobuf 5.29.5.
RUN <<'BASH'
mkdir -p /opt/verl-env
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
assert version("protobuf") == "5.29.5", "Unexpected base-image protobuf"
Path("/opt/verl-env/constraints.txt").write_text(
    "ray==2.54.0\nprotobuf==5.29.5\n"
    "torch==2.11.0+cu130\ntorchaudio==2.11.0+cu130\n"
    "torchvision==0.26.0+cu130\nnumpy==1.26.4\n"
)
Path("/opt/verl-env/base-ray-version.txt").write_text(version("ray"))
PY
BASH

# Mount wheels only during installation; keep credentials outside the image.
RUN --mount=type=bind,source=packages,target=/root/packages <<'BASH'
export PIP_CONSTRAINT=/opt/verl-env/constraints.txt
python -m pip install --no-deps vllm==0.24.0
python -m pip install 'ray[default]==2.54.0'
tar -C / -xzf /root/packages/cuda-compat-13-0-580.95.05.tgz
printf '%s\n' /usr/local/cuda-13.0/compat > /etc/ld.so.conf.d/00-oai-cuda-compat.conf
ldconfig
python -m pip install numpy==1.26.4 'sympy>=1.13.3'
python -m pip install --no-index --find-links /root/packages/torch211-cu130 \
    'cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink]==13.0.2' \
    'cuda-bindings>=13.0.3,<14' \
    'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'

python -m pip install \
    torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 torchdata==0.11.0 \
    transformers==5.7.0 accelerate==1.14.0 datasets==4.0.0 peft==0.19.1 \
    safetensors==0.5.3 tokenizers==0.22.2 huggingface_hub==1.13.0 liger_kernel==0.8.0 \
    tensordict==0.10.0 'numpy<2.0.0' \
    flashinfer-python==0.6.12 flashinfer-cubin==0.6.12 \
    'nvidia-cutlass-dsl[cu13]==4.8.1' xgrammar==0.1.29 sentencepiece==0.2.1 \
    compressed-tensors depyf outlines_core partial-json-parser \
    lm-format-enforcer gguf mistral_common blake3 \
    cbor2 pydantic-extra-types kaldi-native-fbank rignore \
    fastar quack-kernels llguidance hydra-core==1.3.3 \
    pandas 'pyarrow>=19.0.0,<26.0.0' pybind11 pylatexenc \
    codetiming dill wandb 'packaging>=20.0' \
    uvicorn fastapi \
    latex2sympy2_extended math_verify blobfile scipy soundfile \
    more-itertools whisper_normalizer \
    'jiwer>=4.0.0' kaldialign voi-oiwer==0.1.4 editdistance \
    num2words evaluate tensorboard prometheus-fastapi-instrumentator \
    model-hosting-container-standards openai-harmony protobuf==5.29.5 bitsandbytes==0.50.2

# FlashInfer JIT requires a coherent CUDA compiler/runtime generation.
python -m pip install \
    nvidia-cuda-runtime==13.3.29 nvidia-cuda-nvcc==13.3.73 \
    nvidia-cuda-crt==13.3.73 nvidia-cuda-nvdisasm==13.3.73 \
    nvidia-cuda-tileiras==13.3.36
cuda_dir=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')
ln -sfn lib "$cuda_dir/lib64"
ln -sfn libcudart.so.13 "$cuda_dir/lib/libcudart.so"
python -m pip install --no-deps \
    /root/packages/transferqueue-0.1.11.dev0-py3-none-any.whl \
    /root/packages/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
python -m pip install --no-index --find-links /root/packages/bitbank bitbank==0.2.0

# Install attention backends without resolving Torch/CUDA dependencies.
python -m pip install --no-deps \
    fla-core==0.6.0 flash-linear-attention==0.6.0 \
    tilelang==0.1.9 apache-tvm-ffi==0.1.10 ml-dtypes==0.5.4

# This staged wheel was compiled for the supported Python/Torch/CUDA ABI.
python -m pip install --no-deps \
    /root/packages/causal-conv1d/1.7.0/fb3d8669723a4a161a5517ed/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
apt-get update
apt-get install -y --no-install-recommends lsof
rm -rf /var/lib/apt/lists/* /root/.cache/pip
BASH

COPY source/ /opt/verl-source/
WORKDIR /opt/verl-source
RUN <<'BASH'
python -m pip install --no-deps -e . -e plugins/qwen35_audio
# Validate versions and extension loading; GPU/NCCL checks run on a fresh pool.
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
from packaging.version import Version
expected = {
    "torch": "2.11.0+cu130", "vllm": "0.24.0", "transformers": "5.7.0",
    "flash-attn": "2.8.3", "flashinfer-python": "0.6.12", "flashinfer-cubin": "0.6.12",
    "flash-linear-attention": "0.6.0", "fla-core": "0.6.0", "causal-conv1d": "1.7.0",
    "tilelang": "0.1.9", "apache-tvm-ffi": "0.1.10", "ml-dtypes": "0.5.4",
    "nvidia-cutlass-dsl": "4.8.1",
    "numpy": "1.26.4", "tensordict": "0.10.0", "bitbank": "0.2.0",
    "TransferQueue": "0.1.11.dev0", "protobuf": "5.29.5", "bitsandbytes": "0.50.2",
    "voi-oiwer": "0.1.4", "nvidia-cuda-runtime": "13.3.29",
    "nvidia-cuda-nvcc": "13.3.73", "nvidia-cuda-crt": "13.3.73",
    "nvidia-cuda-nvdisasm": "13.3.73", "nvidia-cuda-tileiras": "13.3.36",
    "ray": "2.54.0",
}
for package, pin in expected.items():
    assert Version(version(package)) == Version(pin), (package, version(package), pin)
import torch
import causal_conv1d_cuda
import flash_attn_2_cuda
# Candidate marker only: job startup must still perform its GPU checks.
Path("/opt/verl-env/image-ray-version.txt").write_text(version("ray"))
Path("/opt/verl-env/.env_done_torch2.11.0_vllm0.24.0_flashattn2.8.3_cu130_v3").touch()
PY
python -m pip freeze > /opt/verl-env/requirements.freeze.txt
rm -rf /root/.cache/pip
BASH

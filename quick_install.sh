# pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl
# uv pip install --system torch==2.6.0 torchvision torchaudio transformers==4.51.3 datasets==4.0.0 trl peft tensorboardX blobfile soundfile more-itertools whisper_normalizer fire
# pip install vllm==0.8.5.post1 && pip install ray==2.46.0
# pip install torch==2.6.0 flash-attn
# pip install
# pip uninstall -y trl
pip install -r requirements_vllm.txt
pip install --no-deps -e .
apt install lsof

# pip install --resume-retries 999 --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Install flash-attn-2.7.4.post1, although built with torch2.6, it is compatible with torch2.7
# https://github.com/Dao-AILab/flash-attention/issues/1644#issuecomment-2899396361
# ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
# URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl" && \
# FILE="flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl" && \
# wget -nv "${URL}" && \
# pip install --no-cache-dir "${FILE}"

# Fix packages
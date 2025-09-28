pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl
uv pip install --system torch==2.6.0 torchvision torchaudio transformers==4.51.3 datasets==4.0.0 trl peft tensorboardX blobfile soundfile more-itertools whisper_normalizer fire
pip install vllm==0.8.5.post1 && pip install ray==2.46.0
pip install torch==2.6.0 flash-attn
pip install
pip uninstall -y trl
pip install -e /root/code/trl --no-deps
#!/usr/bin/env bash
# Bind this checkout to the curated image without modifying its runtime stack.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec 9>/tmp/verl-checkout-install.lock
flock 9
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
from packaging.version import Version

if not Path('/opt/verl-env/image-ray-version.txt').is_file():
    raise SystemExit('Use the verified docker/remax image before preparing this checkout.')
expected = {
    'ray': '2.54.0', 'torch': '2.11.0+cu130', 'vllm': '0.24.0',
    'transformers': '5.7.0', 'TransferQueue': '0.1.11.dev0',
    'opencc-python-reimplemented': '0.1.7', 'wetext': '0.2.1',
    'liger-kernel': '0.8.2', 'numpy': '2.2.6',
}
for package, pin in expected.items():
    actual = version(package)
    if Version(actual) != Version(pin):
        raise SystemExit(f'{package}: expected {pin}, found {actual}; rebuild the image.')
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() > 0, 'No GPUs allocated'
PY
echo '[INFO] Reusing baked dependencies; installing editable source only.'
uv pip install --system --no-deps --no-build-isolation -e . -e plugins/phimm -e plugins/qwen35_audio

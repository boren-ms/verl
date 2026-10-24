"""Run inside a fresh Orange GPU pod after setup; no models or datasets needed."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import torch

expected = {
    "ray": "2.54.0", "torch": "2.11.0", "vllm": "0.24.0", "transformers": "5.7.0",
    "flash-attn": "2.8.3", "flashinfer-python": "0.6.12",
    "flash-linear-attention": "0.6.0", "fla-core": "0.6.0",
    "causal-conv1d": "1.7.0", "protobuf": "5.29.5",
    "bitbank": "0.2.0", "tensordict": "0.10.0",
    "tilelang": "0.1.9", "apache-tvm-ffi": "0.1.10",
    "ml-dtypes": "0.5.4", "numpy": "1.26.4",
    "nvidia-cutlass-dsl": "4.8.1",
    "flashinfer-cubin": "0.6.12", "bitsandbytes": "0.50.2",
    "nvidia-cuda-runtime": "13.3.29", "nvidia-cuda-nvcc": "13.3.73",
    "nvidia-cuda-crt": "13.3.73", "nvidia-cuda-nvdisasm": "13.3.73",
    "nvidia-cuda-tileiras": "13.3.36",
}
versions = {name: metadata.version(name) for name in expected}
for name, version in versions.items():
    assert version.split("+")[0] == expected[name], (name, version)
baked_ray = Path("/opt/verl-env/image-ray-version.txt")
# Verify the runtime and worker versions; record the baked Ray version too.
image_ray = baked_ray.read_text().strip() if baked_ray.exists() else None
assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0
for device in range(torch.cuda.device_count()):
    x = torch.randn(64, 64, device=f"cuda:{device}", requires_grad=True)
    x.square().mean().backward()
    assert torch.isfinite(x.grad).all()

from causal_conv1d import causal_conv1d_fn
x = torch.randn(1, 8, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(8, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = causal_conv1d_fn(x, weight, activation="silu")
y.float().square().mean().backward()
assert torch.isfinite(y).all() and all(
    t.grad is not None and torch.isfinite(t.grad).all() for t in (x, weight)
)

from flash_attn import flash_attn_func
q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = flash_attn_func(q, q, q, causal=True)
y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(q.grad).all()

from flashinfer.norm import rmsnorm
x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
y = rmsnorm(x, weight, eps=1e-6)
reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
assert torch.isfinite(y).all()
torch.testing.assert_close(y.float(), reference, atol=0.03, rtol=0.03)

from transformers.models.qwen3_5 import modeling_qwen3_5 as native
assert native.is_fast_path_available
assert native.chunk_gated_delta_rule.__module__.startswith("fla.")
q = torch.randn(1, 16, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
g = torch.full((1, 16, 2), -0.1, device="cuda", dtype=torch.float32)
beta = torch.full((1, 16, 2), 0.5, device="cuda", dtype=torch.bfloat16)
y, _ = native.chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
y.float().square().mean().backward()
assert torch.isfinite(y).all()
assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
torch.cuda.synchronize()

import vllm
import ray
import bitbank
import verl
from verl.trainer.ppo import core_algos

assert hasattr(core_algos, "compute_remax_outcome_advantage")
ray.init(address="auto", log_to_driver=False)
@ray.remote
def worker_versions():
    import ray
    import google.protobuf
    import torch
    return {"ray": ray.__version__, "protobuf": google.protobuf.__version__, "torch": torch.__version__}
worker = ray.get(worker_versions.remote(), timeout=120)
assert worker == {"ray": ray.__version__, "protobuf": "5.29.5", "torch": torch.__version__}, worker
report = {
    "packages": versions, "ray_worker": worker, "image_ray": image_ray,
    "gpu_count": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(0),
    "torch_cuda": torch.version.cuda,
    "checks": ["all_gpu_backward", "causal_conv1d_backward", "flash_attention_backward", "flashinfer_rmsnorm", "fla_backward", "remax_import", "managed_ray_worker"],
    "verl_path": verl.__file__,
}
text = json.dumps(report, indent=2)
print(text)
Path(os.environ.get("VERL_VERIFY_REPORT", "/tmp/verl-env-verification.json")).write_text(text + "\n")
ray.shutdown()

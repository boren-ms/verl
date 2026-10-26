"""Check explicit runtime selection without importing vLLM or CUDA."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location(
    "cuda_runtime", Path(__file__).parents[4] / "verl/utils/vllm/cuda_runtime.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_explicit_runtime_overrides_stub_discovery_and_preserves_explicit_argument(monkeypatch):
    class Runtime:
        def __init__(self, so_file=None):
            self.path = so_file or "libcudart_stub.so"

    monkeypatch.setitem(
        sys.modules, "vllm.distributed.device_communicators.cuda_wrapper", SimpleNamespace(CudaRTLibrary=Runtime)
    )
    monkeypatch.delenv("VLLM_CUDART_SO_PATH", raising=False)
    module.configure_cuda_runtime()
    assert Runtime().path == "libcudart_stub.so"

    monkeypatch.setenv("VLLM_CUDART_SO_PATH", "/cuda/libcudart.so.13")
    module.configure_cuda_runtime()
    module.configure_cuda_runtime()
    assert Runtime().path == "/cuda/libcudart.so.13"
    assert Runtime("/explicit/libcudart.so").path == "/explicit/libcudart.so"

    monkeypatch.delenv("VLLM_CUDART_SO_PATH")
    assert Runtime().path == "libcudart_stub.so"

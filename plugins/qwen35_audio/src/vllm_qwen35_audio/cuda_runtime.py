"""Honor an explicit CUDA runtime when TileLang also loads a cudart stub."""

import os
from functools import wraps


def configure_cuda_runtime():
    if not os.environ.get("VLLM_CUDART_SO_PATH"):
        return

    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    original = CudaRTLibrary.__init__
    if getattr(original, "_verl_explicit_cudart", False):
        return

    @wraps(original)
    def initialize(self, so_file=None):
        # vLLM 0.24 otherwise takes the first libcudart substring in /proc/maps,
        # which can be TileLang's incomplete libcudart_stub.so.
        return original(self, so_file=so_file or os.environ.get("VLLM_CUDART_SO_PATH"))

    initialize._verl_explicit_cudart = True
    CudaRTLibrary.__init__ = initialize

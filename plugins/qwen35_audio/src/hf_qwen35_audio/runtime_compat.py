"""Import ordering for the optional vLLM/TileLang runtime."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def prepare_optional_flashinfer() -> bool:
    """Initialize FlashInfer before the affected bundled TVM is imported.

    With these versions, importing TileLang first makes FlashInfer's first
    TopKMaskLogits call fail with an invalid CUDA resource handle. Importing
    FlashInfer first passes the same Ray/FSDP and vLLM dummy sampler checks.
    This imports no model and leaves the sampler implementation unchanged.
    HF-only environments do not need the optional inference packages.
    """
    affected = {
        "vllm": "0.24.0",
        "flashinfer-python": "0.6.12",
        "tilelang": "0.1.9",
        "apache-tvm-ffi": "0.1.9",
    }
    try:
        if any(version(package) != expected for package, expected in affected.items()):
            return False
    except PackageNotFoundError:
        return False
    import_module("flashinfer")
    return True

"""Exercise the vLLM spawn/config-unpickle order with the installed CUDA stack."""

import os
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest


def test_spawned_config_binds_real_cuda_before_tilelang(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if not torch.version.cuda:
        pytest.skip("Requires a CUDA PyTorch installation")
    runtime = next(
        (Path(sysconfig.get_path("purelib")) / "nvidia").glob(
            f"*/lib/libcudart.so.{torch.version.cuda.split('.')[0]}"
        ),
        None,
    )
    if runtime is None:
        pytest.skip("Requires NVIDIA's installed CUDA runtime")
    script = tmp_path / "spawn_cuda_probe.py"
    script.write_text(textwrap.dedent('''\
        import multiprocessing
        import os
        import vllm.v1.executor.multiproc_executor

        def worker(config):
            import ctypes
            from flashinfer.comm.cuda_ipc import cudart
            from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
            from vllm_qwen35_audio.plugin import register

            register()
            expected = os.environ["VLLM_CUDART_SO_PATH"]
            assert "libcudart_stub" not in cudart.lib._name, cudart.lib._name
            # torchvision can supply another complete runtime with the same
            # CUDA major; FlashInfer has no explicit-library override.
            version = ctypes.c_int()
            assert cudart.lib.cudaRuntimeGetVersion(ctypes.byref(version)) == 0
            assert version.value // 1000 == int(expected.rsplit(".", 1)[1])
            assert os.path.samefile(CudaRTLibrary().lib._name, expected)
            assert config.model_type == "qwen3_5_audio"
            print("Spawned config, plugin and both CUDA runtimes passed", flush=True)

        if __name__ == "__main__":
            from hf_qwen35_audio import Qwen3_5AudioConfig

            process = multiprocessing.get_context("spawn").Process(
                target=worker, args=(Qwen3_5AudioConfig(),)
            )
            process.start()
            process.join(120)
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError("Spawned import did not finish")
            assert process.exitcode == 0, process.exitcode
    '''))
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "VLLM_CUDART_SO_PATH": str(runtime), "VLLM_PLUGINS": "qwen35_audio"},
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

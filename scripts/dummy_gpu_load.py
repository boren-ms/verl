#!/usr/bin/env python3
"""Run a disposable compute load on otherwise idle NVIDIA GPUs."""

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuState:
    index: str
    uuid: str
    utilization: int
    compute_pids: frozenset[int]


def run_nvidia_smi(*arguments: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_gpu_states() -> dict[str, GpuState]:
    process_output = run_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    )
    pids_by_uuid: dict[str, set[int]] = {}
    for line in process_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[1].isdigit():
            pids_by_uuid.setdefault(fields[0], set()).add(int(fields[1]))

    gpu_output = run_nvidia_smi(
        "--query-gpu=index,uuid,utilization.gpu", "--format=csv,noheader,nounits"
    )
    states = {}
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[2].isdigit():
            continue
        index, uuid, utilization = fields
        states[index] = GpuState(
            index=index,
            uuid=uuid,
            utilization=int(utilization),
            compute_pids=frozenset(pids_by_uuid.get(uuid, set())),
        )
    return states


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def gpu_is_idle(gpu_uuid: str) -> bool:
    utilization = run_nvidia_smi(
        f"--id={gpu_uuid}",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    processes = run_nvidia_smi(
        f"--id={gpu_uuid}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    )
    return utilization.strip() == "0" and not any(
        line.strip().isdigit() for line in processes.splitlines()
    )


def cuda_check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with CUDA status {status}")


def run_worker(gpu_uuid: str, spin_cycles: int) -> int:
    if not gpu_is_idle(gpu_uuid):
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    ptx = b"""
.version 7.0
.target sm_70
.address_size 64
.visible .entry spin(.param .u64 duration) {
    .reg .pred running;
    .reg .u64 start, now, deadline, duration_value;
    ld.param.u64 duration_value, [duration];
    mov.u64 start, %clock64;
    add.u64 deadline, start, duration_value;
loop:
    mov.u64 now, %clock64;
    setp.lt.u64 running, now, deadline;
    @running bra loop;
    ret;
}
"""

    cuda_check(cuda.cuInit(0), "cuInit")
    cuda_check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    cuda_check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
    try:
        cuda_check(
            cuda.cuModuleLoadDataEx(ctypes.byref(module), ptx, 0, None, None),
            "cuModuleLoadDataEx",
        )
        cuda_check(
            cuda.cuModuleGetFunction(ctypes.byref(function), module, b"spin"),
            "cuModuleGetFunction",
        )
        duration = ctypes.c_ulonglong(spin_cycles)
        kernel_params = (ctypes.c_void_p * 1)(
            ctypes.cast(ctypes.pointer(duration), ctypes.c_void_p)
        )
        while True:
            cuda_check(
                cuda.cuLaunchKernel(
                    function,
                    128,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    None,
                    kernel_params,
                    None,
                ),
                "cuLaunchKernel",
            )
            cuda_check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
    finally:
        if module:
            cuda.cuModuleUnload(module)
        cuda.cuCtxDestroy_v2(context)


def stop_worker(index: str, worker: subprocess.Popen[bytes]) -> None:
    log(f"GPU {index}: stopping dummy load (PID {worker.pid})")
    worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()


def monitor(interval: float, spin_cycles: int, dry_run: bool) -> int:
    workers: dict[str, subprocess.Popen[bytes]] = {}
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stopping:
            states = get_gpu_states()
            for index, worker in list(workers.items()):
                state = states.get(index)
                external_pids = state.compute_pids if state and len(state.compute_pids) > 1 else set()
                if worker.poll() is not None:
                    log(f"GPU {index}: dummy load exited with status {worker.returncode}")
                    del workers[index]
                elif state is None or external_pids:
                    if external_pids:
                        log(f"GPU {index}: multiple compute PID(s) detected: {sorted(external_pids)}")
                    stop_worker(index, worker)
                    del workers[index]

            for index, state in states.items():
                if index in workers:
                    continue
                if state.utilization == 0 and not state.compute_pids:
                    if dry_run:
                        log(f"GPU {index}: idle; would start dummy load")
                        continue
                    command = [
                        sys.executable,
                        os.path.abspath(__file__),
                        "--worker",
                        state.uuid,
                        "--spin-cycles",
                        str(spin_cycles),
                    ]
                    workers[index] = subprocess.Popen(command)
                    log(f"GPU {index}: started dummy load (PID {workers[index].pid})")
                elif dry_run:
                    log(
                        f"GPU {index}: busy ({state.utilization}% util, "
                        f"compute PIDs {sorted(state.compute_pids)}); skipping"
                    )

            if dry_run:
                return 0
            time.sleep(interval)
    finally:
        for index, worker in list(workers.items()):
            stop_worker(index, worker)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10, help="poll interval in seconds")
    parser.add_argument(
        "--spin-cycles",
        type=int,
        default=10_000_000,
        help="GPU clock cycles per dummy kernel launch",
    )
    parser.add_argument("--dry-run", action="store_true", help="report decisions without starting load")
    parser.add_argument("--worker", metavar="GPU_UUID", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.spin_cycles <= 0:
        parser.error("--spin-cycles must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.worker:
            return run_worker(args.worker, args.spin_cycles)
        return monitor(args.interval, args.spin_cycles, args.dry_run)
    except (FileNotFoundError, PermissionError) as error:
        log(f"error: cannot execute nvidia-smi: {error}")
        return 1
    except subprocess.CalledProcessError as error:
        log(f"error: nvidia-smi failed with status {error.returncode}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""Exercise actual cleanup/rotation code without importing CUDA/FSDP dependencies."""

import ast
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from filelock import FileLock
import pytest


ROOT = Path(__file__).resolve().parents[4]


def _method(filename, class_name, method_name):
    tree = ast.parse((ROOT / "verl/utils/checkpoint" / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def _code(nodes):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<actual-checkpoint>", "exec")


def _run_rank(rank, roots, remote, barrier, uploads, keep):
    cleanup = _method("checkpoint_manager.py", "BaseCheckpointManager", "remove_previous_save_local_path")
    ns = {"os": os, "shutil": shutil, "FileLock": FileLock}
    exec(_code([cleanup]), ns)
    manager = SimpleNamespace(rank=rank, previous_saved_paths=[])
    manager.remove_previous_save_local_path = ns[cleanup.name].__get__(manager)
    save = _method("fsdp_checkpoint_manager.py", "FSDPCheckpointManager", "save_checkpoint")
    # Run the production prefix including upload drain, barriers, rotation and mkdir.
    end = next(
        i for i, n in enumerate(save.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "self.should_save_model"
    )
    # local_path is always supplied here; omit its early-return guard.
    assert ast.unparse(save.body[1].test) == "local_path is None"
    prefix = _code(save.body[2:end])
    node = Path(roots[rank % len(roots)])
    previous = None
    for step in (10, 20, 30, 40):
        def drain(timeout, previous=previous, step=step):
            assert timeout == 1800
            if previous is not None:
                # A delayed upload must still find the previous local shard.
                time.sleep(0.01 * (rank % 4))
                source = node / f"global_step_{previous}/actor" / f"rank_{rank}.pt"
                dest = Path(remote) / f"global_step_{previous}" / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read_bytes())
            uploads[rank] = step

        def sync(step=step):
            barrier.wait(timeout=20)
            assert all(value == step for value in uploads)

        def mkdir(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            return path

        path = str(node / f"global_step_{step}/actor")
        exec(prefix, {"self": manager, "local_path": path, "global_step": step,
                      "max_ckpt_to_keep": keep, "wait_for_remote_uploads": drain,
                      "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=sync)),
                      "local_mkdir_safe": mkdir})
        (Path(path) / f"rank_{rank}.pt").write_text(str(step))
        manager.previous_saved_paths.append(path)
        previous = step
        barrier.wait(timeout=20)


@pytest.mark.parametrize("nodes,keep", [(2, 2), (1, 2), (2, None)])
def test_multirank_local_retention_preserves_uploaded_shards(tmp_path, nodes, keep):
    context = multiprocessing.get_context("spawn")
    roots = [str(tmp_path / f"node_{i}") for i in range(nodes)]
    remote = str(tmp_path / "remote_fixture")
    barrier = context.Barrier(8)
    uploads = context.Array("i", 8)
    processes = [context.Process(target=_run_rank, args=(rank, roots, remote, barrier, uploads, keep))
                 for rank in range(8)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=35)
            assert process.exitcode == 0
        for root in roots:
            steps = sorted(int(p.parent.name.removeprefix("global_step_")) for p in Path(root).glob("*/actor"))
            assert steps == ([30, 40] if keep else [10, 20, 30, 40])
        # Earlier saves remain durable after local rotation on both nodes.
        for step in (10, 20, 30):
            assert len(list((Path(remote) / f"global_step_{step}").glob("rank_*.pt"))) == 8
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

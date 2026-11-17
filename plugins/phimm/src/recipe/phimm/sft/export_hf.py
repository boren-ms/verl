"""Export a completed native SFT checkpoint through the synced repository converter."""

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import blobfile as bf
from omegaconf import OmegaConf


def _revision(repository):
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def _read_json(path):
    with bf.BlobFile(path, "r") as stream:
        return json.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--expected-producer-revision", required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-training-submission-id", required=True)
    parser.add_argument("--expected-experiment", required=True)
    parser.add_argument("--expected-base-model", required=True)
    parser.add_argument("--expected-lora-rank", required=True, type=int)
    parser.add_argument("--expected-lora-alpha", required=True, type=float)
    parser.add_argument("--expected-world-size", required=True, type=int)
    parser.add_argument("--expected-adapter-pairs", type=int)
    parser.add_argument("--expected-tensor-count", type=int)
    parser.add_argument("--preserve-lora-precision", action="store_true", required=True)
    parser.add_argument("--work-dir")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.step <= 0 or args.workers <= 0:
        parser.error("step and workers must be positive")
    for value in (args.expected_producer_revision, args.expected_execution_revision):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            parser.error("producer and execution revisions must be full commit SHA strings")

    repository = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    execution_revision = _revision(repository)
    assert execution_revision == args.expected_execution_revision, "Export execution checkout changed"
    root = args.checkpoint_root.rstrip("/")
    assert root.startswith("az://"), "Checkpoint root must identify the verified Azure producer"
    monitor_path = root + "/pipeline_evidence/training_monitor_latest.json"
    monitor = _read_json(monitor_path)
    assert monitor["submission_id"] == args.expected_training_submission_id
    assert monitor["status"] == "SUCCEEDED", monitor["status"]
    assert monitor["target_optimizer_step"] == args.step == monitor["last_completed_optimizer_step"]
    assert monitor["checkpoint"]["step"] == args.step
    assert monitor["checkpoint"]["complete_expected_files"] == args.expected_world_size * 3 + 2
    assert not monitor["nonfinite_metrics"]
    assert monitor["teacher_forced_validation"][-1]["step"] == args.step
    with bf.BlobFile(root + "/latest_checkpointed_iteration.txt", "r") as stream:
        assert int(stream.read().strip()) == args.step, "Final checkpoint tracker differs"
    with bf.BlobFile(root + "/effective_config.yaml", "rb") as stream:
        producer_raw = stream.read()
    config = OmegaConf.create(producer_raw.decode())
    assert config.trainer.experiment_name == args.expected_experiment
    assert config.model.lora_rank == args.expected_lora_rank
    assert config.model.lora_alpha == args.expected_lora_alpha
    assert config.trainer.nnodes * config.trainer.n_gpus_per_node == args.expected_world_size
    base = config.model.path.rstrip("/")
    assert base == args.expected_base_model.rstrip("/")
    checkpoint = f"{root}/global_step_{args.step}"
    metadata = _read_json(checkpoint + "/lora_train_meta.json")
    assert metadata["r"] == args.expected_lora_rank and metadata["lora_alpha"] == args.expected_lora_alpha
    inventory = []
    for name in ["data_0.pt", "lora_train_meta.json"] + [
        f"{kind}_world_size_{args.expected_world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(args.expected_world_size)
    ]:
        info = bf.stat(checkpoint + "/" + name)
        assert info.size > 0, name
        inventory.append({"name": name, "bytes": info.size, "mtime": info.mtime})
    remote = f"{root}/exports/step{args.step}/qwen_hf"
    work = Path(args.work_dir or f"/root/checkpoint_exports/{args.expected_experiment}/step{args.step}")
    assert not work.exists() and not bf.exists(remote), "Reconcile existing export before retrying"
    hf = work / "qwen_hf"
    hf.mkdir(parents=True)
    converter = repository / "plugins/qwen35_audio/src/hf_qwen35_audio/convert_verl_to_pt.py"
    converter_sha = hashlib.sha256(converter.read_bytes()).hexdigest()
    print(
        "FINAL_EXPORT_GATE=" + json.dumps({"step": args.step, "checkpoint": checkpoint, "files": len(inventory)}),
        flush=True,
    )
    with (work / "convert.log").open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--input",
                checkpoint,
                "--output",
                str(work / "model.pt"),
                "--local-cache",
                str(work / "shards"),
                "--match-lora-merged",
                "--preserve-lora-precision",
                "--lora-alpha",
                str(args.expected_lora_alpha),
                "--lora-rank",
                str(args.expected_lora_rank),
                "--workers",
                str(args.workers),
            ],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=repository,
        )
    matches = re.findall(r"Merged (\d+) LoRA adapter pairs", (work / "convert.log").read_text())
    assert len(matches) == 1 and int(matches[0]) > 0
    adapter_pairs = int(matches[0])
    if args.expected_adapter_pairs is not None:
        assert adapter_pairs == args.expected_adapter_pairs

    import hf_qwen35_audio  # noqa: F401 -- register the local model and processor
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch.set_num_threads(args.workers)
    state = torch.load(work / "model.pt", map_location="cpu", weights_only=False)
    clean = {}
    for key, tensor in state.items():
        name = key.replace(".base_layer.", ".")
        assert name not in clean and not any(tag in name for tag in ("lora_A", "lora_B"))
        assert isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(), key
        assert tensor.dtype == torch.bfloat16, key
        clean[name] = tensor.contiguous()
    save_file(clean, str(hf / "model.safetensors"), metadata={"format": "pt"})
    del state, clean
    with safe_open(hf / "model.safetensors", framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
        shapes = {key: list(stream.get_slice(key).get_shape()) for key in keys}
    if args.expected_tensor_count is not None:
        assert len(keys) == args.expected_tensor_count
    for name in bf.listdir(base):
        if name.endswith((".safetensors", ".bin", ".pt", ".index.json")) or bf.isdir(base + "/" + name):
            continue
        if name.endswith((".json", ".py", ".txt", ".model", ".tiktoken", ".jinja")) and name not in (
            "export_provenance.json",
            "final_pipeline_provenance.json",
        ):
            bf.copy(base + "/" + name, str(hf / name), overwrite=False)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        assert (hf / name).is_file(), name
    model, loading = AutoModelForCausalLM.from_pretrained(
        hf, trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True
    )
    assert not loading["missing_keys"] and not loading["unexpected_keys"] and not loading.get("mismatched_keys"), (
        loading
    )
    processor = AutoProcessor.from_pretrained(hf, trust_remote_code=False)
    assert processor is not None
    del model, processor
    assert _revision(repository) == execution_revision, "Execution source changed during export"
    assert hashlib.sha256(converter.read_bytes()).hexdigest() == converter_sha, "Converter changed during export"
    files = []
    for path in sorted(hf.iterdir()):
        if not path.is_file():
            continue
        sha = hashlib.sha256()
        md5 = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                sha.update(block)
                md5.update(block)
        destination = remote + "/" + path.name
        bf.copy(str(path), destination, overwrite=False)
        info = bf.stat(destination)
        assert info.size == path.stat().st_size
        if info.md5:
            assert info.md5.lower() == md5.hexdigest()
        files.append({"path": destination, "bytes": info.size, "sha256": sha.hexdigest(), "remote_md5": info.md5})
    evidence = {
        "checkpoint": checkpoint,
        "checkpoint_step": args.step,
        "checkpoint_inventory": inventory,
        "source_training_revision": args.expected_producer_revision,
        "producer_revision_evidence": "verified launch lineage, distinct from export execution checkout",
        "base_model": base,
        "lora_rank": args.expected_lora_rank,
        "lora_alpha": args.expected_lora_alpha,
        "precision_policy": {
            "preserve_lora_precision": True,
            "adapter_before_merge": "source_dtype",
            "base_before_merge": "bfloat16",
            "merge_arithmetic": "float32",
            "output_dtype": "bfloat16",
        },
        "merged_adapter_pairs": adapter_pairs,
        "tensor_count": len(keys),
        "tensor_shapes": shapes,
        "cpu_loader_verified": True,
        "loading_info": loading,
        "all_tensors_finite": True,
        "converter_sha256": converter_sha,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "producer_config_sha256": hashlib.sha256(producer_raw).hexdigest(),
        "effective_producer_config": OmegaConf.to_container(config, resolve=True),
        "export_revision": execution_revision,
        "files": files,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    provenance = hf / "export_provenance.json"
    provenance.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    bf.copy(str(provenance), remote + "/export_provenance.json", overwrite=False)
    sidecar = {
        "source_training_revision": args.expected_producer_revision,
        "training_completion_evidence": monitor,
        "training_monitor_path": monitor_path,
        "producer_config_sha256": evidence["producer_config_sha256"],
        "exporter_sha256": evidence["exporter_sha256"],
        "execution_revision": execution_revision,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    sidecar_path = hf / "final_pipeline_provenance.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    bf.copy(str(sidecar_path), remote + "/final_pipeline_provenance.json", overwrite=False)
    print("EXPORT_SUCCESS=" + remote, flush=True)


if __name__ == "__main__":
    main()

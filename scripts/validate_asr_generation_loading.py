"""Small CPU-only smoke test of the production distributed generation loader.

Default: two local Ray nodes (different Node IDs, same physical host).
For a real cluster: --address auto --node-ids ID1 ID2. No model is loaded and
no generation output is written. Synthetic WAVs travel in metadata so this
also works without a shared filesystem. This tests placement and correctness,
not model quality or production throughput.
"""

import argparse
import glob
import itertools
import io
import json
import time
from pathlib import Path

import numpy as np
import ray
import soundfile as sf
import torch

from recipe.phimm.data.generation_loader import DistributedGenerationLoader


class AudioValidationDataset:
    is_training = False
    config = {"chunk_load_mode": "indexed"}

    def __init__(self, rows=37):
        self.ds = []
        for index in range(rows):
            waveform = np.sin(np.arange(3200 + index * 17) * (0.03 + index * 0.001)).astype(np.float32)
            stream = io.BytesIO()
            sf.write(stream, waveform, 16000, format="WAV", subtype="FLOAT")
            self.ds.append({"id": index, "wav": stream.getvalue(), "reference": f"sample {index}"})

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        # A bad WAV must fail this exact sample, never replace it with a neighbor.
        audio, rate = sf.read(io.BytesIO(row["wav"]), dtype="float32")
        features = torch.fft.rfft(torch.from_numpy(audio), n=256).abs()
        return {"id": row["id"], "reference": row["reference"], "rate": rate, "audio": audio, "features": features}


def identity_collate(samples):
    return samples


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_equal(left, right)
    else:
        assert actual == expected


def real_dataset(input_jsonl, processor_path, rows):
    """Use small metadata only; load tokenizer/processor assets without weights."""
    import blobfile as bf
    from omegaconf import OmegaConf
    from recipe.phimm.data.rl_dataset import RLHFDataset
    from verl.utils import hf_processor, hf_tokenizer

    matches = sorted(glob.glob(processor_path))
    if not matches:
        raise FileNotFoundError(processor_path)
    processor_path = matches[0]
    with bf.BlobFile(input_jsonl, "r") as stream:
        records = [json.loads(line) for line in itertools.islice(stream, rows)]
    if len(records) != rows:
        raise ValueError(f"Expected {rows} rows, found {len(records)}")
    path = Path("real-loading-input.jsonl").resolve()
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    data = OmegaConf.create(
        {"version": 2609, "max_prompt_length": 1024, "max_audio_dur": None, "chunk_load_mode": "indexed", "num_proc": 1}
    )
    conf = [
        {
            "dataset_name": "jsonl",
            "jsonl_paths": str(path),
            "num_proc": 1,
            "post_process": {"verl_format": {"prompt_key": "prompt", "extra_keys": ["source_family"]}},
        }
    ]
    return RLHFDataset(
        conf,
        hf_tokenizer(processor_path, trust_remote_code=False),
        data,
        hf_processor(processor_path, trust_remote_code=False, use_fast=True),
        is_train=False,
    )


def assert_actors_stopped(handles):
    # ray.kill is asynchronous; an already queued method may win the race.
    for handle in handles:
        deadline = time.monotonic() + 10
        while True:
            try:
                ray.get(handle.ready.remote(), timeout=10)
            except ray.exceptions.RayActorError:
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Loader actor survived cleanup")
            time.sleep(0.05)


def validate(node_ids, rows=37, batch_size=8, workers_per_node=1, dataset=None):
    synthetic = dataset is None
    dataset = AudioValidationDataset(rows) if synthetic else dataset
    baseline = [dataset[i] for i in range(len(dataset))]
    results = []
    for start in [0, min(batch_size * 2, rows)]:
        started = time.monotonic()
        loader = DistributedGenerationLoader(
            dataset,
            batch_size,
            node_ids,
            identity_collate,
            start_index=start,
            workers_per_node=workers_per_node,
            prefetch_batches=2,
            timeout_s=120,
        )
        handles = list(loader.workers)
        batches, seen_nodes = [], set()
        for batch in loader:
            batches.append(batch)
            seen_nodes.update(stat["node_id"] for stat in loader.stats)
        actual = [sample for batch in batches for sample in batch]
        assert_equal(actual, baseline[start:])
        assert loader.closed and not loader.workers
        if len(actual) >= len(node_ids):
            assert seen_nodes == set(node_ids)
        assert_actors_stopped(handles)
        results.append(
            {
                "start_index": start,
                "rows": len(actual),
                "batch_sizes": [len(b) for b in batches],
                "nodes_seen": sorted(seen_nodes),
                "locations": loader.locations,
                "wall_s_including_startup": time.monotonic() - started,
                "exact_waveform_and_feature_match": True,
                "actors_cleaned": True,
            }
        )
    if synthetic:
        # Exercise failure propagation with a real decoder error.
        dataset.ds[0]["wav"] = b"invalid wav"
        loader = DistributedGenerationLoader(
            dataset, batch_size, node_ids, identity_collate, timeout_s=120, workers_per_node=workers_per_node
        )
        try:
            list(loader)
        except ray.exceptions.RayTaskError:
            assert loader.closed and not loader.workers
        else:
            raise AssertionError("Unreadable audio did not fail")
    return {
        "node_ids": node_ids,
        "cases": results,
        "unreadable_audio_fails": True if synthetic else "checked separately with synthetic WAVs",
        "scope": (
            "CPU WAV decoding and feature processing"
            if synthetic
            else "Real RLHFDataset, audio, tokenizer and processor"
        )
        + "; no ASR model or GPU inference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="Existing Ray cluster; omitted starts two local logical nodes")
    parser.add_argument("--node-ids", nargs="+")
    parser.add_argument("--input-jsonl", help="Optional existing generation JSONL to sample (local or az://)")
    parser.add_argument("--processor-path", help="Matching LOCAL cached tokenizer/processor directory; glob accepted")
    parser.add_argument("--rows", type=int, default=37)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers-per-node", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("tmp/multinode-loading/validation.json"))
    args = parser.parse_args()
    if args.address and not args.node_ids:
        parser.error("--address requires explicit --node-ids for the intended test nodes")
    if bool(args.input_jsonl) != bool(args.processor_path):
        parser.error("--input-jsonl and --processor-path must be used together")
    if args.rows < max(3, 2 * args.batch_size + 1):
        parser.error("rows must exceed two batches to test resume and a final partial batch")
    cluster = None
    try:
        if args.address:
            ray.init(address=args.address, logging_level="ERROR")
        else:
            from ray.cluster_utils import Cluster

            cluster = Cluster()
            for i in range(2):
                cluster.add_node(
                    num_cpus=max(2, args.workers_per_node),
                    include_dashboard=False if i == 0 else None,
                    object_store_memory=80 * 1024 * 1024,
                )
            ray.init(address=cluster.address, logging_level="ERROR")
        node_ids = args.node_ids or sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
        if len(node_ids) < 2:
            raise ValueError("At least two nodes are required for this validation")
        torch.set_num_threads(1)
        dataset = real_dataset(args.input_jsonl, args.processor_path, args.rows) if args.input_jsonl else None
        result = validate(node_ids, args.rows, args.batch_size, args.workers_per_node, dataset=dataset)
        if args.input_jsonl:
            result["input_jsonl"] = args.input_jsonl
            result["processor_path"] = args.processor_path
        result["topology"] = "existing cluster" if args.address else "two logical nodes on one physical host"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        ray.shutdown()
        if cluster is not None:
            cluster.shutdown()


if __name__ == "__main__":
    main()

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare lightweight, speaker-free ordered 2/3-segment ASR manifests.

Matches the Fisher contiguous23 window schedule. Downloads manifests only;
keeps audio selectors unchanged. Groups across input shard boundaries before
sorting, separating distinct source audio selectors under the same recording ID.
ICSI is an audit source and is explicitly opt-in via --datasets.
"""

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blobfile as bf

ROOT = "az://orngwus2cresco/data/boren/data/Training"


def windows(size):
    """Return distinct alternating 2/3-segment windows covering an eligible group."""
    if size < 3:
        return []
    result = [(i, min(2 + i % 2, size - i)) for i in range(size - 1)]
    tail = size - 3
    result.append((tail, 5 - result[tail][1]))
    return sorted(result)


def prepare(dataset, args):
    """Cache source manifests, write ordered ASR windows, and optionally upload them."""
    source = f"{args.source_root}/{dataset}"
    remote = f"{args.output_root}/{dataset}_contiguous23_{args.version}"
    work = Path(args.work_dir) / dataset
    cache, out = work / "source_shards", work / "output"
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(n for n in bf.listdir(source) if n.endswith(".jsonl.gz"))
    if not names:
        raise ValueError(f"No source shards: {source}")

    def download(name):
        local = cache / name
        if not local.exists() or local.stat().st_size != bf.stat(f"{source}/{name}").size:
            partial = local.with_suffix(".partial")
            bf.copy(f"{source}/{name}", str(partial), overwrite=True)
            partial.replace(local)
        print(f"{dataset}: cached {name}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, names))
    groups = defaultdict(list)
    input_rows, input_duration = 0, 0.0
    source_stats = []
    for name in names:
        local = cache / name
        source_stats.append({"shard": name, "compressed_sha256": hashlib.sha256(local.read_bytes()).hexdigest()})
        with gzip.open(local, "rt") as stream:
            for line in stream:
                raw = json.loads(line)
                row = {
                    k: raw[k]
                    for k in (
                        "id",
                        "recording_id",
                        "audio_path",
                        "text",
                        "duration_s",
                        "source_start_s",
                        "source_end_s",
                    )
                }
                if not isinstance(row["text"], str) or not row["text"].strip():
                    raise ValueError(f"Empty text: {row['id']}")
                if not isinstance(row["audio_path"], str) or not row["audio_path"]:
                    raise ValueError(f"Invalid audio: {row['id']}")
                for key in ("duration_s", "source_start_s", "source_end_s"):
                    row[key] = float(row[key])
                    if not math.isfinite(row[key]):
                        raise ValueError(f"Invalid {key}: {row['id']}")
                if row["duration_s"] <= 0 or row["source_end_s"] <= row["source_start_s"]:
                    raise ValueError(f"Invalid duration: {row['id']}")
                parent_selector = row["audio_path"].rsplit("#", 1)[0]
                groups[(row["recording_id"], parent_selector)].append(row)
                input_rows += 1
                input_duration += row["duration_s"]
    print(f"{dataset}: grouped {input_rows} clips / {len(groups)} recordings", flush=True)
    stats, counts, skipped = [], Counter(), Counter()
    output_rows, output_duration, eligible = 0, 0.0, 0
    stream, raw_out, digest, part_rows, part_path = None, None, None, 0, None

    def close_part():
        stream.close()
        raw_out.close()
        stats.append(
            {
                "shard": part_path.name,
                "output_rows": part_rows,
                "uncompressed_sha256": digest.hexdigest(),
                "compressed_sha256": hashlib.sha256(part_path.read_bytes()).hexdigest(),
                "compressed_bytes": part_path.stat().st_size,
            }
        )

    for group_key in sorted(groups):
        rec, parent_selector = group_key
        items = groups[group_key]
        items.sort(key=lambda x: (x["source_start_s"], x["id"]))
        if len({x["source_start_s"] for x in items}) != len(items):
            raise ValueError(f"Duplicate timestamp: {rec}")
        if len({x["id"] for x in items}) != len(items):
            raise ValueError(f"Duplicate clip ID: {rec}")
        if len(items) < 3:
            skipped[len(items)] += 1
            continue
        eligible += 1
        for start, count in windows(len(items)):
            if stream is None or part_rows == args.shard_rows:
                if stream is not None:
                    close_part()
                part_path = out / f"part-{len(stats):04d}.jsonl.gz"
                raw_out = part_path.open("wb")
                stream = gzip.GzipFile(filename="", fileobj=raw_out, mode="wb", mtime=0)
                digest, part_rows = hashlib.sha256(), 0
            members = items[start : start + count]
            texts = [x["text"] for x in members]
            ids = [x["id"] for x in members]
            duration = round(math.fsum(x["duration_s"] for x in members), 6)
            identity = hashlib.sha256(json.dumps([rec, ids]).encode()).hexdigest()[:16]
            row = {
                "id": f"contiguous23:{identity}",
                "recording_id": rec,
                "merge_parent_audio_path": rec,
                "audio_path": [x["audio_path"] for x in members],
                "text": " ".join(texts),
                "segment_reference_texts": texts,
                "audio_length": duration,
                "duration_s": duration,
                "merge_count": count,
                "n_segments": count,
                "source_ids": ids,
                "source_audio_lengths": [x["duration_s"] for x in members],
                "source_start_seconds": [x["source_start_s"] for x in members],
                "source_end_seconds": [x["source_end_s"] for x in members],
                "merge_method": "contiguous23",
                "source_order_by": "timestamp",
                "language": "English",
                "data_source": dataset,
            }
            content = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
            stream.write(content)
            digest.update(content)
            part_rows += 1
            output_rows += 1
            output_duration += duration
            counts[count] += 1
        groups[group_key] = None
    if stream is not None:
        close_part()
    if not output_rows:
        raise ValueError("No eligible recordings")
    summary = {
        "dataset": f"{dataset}_contiguous23",
        "method": "contiguous23",
        "source_manifest_dir": source,
        "grouping": "recording_id + source audio selector before #",
        "input_recording_ids": len({key[0] for key in groups}),
        "input_audio_groups": len(groups),
        "remote_output_dir": remote,
        "source_split": "icsi_audit" if dataset == "moe_icsi" else "train",
        "speaker_info_dropped": True,
        "audio_created": False,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "eligible_recordings": eligible,
        "skipped_short_recordings_by_len": dict(skipped),
        "merge_count_distribution": dict(counts),
        "input_audio_hours": input_duration / 3600,
        "output_audio_hours": output_duration / 3600,
        "mean_merged_length_s": output_duration / output_rows,
        "source_shards": source_stats,
        "shards": stats,
    }
    (out / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    import yaml

    extra = [
        "recording_id",
        "merge_parent_audio_path",
        "segment_reference_texts",
        "merge_count",
        "n_segments",
        "source_ids",
        "source_audio_lengths",
        "source_start_seconds",
        "source_end_seconds",
    ]
    block = [
        {
            "dataset_name": "jsonl",
            "jsonl_paths": [f"{remote}/{s['shard']}" for s in stats],
            "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
            "post_process": {
                "add_field": {"fields": {"data_source": dataset}},
                "verl_format": {"prompt_key": "prompt", "extra_keys": extra},
            },
        }
    ]
    config = f"# Provenance: {remote}/preparation.json\n" + yaml.safe_dump(block, sort_keys=False)
    (out / "train_data.yaml").write_text(config)
    if args.upload:
        for name in [s["shard"] for s in stats] + ["preparation.json", "train_data.yaml"]:
            bf.copy(str(out / name), f"{remote}/{name}", overwrite=False)
            assert bf.stat(f"{remote}/{name}").size == (out / name).stat().st_size
        print(f"{dataset}: uploaded {output_rows} windows -> {remote}", flush=True)
    return summary


def main():
    """Parse preparation options and process each requested dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--source-root", default=f"{ROOT}/diarization/moe_en/full_v1/manifests")
    parser.add_argument("--output-root", default=ROOT)
    parser.add_argument("--version", default="20261126_v1")
    parser.add_argument("--work-dir", default="tmp/moe_dialogue_contiguous23_work")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-rows", type=int, default=50000)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.shard_rows <= 0 or args.workers <= 0:
        parser.error("shard-rows and workers must be positive")
    summaries = [prepare(dataset, args) for dataset in args.datasets]
    Path(args.work_dir, "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()

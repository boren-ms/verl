"""Prepare reproducible, source-separated ASR manifests without rewriting audio.

Run with ``python -m recipe.phimm.data.offline_random_merge --help``.
The source spec is a JSON list of {name, config} objects, where config is an
ordinary Base3a JSONL dataset block. Only raw-input transforms are applied.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from recipe.phimm.data.dataset import jsonl_dataset, process_ds
from recipe.phimm.data.random_merge import _iter_random_merge
from recipe.phimm.utils.audio import get_audio_duration

DURATION_FIELDS = ("audio_length", "audio_duration_seconds", "duration")
AUDIO_FIELDS = ("audio_path", "audio_file", "audio_chunk")
TEXT_FIELDS = ("text", "display", "display_ne_tagged", "lexical", "ref", "hyp")
GROUP_FIELDS = ("data_source", "language", "prompt")


def audio_source(row):
    return next((row[k] for k in AUDIO_FIELDS if row.get(k)), None)


def encoded_metadata_fields(rows):
    semantic = {
        *AUDIO_FIELDS,
        *TEXT_FIELDS,
        *DURATION_FIELDS,
        *GROUP_FIELDS,
        "id",
        "keywords",
        "components",
        "gt_output",
    }
    keys = set().union(*(row.keys() for row in rows)) - semantic
    return {key for key in keys if any(row.get(key) is None or isinstance(row.get(key), (dict, list)) for row in rows)}


def merge_metadata(rows, indices, originals, text_separator=" ", encoded_fields=()):
    """Merge semantic fields; retain every original value in ordered provenance.

    Unknown fields are always lists aligned to source_indices, including for
    singleton outputs. This avoids silently keeping just the first value, and
    avoids switching a field between a scalar and a list across JSONL rows.
    Original records are JSON strings to retain arbitrary nested producer
    schemas losslessly without Arrow's heterogeneous-struct inference.
    Nullable/structured ancillary fields are lists of JSON-encoded values;
    statistics.json records their names. Other ancillary values stay typed.
    """
    result = {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    paths = [audio_source(row) for row in rows]
    for key in keys:
        values = [row.get(key) for row in rows]
        if key in AUDIO_FIELDS:
            result[key] = ";".join(paths)
        elif key in TEXT_FIELDS:
            result[key] = text_separator.join(value or "" for value in values)
        elif key in DURATION_FIELDS:
            result[key] = math.fsum(row["audio_length"] for row in rows)
        elif key == "id":
            result[key] = ";".join(
                str(value if value is not None else i) for value, i in zip(values, indices, strict=True)
            )
        elif key == "keywords":
            result[key] = list(dict.fromkeys(word for value in values for word in (value or [])))
        elif key == "components":
            result[key] = [part for value in values for part in (value or [])]
        elif key in GROUP_FIELDS:
            if any(value != values[0] for value in values):
                raise ValueError(f"Incompatible {key} in merged group")
            result[key] = values[0]
        elif key != "gt_output":
            result[key] = (
                [json.dumps(value, ensure_ascii=False) for value in values] if key in encoded_fields else values
            )
    result.update(
        audio_path=";".join(paths),
        audio_length=math.fsum(row["audio_length"] for row in rows),
        random_merge_count=len(rows),
        random_merge_source_indices=list(indices),
        source_audio_lengths=[row["audio_length"] for row in rows],
        source_metadata_json=[json.dumps(originals[i], ensure_ascii=False, sort_keys=True) for i in indices],
    )
    return result


def prepare_rows(rows, originals, *, cap=180, buffer_size=100, seed=42, text_separator=" "):
    encoded_fields = encoded_metadata_fields(rows)
    for packed in _iter_random_merge(
        rows,
        max_audio_length=cap,
        buffer_size=buffer_size,
        seed=seed,
        text_separator=text_separator,
        group_by=GROUP_FIELDS,
    ):
        indices = packed["random_merge_source_indices"]
        yield merge_metadata([rows[i] for i in indices], indices, originals, text_separator, encoded_fields)


def distribution(values, bins):
    hist, edges = np.histogram(values, bins=bins)
    return {f"[{a:g},{b:g})": int(n) for a, b, n in zip(edges[:-1], edges[1:], hist, strict=True)}


def describe(rows):
    durations = [row["audio_length"] for row in rows]
    # Whitespace tokens, explicitly also report characters for unsegmented Chinese.
    words = [len(row["text"].split()) for row in rows]
    return {
        "count": len(rows),
        "total_audio_seconds": math.fsum(durations),
        "mean_audio_seconds": float(np.mean(durations)),
        "min_audio_seconds": min(durations),
        "max_audio_seconds": max(durations),
        "audio_percentiles": dict(
            zip(
                ("p10", "p50", "p90", "p95", "p99"),
                np.percentile(durations, [10, 50, 90, 95, 99]).tolist(),
                strict=True,
            )
        ),
        "mean_word_count": float(np.mean(words)),
        "mean_character_count": float(np.mean([len(row["text"]) for row in rows])),
        "word_count_definition": "whitespace-separated tokens (not Chinese word segmentation)",
        "audio_length_distribution": distribution(
            durations, [0, 30, 60, 90, 120, 150, 170, 175, 179, 180, float("inf")]
        ),
        "word_count_distribution": distribution(words, [0, 10, 50, 100, 200, 300, 400, 500, 750, 1000, float("inf")]),
    }


def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.sources).read_text())
    summaries = {}
    for spec in specs:
        name, conf = spec["name"], spec["config"]
        target = root / name
        target.mkdir(exist_ok=True)
        print(f"[{name}] loading", flush=True)
        local_input = Path(args.input_dir) / f"{name}.jsonl" if args.input_dir else None
        ds = jsonl_dataset(str(local_input) if local_input else conf["jsonl_paths"])
        limit = conf.get("post_process", {}).get("input_egs_limit")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        originals = ds.to_list()
        pre = dict(conf.get("pre_process", {}))
        pre.pop("random_merge", None)
        if set(pre) - {"path_map", "rename_fields", "clean_tagged_text"}:
            raise ValueError(f"Unsupported offline input transforms: {list(pre)}")
        rows = process_ds(ds, **pre).to_list()
        duration_cache = target / "durations.json"
        cached = json.loads(duration_cache.read_text()) if duration_cache.exists() else {}

        def duration(row, cached=cached):
            source = audio_source(row)
            value = next((row[k] for k in DURATION_FIELDS if row.get(k) is not None), None)
            if value is None:
                value = cached.get(source)
            if value is None:
                value = get_audio_duration(source)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid duration for {source}: {value!r}")
            return source, float(value)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                for i, (source, seconds) in enumerate(executor.map(duration, rows)):
                    rows[i]["audio_length"] = seconds
                    cached[source] = seconds
                    if (i + 1) % 500 == 0:
                        duration_cache.write_text(json.dumps(cached))
                        print(f"[{name}] durations {i + 1}/{len(rows)}", flush=True)
        finally:
            duration_cache.write_text(json.dumps(cached))
        output = list(prepare_rows(rows, originals, cap=args.cap, buffer_size=args.buffer_size, seed=args.seed))
        indices = [i for row in output for i in row["random_merge_source_indices"]]
        if sorted(indices) != list(range(len(rows))):
            raise AssertionError("Source rows lost or duplicated")
        if not math.isclose(
            math.fsum(r["audio_length"] for r in output), math.fsum(r["audio_length"] for r in rows), abs_tol=1e-8
        ):
            raise AssertionError("Duration not conserved")
        manifest = target / "data.jsonl"
        with manifest.open("w") as handle:
            for row in output:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        # Exercise the actual Arrow JSONL loader across the entire emitted manifest.
        loaded = jsonl_dataset(str(manifest))
        if loaded.to_list() != Dataset.from_list(output).to_list():
            raise AssertionError("Output JSONL roundtrip changed values")
        stats = {
            "json_encoded_metadata_fields": sorted(encoded_metadata_fields(rows)),
            "input": describe(rows),
            "output": describe(output),
            "cap_seconds_exclusive": args.cap,
            "buffer_size": args.buffer_size,
            "seed": args.seed,
            "source_manifest": conf["jsonl_paths"],
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "audit": {
                "all_sources_used_exactly_once": True,
                "duration_conserved": True,
                "jsonl_loader_roundtrip": True,
            },
        }
        (target / "statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        post = dict(conf.get("post_process", {}))
        post.pop("input_egs_limit", None)
        # Original per-utterance diagnostics now live in ordered provenance;
        # only fields used by the task are passed through as trainer metadata.
        if "verl_format" in post:
            post["verl_format"] = {**post["verl_format"], "extra_keys": ["data_source"]}
        uri = f"{args.blob_root.rstrip('/')}/{name}/data.jsonl" if args.blob_root else str(manifest.resolve())
        block = dict(
            dataset_name="jsonl",
            jsonl_paths=uri,
            add_task_info=conf["add_task_info"],
            post_process=post,
        )
        comments = "\n".join(f"# {line}" for line in json.dumps(stats, indent=2, ensure_ascii=False).splitlines())
        (target / "train_data.yaml").write_text(
            comments + "\n" + yaml.safe_dump([block], sort_keys=False, allow_unicode=True)
        )
        summaries[name] = stats
        print(
            f"[{name}] COMPLETE {len(rows)} -> {len(output)}; mean={stats['output']['mean_audio_seconds']:.3f}s",
            flush=True,
        )
    (root / "statistics.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    (root / "train_data.yaml").write_text(
        "# Offline merge: each dataset packed independently; do not merge again.\n"
        + "\n".join((root / spec["name"] / "train_data.yaml").read_text() for spec in specs)
    )
    (root / "sources.json").write_text(json.dumps(specs, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blob-root")
    parser.add_argument("--cap", type=float, default=180)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not math.isfinite(args.cap) or args.cap <= 0 or args.buffer_size <= 0 or args.workers <= 0:
        parser.error("cap, buffer-size and workers must be positive")
    run(args)


if __name__ == "__main__":
    main()

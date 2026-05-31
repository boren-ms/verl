"""Pre-segment long-audio eval JSONL offline using the SAME Smart VAD logic
used on-the-fly by :func:`recipe.phimm.data.dataset.svad_explode`.

The gen-style long-audio eval (``recipe.phimm.config.eval.long_eval_test``)
chunks each long meeting wav into <= ``max_len_sec`` segments at data-load time
via the ``svad_explode`` pre_process step. That repeats the (expensive) SVAD
chunking on every run. This script runs the identical chunking once, offline,
and writes a NEW pre-segmented JSONL where every row is already a single
segment carrying the audio time-range chunk spec ``WavPath#start:end`` plus the
grouping fields ``seg_index`` / ``n_segments`` / ``seg_start`` / ``seg_end`` /
``parent_audio_path``.

Because it calls the exact same ``svad_explode`` function, the produced rows are
identical to the on-the-fly output — the only difference is WHEN the chunking
happens. The matching val_data config
(``recipe/phimm/config/data/val_data/long_eval_test_seg.yaml``) drops the
``svad_explode`` pre_process and consumes this file directly.

Usage:
    python -m recipe.phimm.data.long_audio_vad_explode \
        --src az://orngwus2cresco/data/boren/data/Evaluation/long_audio_test/sample_210d3d05.jsonl \
        --dst az://orngwus2cresco/data/boren/data/Evaluation/long_audio_test_seg/sample_210d3d05.jsonl \
        --max-len-sec 40 --audio-key WavPath
"""

from __future__ import annotations

import argparse
import json

import blobfile as bf

from recipe.phimm.data.dataset import svad_explode


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with bf.BlobFile(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    bf.makedirs(path.rsplit("/", 1)[0])
    with bf.BlobFile(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True,
                   help="Source long-audio JSONL (one row per long wav).")
    p.add_argument("--dst", required=True,
                   help="Destination pre-segmented JSONL (one row per segment).")
    p.add_argument("--max-len-sec", type=float, default=40.0,
                   help="Max segment length for SVAD (default 40s; matches the eval config).")
    p.add_argument("--audio-key", default="WavPath",
                   help="Field holding the wav URI to chunk (default WavPath, matches eval config).")
    p.add_argument("--min-seg-sec", type=float, default=0.1,
                   help="Drop segments shorter than this (default 0.1s).")
    p.add_argument("--target-sr", type=int, default=16000,
                   help="Resample audio to this rate before chunking (default 16000).")
    p.add_argument("--path-replace", default=None,
                   help='Optional JSON dict of path-prefix rewrites, e.g. '
                        '\'{"/datablob1/": "az://orngwus2cresco/data/speech/"}\'.')
    return p.parse_args()


def main():
    args = parse_args()

    rows = _read_jsonl(args.src)
    print(f"[long_audio_vad_explode] loaded {len(rows)} records from {args.src}", flush=True)

    kwargs = dict(
        max_len_sec=args.max_len_sec,
        audio_key=args.audio_key,
        min_seg_sec=args.min_seg_sec,
        target_sr=args.target_sr,
    )
    if args.path_replace:
        kwargs["path_replace"] = json.loads(args.path_replace)

    # Reuse the exact on-the-fly exploder so the offline output is identical.
    exploded = svad_explode(rows, **kwargs)
    out_rows = [dict(r) for r in exploded]

    # Stable ordering: group by parent then segment index (matches aggregation).
    out_rows.sort(key=lambda r: (r.get("parent_audio_path", ""), r.get("seg_index", 0)))

    _write_jsonl(args.dst, out_rows)
    print(f"[long_audio_vad_explode] {len(rows)} records -> {len(out_rows)} segments  ({args.dst})",
          flush=True)


if __name__ == "__main__":
    main()

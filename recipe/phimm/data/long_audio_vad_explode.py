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

Usage (single file):
    python -m recipe.phimm.data.long_audio_vad_explode \
        --src az://orngwus2cresco/data/boren/data/Evaluation/long_audio_test/sample_210d3d05.jsonl \
        --dst az://orngwus2cresco/data/boren/data/Evaluation/long_audio_test_seg/sample_210d3d05.jsonl \
        --max-len-sec 40 --audio-key WavPath

Usage (batch corpora, one {root}/{corpus}/test.jsonl per corpus):
    python -m recipe.phimm.data.long_audio_vad_explode \
        --src-root az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US \
        --dst-root az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment/en-US \
        --corpora Conversation_DTEST_FY21Q1_en-US ... \
        --max-len-sec 40 --audio-key WavPath \
        --path-replace '{"/datablob1/": "az://orngwus2cresco/data/speech/"}'
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from functools import partial

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


def _explode_one(row: dict, kwargs: dict) -> list[dict]:
    """Run the shared svad_explode on a single record (a 1-row dataset).

    Calling svad_explode on a 1-row list yields the exact same per-record
    output as the on-the-fly exploder, which is what keeps the offline file
    identical. Wrapping per-record lets us fan the (audio-bound) work out
    across a process pool.
    """
    exploded = svad_explode([row], **kwargs)
    return [dict(r) for r in exploded]


def _explode_rows(rows: list[dict], kwargs: dict, n_workers: int) -> list[dict]:
    if n_workers <= 1:
        # Single pass through the exact on-the-fly exploder.
        return [dict(r) for r in svad_explode(rows, **kwargs)]
    out_rows: list[dict] = []
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for seg_rows in pool.imap_unordered(partial(_explode_one, kwargs=kwargs), rows):
            out_rows.extend(seg_rows)
    return out_rows


def explode_file(src: str, dst: str, kwargs: dict, n_workers: int) -> int:
    rows = _read_jsonl(src)
    print(f"[long_audio_vad_explode] loaded {len(rows)} records from {src}", flush=True)

    out_rows = _explode_rows(rows, kwargs, n_workers)

    # Stable ordering: group by parent then segment index (matches aggregation).
    out_rows.sort(key=lambda r: (r.get("parent_audio_path", ""), r.get("seg_index", 0)))

    _write_jsonl(dst, out_rows)
    print(f"[long_audio_vad_explode] {len(rows)} records -> {len(out_rows)} segments  ({dst})",
          flush=True)
    return len(out_rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", help="Source long-audio JSONL (one row per long wav).")
    p.add_argument("--dst", help="Destination pre-segmented JSONL (one row per segment).")
    p.add_argument("--src-root", help="Source root for batch corpora mode (with --corpora).")
    p.add_argument("--dst-root", help="Destination root for batch corpora mode (with --corpora).")
    p.add_argument("--corpora", nargs="*", default=None,
                   help="Corpus subdir names under --src-root/--dst-root. Each maps to "
                        "{root}/{corpus}/test.jsonl.")
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
    p.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="Process-pool workers for per-record explode (default: half cpu count).")
    return p.parse_args()


def main():
    args = parse_args()

    kwargs = dict(
        max_len_sec=args.max_len_sec,
        audio_key=args.audio_key,
        min_seg_sec=args.min_seg_sec,
        target_sr=args.target_sr,
    )
    if args.path_replace:
        kwargs["path_replace"] = json.loads(args.path_replace)

    if args.corpora:
        if not (args.src_root and args.dst_root):
            raise SystemExit("--corpora requires --src-root and --dst-root")
        total = 0
        for c in args.corpora:
            src = f"{args.src_root.rstrip('/')}/{c}/test.jsonl"
            dst = f"{args.dst_root.rstrip('/')}/{c}/test.jsonl"
            total += explode_file(src, dst, kwargs, args.n_workers)
        print(f"[long_audio_vad_explode] total segments: {total}", flush=True)
    else:
        if not (args.src and args.dst):
            raise SystemExit("provide either --src/--dst or --corpora with --src-root/--dst-root")
        explode_file(args.src, args.dst, kwargs, args.n_workers)


if __name__ == "__main__":
    main()

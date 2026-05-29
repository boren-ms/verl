"""Explode long-form InhouseASR_2605 audios into per-segment items using Microsoft Smart VAD.

For corpora whose ``DisplayTranscription`` has no inline ``[start end]`` markers
(e.g. ``OnlineMeetings_CS_Product_FY22`` / ``OnlineMeetings_CS_Shiproom_FY22``),
we cannot use the marker-based exploder. Instead, run SVAD (ported from MoE
``phyagi/eval/tasks/audio/chunk/svad.py``) to chunk each long wav into
``max_len_sec`` segments and emit one row per segment with the audio time-range
chunk spec ``WavPath#start_sec:end_sec``.

Each segment row carries the FULL parent ``DisplayTranscription`` as ``label``
(mirroring MoE's ``JsonlDataset`` convention). Downstream scoring must collate
per-segment hyps by ``parent_uuid`` (sorted by ``seg_start``) and compute TER on
the concatenated hyp vs the single full reference. Per-segment DTER from this
file is meaningless on its own.

Usage:
    python -m recipe.phimm.data.inhouse_2605_vad_explode --max-len-sec 40
"""

from __future__ import annotations
import argparse
import io
import json
import multiprocessing as mp
import os
from functools import partial

import blobfile as bf
import numpy as np
import soundfile as sf

from recipe.phimm.utils.audio import resample_audio
from recipe.phimm.utils.svad.svad import SVadChunker


SRC_ROOT = "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US"
DST_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg/en-US"
WAV_AZ_PREFIX = "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/"
WAV_LOCAL_PREFIX = "/datablob1/users/ruchaofan/Evaluation/InhouseASR_2605/"

DEFAULT_CORPORA = [
    "OnlineMeetings_CS_Product_FY22_en-US_DTEST",
    "OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST",
]


def _load_wav_16k(az_path: str):
    with bf.BlobFile(az_path, "rb") as f:
        data, sr = sf.read(f)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    if sr != 16000:
        data, sr = resample_audio(data, sr, 16000)
    return data, sr


def _process_record(d: dict, max_len_sec: float, corpus_dir: str):
    """Run SVAD on a single record and return a list of segment rows."""
    az_wav = d["WavPath"].replace(WAV_LOCAL_PREFIX, WAV_AZ_PREFIX)
    uuid = d.get("UUID", "")
    try:
        audio, sr = _load_wav_16k(az_wav)
    except Exception as exc:
        print(f"[{corpus_dir}] FAILED to load {az_wav}: {exc}", flush=True)
        return []

    chunker = SVadChunker(max_len_sec=max_len_sec, verbose=False)
    spans = chunker.chunk(audio, sr)
    full_dt = d.get("DisplayTranscription", "") or ""
    full_tx = d.get("Transcription", "") or ""
    corpus = d.get("CorpusName", corpus_dir)
    locale = d.get("locale", "")

    rows = []
    for idx, (s, e) in enumerate(spans):
        if e - s < 0.1:
            continue
        rows.append({
            "CorpusName": corpus,
            "UUID": f"{uuid}_s{idx:03d}",
            "parent_uuid": uuid,
            "seg_index": idx,
            "n_segments": len(spans),
            "WavPath": f"{az_wav}#{round(s, 3)}:{round(e, 3)}",
            "DisplayTranscription": full_dt,
            "Transcription": full_tx,
            "locale": locale,
            "seg_start": round(s, 3),
            "seg_end": round(e, 3),
            "audio_dur": len(audio) / sr,
        })
    print(f"[{corpus_dir}] {uuid}: {len(audio)/sr:.1f}s -> {len(rows)} segments", flush=True)
    return rows


def _worker(d, max_len_sec, corpus_dir):
    return _process_record(d, max_len_sec, corpus_dir)


def explode_corpus(corpus_dir: str, max_len_sec: float, n_workers: int) -> int:
    src = f"{SRC_ROOT}/{corpus_dir}/test.jsonl"
    dst = f"{DST_ROOT}/{corpus_dir}/test.jsonl"

    records = []
    with bf.BlobFile(src, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[{corpus_dir}] loaded {len(records)} records from {src}", flush=True)

    out_rows = []
    if n_workers <= 1:
        for d in records:
            out_rows.extend(_process_record(d, max_len_sec, corpus_dir))
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for rows in pool.imap_unordered(
                partial(_worker, max_len_sec=max_len_sec, corpus_dir=corpus_dir),
                records,
            ):
                out_rows.extend(rows)

    out_rows.sort(key=lambda r: (r["parent_uuid"], r["seg_index"]))
    with bf.BlobFile(dst, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{corpus_dir}] {len(records)} records -> {len(out_rows)} segments  ({dst})", flush=True)
    return len(out_rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpora", nargs="*", default=DEFAULT_CORPORA,
                   help="Source corpus dir names under SRC_ROOT (default: Product + Shiproom).")
    p.add_argument("--max-len-sec", type=float, default=40.0,
                   help="Max segment length for SVAD (default 40s; leaves margin below the 50s eval cap).")
    p.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="Multiprocessing workers (default: half cpu count).")
    return p.parse_args()


def main():
    args = parse_args()
    total = 0
    for c in args.corpora:
        total += explode_corpus(c, args.max_len_sec, args.n_workers)
    print(f"Total segments: {total}", flush=True)


if __name__ == "__main__":
    main()

"""Explode InhouseASR_2605 records into per-segment items using DisplayTranscription's [s e] markers.

Each input record has one wav file and a multi-segment DisplayTranscription.
Output: one record per segment with audio_path = "<az_wav_path>#start:end" (time-range chunk spec)
and text = the segment's display text, plus CorpusName preserved as data_source.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import blobfile as bf

SRC_ROOT = "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US"
DST_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg/en-US"
WAV_AZ_PREFIX = "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/"
WAV_LOCAL_PREFIX = "/datablob1/users/ruchaofan/Evaluation/InhouseASR_2605/"

TIME_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\]")


def explode_corpus(corpus_dir: str) -> int:
    src = f"{SRC_ROOT}/{corpus_dir}/test.jsonl"
    dst = f"{DST_ROOT}/{corpus_dir}/test.jsonl"
    out_lines = []
    n_in = n_out = 0
    with bf.BlobFile(src, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            dt = d.get("DisplayTranscription", "")
            wav = d["WavPath"].replace(WAV_LOCAL_PREFIX, WAV_AZ_PREFIX)
            corpus = d.get("CorpusName", corpus_dir)
            uuid = d.get("UUID", "")
            locale = d.get("locale", "")

            parts = TIME_RE.split(dt)
            # parts = [preamble, s, e, text, s, e, text, ...]
            for seg_idx, i in enumerate(range(1, len(parts), 3)):
                s = float(parts[i])
                e = float(parts[i + 1])
                text = parts[i + 2].strip() if i + 2 < len(parts) else ""
                if not text:
                    continue
                if e - s < 0.1:
                    continue
                out_lines.append({
                    "CorpusName": corpus,
                    "UUID": f"{uuid}_s{seg_idx:03d}",
                    "WavPath": f"{wav}#{s}:{e}",
                    "DisplayTranscription": text,
                    "locale": locale,
                    "seg_start": s,
                    "seg_end": e,
                })
                n_out += 1

    with bf.BlobFile(dst, "w") as f:
        for r in out_lines:
            f.write(json.dumps(r) + "\n")
    print(f"[{corpus_dir}] {n_in} records -> {n_out} segments  ({dst})", flush=True)
    return n_out


def main():
    corpora = []
    for entry in bf.listdir(SRC_ROOT):
        if entry.endswith("/"):
            entry = entry[:-1]
        corpora.append(entry)
    print("Corpora:", corpora, flush=True)

    total = 0
    for c in corpora:
        total += explode_corpus(c)
    print(f"Total segments: {total}", flush=True)


if __name__ == "__main__":
    main()

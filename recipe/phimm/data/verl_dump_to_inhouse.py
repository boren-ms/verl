"""Convert verl ``val_data_gen`` DTER dumps into the ``inhouse-asr-compare`` JSON.

The ASR eval reward (``recipe/phimm/reward/asr_inhouse_measure.py::eval_score``)
emits a ``dter_detail`` field on every dumped row — a single
``UtteranceTERMetrics``-style entry carrying the DisfluencyTolerant TER word
alignment (``word_align``), per-word TER classes (``word_ter_class``), the
category breakdown (``ter_category_info``), and ``display_form_tx`` /
``display_form_hyp``. This script reads those rows and writes the JSON-list
format consumed by the ``inhouse-asr-compare`` skill::

    [
      {"UtteranceId": "...", "UtteranceTERMetrics": [ {dter_detail...} ]},
      ...
    ]

For long-audio grouped eval (``long_audio_grouped`` reward manager) every
segment row of a parent wav carries the SAME per-parent ``dter_detail``, so rows
are de-duplicated by ``UtteranceId`` (default: ``parent_audio_path``, falling
back to ``audio_path`` stripped of its ``#start:end`` suffix, then ``id``).

Usage:
    python -m recipe.phimm.data.verl_dump_to_inhouse \\
        --val-data-dir ~/checkpoints/.../val_data_gen \\
        --output tmp/long_audio_inhouse.json \\
        [--step 0] [--uid-key parent_audio_path]
"""

from __future__ import annotations

import argparse
import json
import os

import blobfile as bf


def _strip_seg(path: str) -> str:
    return str(path).split("#", 1)[0]


def _utterance_id(row: dict, uid_key: str | None) -> str | None:
    if uid_key:
        v = row.get(uid_key)
        if v:
            return _strip_seg(v)
    for k in ("parent_audio_path", "audio_path", "id"):
        v = row.get(k)
        if v:
            return _strip_seg(v)
    return None


def _iter_rows(val_data_dir: str, step: int | None):
    sources = [p.rstrip("/") for p in bf.glob(os.path.join(val_data_dir, "*/"))]
    if not sources:
        # Allow pointing directly at a data_source dir or a single file.
        sources = [val_data_dir.rstrip("/")]
    print(f"Scanning {len(sources)} dir(s) under {val_data_dir}", flush=True)
    for src_dir in sources:
        files = list(bf.glob(os.path.join(src_dir, "*.jsonl")))
        if not files and bf.exists(src_dir) and src_dir.endswith(".jsonl"):
            files = [src_dir]
        if not files:
            continue
        if step is not None:
            files = [f for f in files if os.path.basename(f) == f"{step}.jsonl"]
        else:
            def _step_of(p):
                try:
                    return int(os.path.basename(p).removesuffix(".jsonl"))
                except ValueError:
                    return -1
            files = [max(files, key=_step_of)]
        for f in files:
            with bf.BlobFile(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)


def convert(val_data_dir: str, step: int | None, uid_key: str | None) -> list[dict]:
    by_uid: dict[str, dict] = {}
    rows_seen = 0
    rows_with_detail = 0
    for row in _iter_rows(val_data_dir, step):
        rows_seen += 1
        uid = _utterance_id(row, uid_key)
        if not uid:
            continue
        detail = row.get("dter_detail")
        if not isinstance(detail, dict) or not detail.get("word_align"):
            # Keep the first seen uid even without a usable alignment so the
            # utterance is still represented; a later row may fill it in.
            by_uid.setdefault(uid, {"UtteranceId": uid, "UtteranceTERMetrics": []})
            continue
        rows_with_detail += 1
        if uid in by_uid and by_uid[uid]["UtteranceTERMetrics"]:
            continue  # already have this parent's per-group detail
        by_uid[uid] = {"UtteranceId": uid, "UtteranceTERMetrics": [detail]}

    records = list(by_uid.values())
    print(
        f"Read {rows_seen} rows ({rows_with_detail} with alignment) "
        f"-> {len(records)} unique utterances",
        flush=True,
    )
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-data-dir", required=True,
                   help="verl trainer val_data_gen dir, a data_source subdir, or a single .jsonl (az:// or local).")
    p.add_argument("--output", required=True,
                   help="Local JSON file to write (inhouse-asr-compare input format).")
    p.add_argument("--step", type=int, default=None,
                   help="Specific eval step jsonl to convert (default: latest per data_source).")
    p.add_argument("--uid-key", default=None,
                   help="Row field to use as UtteranceId (default: parent_audio_path -> audio_path -> id).")
    return p.parse_args()


def main():
    args = parse_args()
    records = convert(args.val_data_dir, args.step, args.uid_key)
    out = os.path.expanduser(args.output)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()

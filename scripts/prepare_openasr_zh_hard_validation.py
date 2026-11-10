"""Build error-ranked Mandarin validation manifests from pinned 2611r2 details."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import blobfile as bf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recipe.phimm.reward.openasr_expansion import openasr_expansion_eval


def read(path):
    with bf.BlobFile(str(path), "rb") as stream:
        raw = stream.read()
    return [json.loads(line) for line in raw.splitlines() if line.strip()], hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="reports/openasr_multilingual_expansion_20261107/comparison.json")
    parser.add_argument("--details-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--publish-root")
    parser.add_argument("--topk", type=int, default=500)
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be positive")
    comparison = json.loads(Path(args.comparison).read_text())
    model = comparison["models"]["2611r2"]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    counts = ("n_err", "n_ref", "n_ins", "n_del", "n_sub")
    names = {
        "zh_aishell1": ("aishell1_zh", f"aishell1_test_top{args.topk}_2611r2.jsonl", 7176),
        "zh_fleurs": ("fleurs_zh", f"fleurs_zh_test_top{min(args.topk, 945)}_2611r2.jsonl", 945),
    }
    for source, (folder, filename, expected) in names.items():
        details_path = Path(args.details_dir) / (source + ".source_details.jsonl")
        details, details_sha = read(details_path)
        manifest_path = f"az://orngwus2cresco/data/boren/data/openasr_ml_expansion/18143a6e/{folder}/data_zip.jsonl"
        original, manifest_sha = read(manifest_path)
        by_id = {row["id"]: row for row in original}
        assert len(details) == len(original) == len(by_id) == expected
        assert {row["id"] for row in details} == set(by_id)
        for row in details:
            human = by_id[row["id"]]
            assert row["gts"] == human["text"]
            assert row["audio_path"] == human["audio_path"]
            assert all(type(row[k]) is int and row[k] >= 0 for k in counts)
            assert row["n_ref"] > 0 and row["n_err"] == sum(row[k] for k in counts[2:])
        historical = model["datasets"][source]
        assert all(sum(row[k] for row in details) == historical["counts"][k] for k in counts)
        ranked = sorted(details, key=lambda row: (-row["n_err"], row["source_row"], row["id"]))
        selected, audit = [], []
        for rank, row in enumerate(ranked[:args.topk], 1):
            result = openasr_expansion_eval(row["output"], row["gts"], version=2609,
                                            extra_info={"language": "Chinese"})
            assert all(result[k] == row[k] for k in counts), row["id"]
            record = dict(by_id[row["id"]])
            record.update(selection_rank=rank, selection_n_err=row["n_err"],
                          selection_n_ref=row["n_ref"], selection_model=model["model_uri"],
                          source_split="test", diagnostic_subset=True,
                          source_manifest_sha256=manifest_sha)
            selected.append(record)
            audit.append({**record, "scored_hypothesis": row["output"],
                          **{k: row[k] for k in counts}})
        files = []
        for suffix, rows in [("", selected), (".audit.jsonl", audit)]:
            path = root / (filename + suffix)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            files.append(path)
        meta = {
            "source": source, "rows": len(selected), "candidate_rows": expected,
            "requested_topk": args.topk, "shortfall": max(0, args.topk - expected),
            "sort": ["n_err descending", "source_row ascending", "id ascending"],
            "source_model": model["model_uri"], "source_detail_path": historical["detail_path"],
            "source_detail_sha256": details_sha, "source_manifest": manifest_path,
            "source_manifest_sha256": manifest_sha,
            "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
            "selected_min_n_err": selected[-1]["selection_n_err"],
            "selected_max_n_err": selected[0]["selection_n_err"],
            "zero_error_rows_selected": sum(row["selection_n_err"] == 0 for row in selected),
            "selected_n_err": sum(row["selection_n_err"] for row in selected),
            "selected_n_ref": sum(row["selection_n_ref"] for row in selected),
            "all_selected_rows_rescored": True, "human_targets_retained": True,
            "audio_selectors_unchanged": True, "diagnostic_subset": True,
            "coverage_note": "Requested topk exceeds available rows; all available rows retained without duplication."
                if expected < args.topk else "Baseline-selected hard-case test subset; not full-test CER.",
        }
        path = root / (filename + ".meta.json")
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        files.append(path)
        if args.publish_root:
            for path in files:
                destination = args.publish_root.rstrip("/") + "/" + path.name
                if bf.exists(destination):
                    with bf.BlobFile(destination, "rb") as stream:
                        assert stream.read() == path.read_bytes(), destination
                else:
                    bf.copy(str(path), destination, overwrite=False)
                with bf.BlobFile(destination, "rb") as stream:
                    assert hashlib.sha256(stream.read()).digest() == hashlib.sha256(path.read_bytes()).digest()
        print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

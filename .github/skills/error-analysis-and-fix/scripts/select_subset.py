#!/usr/bin/env python3
"""Select auditable subsets from canonical JSONL adapters; never modifies labels."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path, canonical=True):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            require(isinstance(row, dict), f"{path}:{number}: expected object")
            key = row.get("id")
            require(isinstance(key, str) and key.strip(), f"{path}:{number}: invalid id")
            require(key not in rows, f"{path}: duplicate id {key}")
            if canonical:
                require(isinstance(row.get("text"), str), f"{key}: missing reference text")
                require(isinstance(row.get("audio_path"), str) and row["audio_path"].strip(),
                        f"{key}: missing utterance audio selector")
                require(row.get("source_split") in {"train", "dev", "validation", "test"},
                        f"{key}: invalid source_split")
                if "source_row" in row:
                    require(type(row["source_row"]) is int and row["source_row"] >= 0,
                            f"{key}: invalid source_row")
                if row.get("parent_id") is not None:
                    require(isinstance(row["parent_id"], str) and row["parent_id"].strip(),
                            f"{key}: invalid parent_id")
            rows[key] = row
    return rows


def text_key(text):
    return " ".join(text.split())


def run(args):
    require(args.expected_source_count >= 0, "expected count must be nonnegative")
    require(args.top_k > 0, "top-k must be positive")
    require(not args.output_dir.exists(), "output directory must be new")
    source = read_rows(args.source)
    require(len(source) == args.expected_source_count, "source count differs from expected universe")
    allowed = {"train"} if args.role == "train" else (
        {"dev", "validation"} if args.role == "validation" else {"dev", "validation", "test"})
    require(all(row["source_split"] in allowed for row in source.values()),
            f"source contains a split incompatible with role {args.role}")
    require(args.role != "train" or args.heldout, "train selection requires heldout manifests")
    require(args.rank != "errors" or args.scores, "error ranking requires complete scores")
    require(not args.scores or args.scorer_id, "scores require a pinned --scorer-id")
    require(args.scores or args.patterns, "supply scores or a feature pattern definition")
    minimum = args.min_errors if args.min_errors is not None else (1 if args.scores else 0)
    require(minimum >= 0, "min-errors must be nonnegative")
    require(args.scores or minimum == 0, "min-errors requires scores")

    scores = read_rows(args.scores, canonical=False) if args.scores else {}
    if args.scores:
        require(source.keys() == scores.keys(), "score IDs do not cover the exact source universe")
        for key, row in source.items():
            score = scores[key]
            for field in ("text", "audio_path"):
                require(score.get(field) == row[field], f"{key}: score {field} differs from source")
            require(isinstance(score.get("hypothesis"), str), f"{key}: missing hypothesis")
            for field in ("n_ref", "n_err", "n_ins", "n_del", "n_sub"):
                require(type(score.get(field)) is int and score[field] >= 0,
                        f"{key}: invalid {field}")
            require(score["n_err"] == sum(score[f] for f in ("n_ins", "n_del", "n_sub")),
                    f"{key}: edit counts disagree")
            for field in row.keys() & score.keys():
                require(row[field] == score[field], f"{key}: conflicting source/score field {field}")

    pattern_spec = None
    patterns = []
    if args.patterns:
        pattern_spec = json.loads(args.patterns.read_text(encoding="utf-8"))
        require(isinstance(pattern_spec, dict), "pattern definition must be an object")
        require(isinstance(pattern_spec.get("version"), str) and pattern_spec["version"].strip(),
                "pattern version is required")
        require(pattern_spec.get("mode") in {"any", "all"}, "pattern mode must be any or all")
        require(isinstance(pattern_spec.get("patterns"), list) and pattern_spec["patterns"],
                "patterns must be a nonempty list")
        names = set()
        for entry in pattern_spec["patterns"]:
            require(isinstance(entry, dict), "each pattern must be an object")
            for field in ("id", "field", "regex"):
                require(isinstance(entry.get(field), str) and entry[field], f"pattern needs {field}")
            require(entry["id"] not in names, "duplicate pattern id")
            names.add(entry["id"])
            patterns.append((entry["id"], entry["field"], re.compile(entry["regex"])))

    heldout_rows = []
    for path in args.heldout:
        protected = read_rows(path)
        require(all(r["source_split"] != "train" for r in protected.values()),
                "heldout manifest contains train rows")
        heldout_rows.extend(protected.values())
    protected_keys = {
        "id": {r["id"] for r in heldout_rows},
        "audio_path": {r["audio_path"] for r in heldout_rows},
        "parent_id": {r["parent_id"] for r in heldout_rows if r.get("parent_id")},
    }
    protected_text = {text_key(r["text"]) for r in heldout_rows}
    exclusions = {"heldout_overlap": 0, "below_min_errors": 0, "pattern_miss": 0}
    overlap_fields = {field: 0 for field in protected_keys}
    overlap_fields["text"] = 0
    candidates = []
    for key, row in source.items():
        combined = dict(row, **{k: v for k, v in scores.get(key, {}).items() if k not in row})
        hits = []
        # Validate searched fields for every row, even those later excluded.
        for name, field, regex in patterns:
            value = combined.get(field)
            require(isinstance(value, (str, int, float, bool)), f"{key}: missing/non-scalar {field}")
            if regex.search(str(value)):
                hits.append(name)
        overlaps = [field for field, values in protected_keys.items()
                    if row.get(field) is not None and row[field] in values]
        if args.exclude_same_text and text_key(row["text"]) in protected_text:
            overlaps.append("text")
        if overlaps:
            exclusions["heldout_overlap"] += 1
            for field in overlaps:
                overlap_fields[field] += 1
            continue
        if args.scores and scores[key]["n_err"] < minimum:
            exclusions["below_min_errors"] += 1
            continue
        if patterns and not (len(hits) == len(patterns) if pattern_spec["mode"] == "all" else hits):
            exclusions["pattern_miss"] += 1
            continue
        candidates.append((row, hits))
    if args.rank == "errors":
        candidates.sort(key=lambda item: (-scores[item[0]["id"]]["n_err"],
                                         item[0].get("source_row", sys.maxsize), item[0]["id"]))
    else:
        candidates.sort(key=lambda item: (hashlib.sha256(
            (args.seed + "\0" + item[0]["id"]).encode()).hexdigest(), item[0]["id"]))
    require(len(candidates) >= args.top_k,
            f"only {len(candidates)} eligible rows for top-k={args.top_k}; refusing padding")
    selected = candidates[:args.top_k]
    audit = []
    for rank, (row, hits) in enumerate(selected, 1):
        score = scores.get(row["id"])
        confirmed = score is not None and score["n_err"] > 0
        if score is None:
            evidence = ("reference_feature_candidate" if all(field == "text" for _, field, _ in patterns)
                        else "feature_candidate")
        elif patterns:
            evidence = "hybrid_confirmed" if confirmed else "feature_candidate_scored_correct"
        else:
            evidence = "model_confirmed" if confirmed else "scored_correct_candidate"
        audit.append({"id": row["id"], "rank": rank, "pattern_hits": hits,
                      "evidence_type": evidence,
                      "score": score})
    inputs = [args.source] + ([args.scores] if args.scores else []) + (
        [args.patterns] if args.patterns else []) + args.heldout
    summary = {
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in inputs],
        "source_count": len(source), "score_count": len(scores),
        "eligible_count": len(candidates), "selected_count": len(selected),
        "role": args.role, "rank": args.rank, "min_errors": minimum, "seed": args.seed,
        "scorer_id": args.scorer_id, "patterns": pattern_spec,
        "tie_break": "source_row,id" if args.rank == "errors" else "sha256(seed,NUL,id),id",
        "exclusions": exclusions, "overlap_field_counts": overlap_fields,
        "heldout_count": len(heldout_rows), "exclude_same_text": args.exclude_same_text,
        "missing_parent_source_count": sum(not r.get("parent_id") for r in source.values()),
        "missing_parent_heldout_count": sum(not r.get("parent_id") for r in heldout_rows),
        "overlap_limits": "Exact canonical identities only; content/aliases not checked here.",
        "membership_sha256": hashlib.sha256(json.dumps(
            sorted(r["id"] for r, _ in selected), ensure_ascii=False).encode()).hexdigest(),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".subset-", dir=args.output_dir.parent))
    try:
        for name, records in (("manifest.jsonl", [r for r, _ in selected]), ("audit.jsonl", audit)):
            with (temporary / name).open("w", encoding="utf-8") as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["manifest_sha256"] = digest(temporary / "manifest.jsonl")
        summary["audit_sha256"] = digest(temporary / "audit.jsonl")
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        require(not args.output_dir.exists(), "output directory appeared during selection")
        temporary.rename(args.output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"output_dir": str(args.output_dir), "selected_count": len(selected),
                      "membership_sha256": summary["membership_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-source-count", required=True, type=int)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--scorer-id")
    parser.add_argument("--patterns", type=Path)
    parser.add_argument("--rank", required=True, choices=("errors", "deterministic"))
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--min-errors", type=int)
    parser.add_argument("--role", required=True, choices=("train", "validation", "diagnostic"))
    parser.add_argument("--heldout", nargs="+", default=[], type=Path)
    parser.add_argument("--exclude-same-text", action="store_true")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, re.error) as error:
        parser.exit(2, f"selection failed: {error}\n")


if __name__ == "__main__":
    main()

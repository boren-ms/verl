"""Summarize per-dataset validation JSONL using corpus WER/BWER/UWER counts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def ratio(n, d):
    return n / d if d else None


def summarize(paths):
    groups = defaultdict(list)
    seen = set()
    for path in paths:
        for line in Path(path).open():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["data_source"], row.get("id"), row.get("step"))
            if key in seen:
                raise ValueError(f"duplicate result: {key}")
            seen.add(key)
            if row["n_err"] != row["nb_err"] + row["nu_err"] or row["n_ref"] != row["nb_ref"] + row["nu_ref"]:
                raise ValueError(f"WER partition mismatch: {key}")
            groups[row["data_source"]].append(row)
    result = {}
    for source, rows in groups.items():
        if len({r.get("step") for r in rows}) > 1:
            raise ValueError(f"select one checkpoint step per dataset: {source}")
        sums = {k: sum(r[k] for r in rows) for k in ("n_err", "n_ref", "nb_err", "nb_ref", "nu_err", "nu_ref")}
        output = {
            "utterances": len(rows),
            **sums,
            "WER": ratio(sums["n_err"], sums["n_ref"]),
            "BWER": ratio(sums["nb_err"], sums["nb_ref"]),
            "UWER": ratio(sums["nu_err"], sums["nu_ref"]),
            "NWER": ratio(sums["nu_err"], sums["nu_ref"]),
        }
        output["keyword_word_accuracy"] = None if output["BWER"] is None else 1 - output["BWER"]
        if "kw_list_f1" in rows[0]:
            tp = sum(r["kw_tp"] for r in rows)
            fp = sum(r["kw_fp"] for r in rows)
            fn = sum(r["kw_fn"] for r in rows)
            output.update(
                kw_micro_precision=ratio(tp, tp + fp),
                kw_micro_recall=ratio(tp, tp + fn),
                kw_micro_f1=ratio(2 * tp, 2 * tp + fp + fn),
                kw_macro_f1=sum(r["kw_list_f1"] for r in rows) / len(rows),
                kw_exact=sum(r["kw_list_exact"] for r in rows) / len(rows),
                kw_format_success=sum(r["kw_block_fmt"] for r in rows) / len(rows),
                kw_empty_false_positive=ratio(
                    sum(r["kw_empty_false_positive"] for r in rows), sum(r["kw_empty_target"] for r in rows)
                ),
            )
        result[source] = output
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

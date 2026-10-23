"""Freeze an easy training-validation subset from an untrained step-0 dev evaluation."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


COUNTS = ("n_err", "n_ref", "nb_err", "nb_ref", "nu_err", "nu_ref")


def read_unique(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"duplicate id: {row['id']}")
        rows[row["id"]] = row
    return rows


def select(manifest_path, baseline_path, count=1000, expected_count=6096):
    manifest = read_unique(manifest_path)
    baseline = read_unique(baseline_path)
    if len(manifest) != expected_count or manifest.keys() != baseline.keys():
        raise ValueError("baseline must cover the entire dev manifest exactly")
    eligible = []
    for uid, row in baseline.items():
        original = manifest[uid]
        if original.get("split") != "dev" or row.get("split") != "dev":
            raise ValueError(f"dev only; test/train records are forbidden: {uid}")
        if row.get("data_source") != "ls_dev" or row.get("step") != 0:
            raise ValueError(f"expected untrained step-0 ls_dev baseline: {uid}")
        if row.get("gts") != original["text"] or row.get("keywords") != original["keywords"]:
            raise ValueError(f"reference or keyword mismatch: {uid}")
        for key in ("source_manifest_sha", "annotation_policy_sha"):
            if not original.get(key) or row.get(key) != original[key]:
                raise ValueError(f"{key} mismatch: {uid}")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in COUNTS):
            raise ValueError(f"invalid error/reference counts: {uid}")
        if (row["n_err"] != row["nb_err"] + row["nu_err"]
                or row["n_ref"] != row["nb_ref"] + row["nu_ref"] or not row["n_ref"]):
            raise ValueError(f"invalid WER partition: {uid}")
        if row["nb_ref"]:
            eligible.append(row)
    if not 0 < count <= len(eligible):
        raise ValueError("requested size exceeds eligible dev utterances with defined BWER")
    ranked = sorted(eligible, key=lambda r: (
        Fraction(r["nb_err"], r["nb_ref"]), Fraction(r["n_err"], r["n_ref"]), r["id"],
    ))[:count]
    sums = {key: sum(row[key] for row in ranked) for key in COUNTS}
    provenance = {
        "selection": "ascending (nb_err/nb_ref, n_err/n_ref, id), exact rational ordering",
        "purpose": "training validation and checkpoint selection only; easy subset, not unbiased evaluation",
        "baseline_step": 0,
        "baseline_data_source": "ls_dev",
        "source_count": len(manifest),
        "eligible_count": len(eligible),
        "excluded_undefined_bwer": len(manifest) - len(eligible),
        "selected_count": count,
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
        "baseline_corpus": {**sums, "WER": sums["n_err"] / sums["n_ref"],
                            "BWER": sums["nb_err"] / sums["nb_ref"],
                            "UWER": sums["nu_err"] / sums["nu_ref"] if sums["nu_ref"] else None},
        "selected": [{key: row[key] for key in ("id", *COUNTS)} for row in ranked],
    }
    return [manifest[row["id"]] for row in ranked], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--expected-count", type=int, default=6096)
    args = parser.parse_args()
    if args.output.exists() or args.provenance.exists() or args.output == args.provenance:
        parser.error("use new, distinct output and provenance paths; refusing to overwrite")
    rows, provenance = select(args.manifest, args.baseline, args.count, args.expected_count)
    output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    provenance["output_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "selected"}, indent=2))


if __name__ == "__main__":
    main()

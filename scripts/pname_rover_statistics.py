"""Summarize verified Rover generation, canonical keyword edits, and source names.

Run with ``python -m scripts.pname_rover_statistics --base <selection-directory>``.
The original source manifest must exist at the path recorded in selection_summary.json.
"""

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import blobfile as bf
from jiwer import process_words
import numpy as np

from recipe.phimm.reward.asr_edge import _norm_text
from recipe.phimm.reward.asr_measure import _keyword_indices


@lru_cache(maxsize=100000)
def normalize(text):
    return _norm_text(text, name="hf_english", lang="english")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def masked_edits(chunks, ref_indices, hyp_indices):
    counts = Counter(substitution=0, deletion=0, insertion=0)
    ref_errors = set()
    for chunk in chunks:
        if chunk.type in ("substitute", "delete"):
            errors = ref_indices.intersection(range(chunk.ref_start_idx, chunk.ref_end_idx))
            counts["substitution" if chunk.type == "substitute" else "deletion"] += len(errors)
            ref_errors.update(errors)
        elif chunk.type == "insert":
            counts["insertion"] += len(hyp_indices.intersection(range(chunk.hyp_start_idx, chunk.hyp_end_idx)))
    return counts, ref_errors


def aggregate(rows):
    counts = {k: sum(r[k] for r in rows) for k in ("n_err", "n_ref", "nb_err", "nb_ref", "name_err", "name_ref")}
    return {
        "records": len(rows),
        "audio_hours": sum(r["duration_seconds"] for r in rows) / 3600,
        "counts": counts,
        "wer": counts["n_err"] / max(counts["n_ref"], 1),
        "keyword_word_error_rate": counts["nb_err"] / max(counts["nb_ref"], 1),
        "source_names_word_error_rate": counts["name_err"] / max(counts["name_ref"], 1),
        "records_with_word_errors": sum(r["n_err"] > 0 for r in rows),
        "records_with_keyword_errors": sum(r["nb_err"] > 0 for r in rows),
        "records_with_source_name_errors": sum(r["name_err"] > 0 for r in rows),
        "records_without_reference_keyword_match": sum(r["nb_ref"] == 0 for r in rows),
        "response_at_limit": sum(r["response_at_limit"] for r in rows),
        "duration_seconds": distribution([r["duration_seconds"] for r in rows]),
        "keyword_errors_per_record": distribution([r["nb_err"] for r in rows]),
        "reference_words_per_record": distribution([r["n_ref"] for r in rows]),
        "response_tokens": distribution([r["response_tokens"] for r in rows]),
        "keyword_error_histogram": dict(sorted(Counter(r["nb_err"] for r in rows).items())),
        "keyword_edit_types": {
            k: sum(r["keyword_" + k] for r in rows) for k in ("substitution", "deletion", "insertion")
        },
    }


def run(base):
    base = Path(base).resolve()
    summary = json.loads((base / "selection_summary.json").read_text())
    out = base / "statistics"
    out.mkdir(exist_ok=True)
    source_path = Path(summary["source"])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == summary["source_sha256"]
    source = {r["id"]: r for r in map(json.loads, source_path.read_text().splitlines())}
    generated = []
    for part in summary["parts"]:
        local = base / "generation" / bf.basename(part["path"])
        local.parent.mkdir(exist_ok=True)
        if not local.exists():
            bf.copy(part["path"], str(local))
        assert hashlib.sha256(local.read_bytes()).hexdigest() == part["sha256"]
        records = [json.loads(line) for line in local.read_text().splitlines()]
        assert len(records) == part["rows"]
        generated.extend(records)
    assert len(generated) == len(source) == summary["expected_records"]
    assert len({r["id"] for r in generated}) == len(generated)
    assert {r["id"] for r in generated} == set(source)
    selected_ids = {json.loads(line)["id"] for line in (base / "keyword_errors.jsonl").read_text().splitlines()}
    assert selected_ids == {r["id"] for r in generated if r["nb_err"] > 0}

    per_record = []
    phrases = defaultdict(Counter)
    confusions = Counter()
    confusion_ids = defaultdict(set)
    audit = Counter()
    for index, row in enumerate(generated, 1):
        original = source[row["id"]]
        ref = normalize(row["text"]).split()
        hyp = normalize(row["response"]).split()
        chunks = process_words(" ".join(ref), " ".join(hyp)).alignments[0]
        keywords = [normalize(k).split() for k in row["keywords"]]
        ref_indices = _keyword_indices(ref, keywords)
        hyp_indices = _keyword_indices(hyp, keywords)
        edits, ref_errors = masked_edits(chunks, ref_indices, hyp_indices)
        assert sum(edits.values()) == row["nb_err"], row["id"]
        assert len(ref_indices) == row["nb_ref"], row["id"]
        names = [normalize(k).split() for k in original.get("names", [])]
        name_indices = _keyword_indices(ref, names)
        name_edits, _ = masked_edits(chunks, name_indices, _keyword_indices(hyp, names))
        audit["rows_without_reference_name_match"] += not bool(name_indices)
        audit["source_display_updated_text_differ_raw"] += original["display"] != original["updated_text"]
        audit["source_display_updated_text_differ_normalized"] += normalize(original["display"]) != normalize(
            original["updated_text"]
        )
        for keyword in sorted(set(" ".join(k) for k in keywords if k)):
            words = keyword.split()
            starts = [i for i in range(len(ref) - len(words) + 1) if ref[i : i + len(words)] == words]
            entry = phrases[keyword]
            entry["rows_declared"] += 1
            entry["rows_matched"] += bool(starts)
            entry["reference_occurrences"] += len(starts)
            error_tokens = sum(len(ref_errors.intersection(range(i, i + len(words)))) for i in starts)
            entry["reference_tokens"] += len(starts) * len(words)
            entry["reference_token_errors"] += error_tokens
            entry["rows_with_reference_error"] += error_tokens > 0
            entry["errored_reference_occurrences"] += sum(
                bool(ref_errors.intersection(range(i, i + len(words)))) for i in starts
            )
        for chunk in chunks:
            if chunk.type in ("substitute", "delete"):
                for i in sorted(ref_indices.intersection(range(chunk.ref_start_idx, chunk.ref_end_idx))):
                    if chunk.type == "delete":
                        pair = (ref[i], "<deleted>", "deletion")
                    elif chunk.ref_end_idx - chunk.ref_start_idx == chunk.hyp_end_idx - chunk.hyp_start_idx:
                        pair = (ref[i], hyp[chunk.hyp_start_idx + i - chunk.ref_start_idx], "substitution")
                    else:
                        pair = (ref[i], " ".join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]), "substitution_span")
                    confusions[pair] += 1
                    confusion_ids[pair].add(row["id"])
            elif chunk.type == "insert":
                for i in sorted(hyp_indices.intersection(range(chunk.hyp_start_idx, chunk.hyp_end_idx))):
                    pair = ("<inserted>", hyp[i], "insertion")
                    confusions[pair] += 1
                    confusion_ids[pair].add(row["id"])
        audio = row["audio_path"]
        parent = audio.split("/ChunkFiles/")[0].split("/speech/am_data/")[-1]
        per_record.append(
            {
                "id": row["id"],
                "selected": row["nb_err"] > 0,
                "source_group": parent,
                "duration_seconds": row["audio_duration_seconds"],
                **{
                    k: row[k]
                    for k in ("n_err", "n_ref", "nb_err", "nb_ref", "wer", "response_tokens", "response_at_limit")
                },
                "name_err": sum(name_edits.values()),
                "name_ref": len(name_indices),
                **{"keyword_" + k: v for k, v in edits.items()},
            }
        )
        if index % 2000 == 0:
            print(f"Verified keyword alignments {index}/{len(generated)}", flush=True)

    selected = [r for r in per_record if r["selected"]]
    clean = [r for r in per_record if not r["selected"]]
    result = {
        "model": summary["model"],
        "source_sha256": summary["source_sha256"],
        "generation_parts": summary["parts"],
        "definitions": {
            "selection": "canonical keyword nb_err > 0; membership unchanged",
            "wer": "micro-averaged canonical compound-aware English WER from generation",
            "keyword_errors": (
                "HF-English normalized word alignment; keyword-token substitutions, deletions "
                "and matched-keyword insertions"
            ),
            "source_names": (
                "Same word alignment scored against original source names field, "
                "not independently verified person-name labels"
            ),
            "per_keyword": (
                "Reference occurrences only; overlapping keyword phrases may share tokens "
                "and must not be summed; insertions separate"
            ),
            "confusions": (
                "Normalized aligned keyword tokens; spelling alternatives and reference mistakes "
                "need human/audio review"
            ),
        },
        "full": aggregate(per_record),
        "selected": aggregate(selected),
        "not_selected": aggregate(clean),
        "audit": dict(audit),
        "source_names_selection_overlap": {
            "keyword_and_source_name_errors": sum(r["nb_err"] > 0 and r["name_err"] > 0 for r in per_record),
            "keyword_errors_only": sum(r["nb_err"] > 0 and r["name_err"] == 0 for r in per_record),
            "source_name_errors_only": sum(r["nb_err"] == 0 and r["name_err"] > 0 for r in per_record),
        },
        "limitations": [
            "Baseline-selected validation diagnostic; no causal improvement claim.",
            "Supplied updated references and names have not been independently checked against audio.",
            "Keywords absent from reference do not contribute reference keyword tokens.",
            "WER uses compound-aware scoring; keyword counts use ordinary word alignment, "
            "so their error counts are not interchangeable.",
        ],
    }
    assert result["full"]["counts"]["nb_err"] == summary["counts"]["nb_err"]
    assert result["full"]["counts"]["nb_ref"] == summary["counts"]["nb_ref"]
    assert sum(confusions.values()) == summary["counts"]["nb_err"]
    result["keyword_reference_audit"] = summary["keyword_reference_audit"]
    groups = defaultdict(list)
    for row in per_record:
        groups[row["source_group"]].append(row)
    group_rows = []
    for group, rows in groups.items():
        a = aggregate(rows)
        group_rows.append(
            {
                "source_group": group,
                "records": len(rows),
                "selected_records": a["records_with_keyword_errors"],
                "selection_rate": a["records_with_keyword_errors"] / len(rows),
                "wer": a["wer"],
                "keyword_word_error_rate": a["keyword_word_error_rate"],
                "keyword_errors": a["counts"]["nb_err"],
                "keyword_reference_tokens": a["counts"]["nb_ref"],
                "audio_hours": a["audio_hours"],
            }
        )
    phrase_rows = [
        {"keyword": k, **v, "reference_token_error_rate": v["reference_token_errors"] / max(v["reference_tokens"], 1)}
        for k, v in phrases.items()
    ]
    phrase_rows.sort(key=lambda r: (-r["reference_token_errors"], -r["reference_occurrences"], r["keyword"]))
    confusion_rows = [
        {
            "reference": k[0],
            "hypothesis": k[1],
            "type": k[2],
            "keyword_token_errors": v,
            "records": len(confusion_ids[k]),
            "example_id": sorted(confusion_ids[k])[0],
        }
        for k, v in sorted(confusions.items(), key=lambda x: (-x[1], x[0]))
    ]
    write_csv(out / "per_record.csv", per_record)
    write_csv(out / "keywords.csv", phrase_rows)
    write_csv(out / "confusions.csv", confusion_rows)
    write_csv(out / "source_groups.csv", sorted(group_rows, key=lambda r: (-r["keyword_errors"], r["source_group"])))
    (out / "statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    render_report(out, result, confusion_rows)
    print(
        json.dumps(
            {k: result[k] for k in ("full", "selected", "not_selected", "source_names_selection_overlap")}, indent=2
        )
    )
    return result


def render_report(out, stats, confusions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full, selected = stats["full"], stats["selected"]
    edits = full["keyword_edit_types"]
    hist = {int(k): v for k, v in selected["keyword_error_histogram"].items()}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = ["#2b6cb0", "#e69f00", "#009e73"]
    labels = ["Full set", "Keyword-error subset"]
    axes[0].bar(labels, [full["wer"] * 100, selected["wer"] * 100], color=colors[:2])
    axes[0].set_ylabel("Canonical WER (%)")
    axes[0].set_title("Word error rate")
    axes[1].bar(["Substitution", "Deletion", "Insertion"], list(edits.values()), color=colors)
    axes[1].set_ylabel("Keyword-token errors")
    axes[1].set_title("Keyword error types")
    bins = {
        "1": hist.get(1, 0),
        "2": hist.get(2, 0),
        "3": hist.get(3, 0),
        "4+": sum(v for k, v in hist.items() if k >= 4),
    }
    axes[2].bar(list(bins), list(bins.values()), color=colors[0])
    axes[2].set_xlabel("Keyword-token errors per selected record")
    axes[2].set_ylabel("Records")
    axes[2].set_title("Selected-record severity")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        for bar in ax.patches:
            ax.annotate(
                f"{bar.get_height():.2f}" if ax is axes[0] else str(int(bar.get_height())),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.margins(y=0.15)
    fig.suptitle(
        "pname Rover v1 · keyword-error statistics\n"
        f"{full['records']:,} records; {selected['records']:,} selected by keyword nb_err > 0",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out / "statistics.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "statistics.pdf", bbox_inches="tight")
    plt.close(fig)
    text = "# pname Rover v1 generation statistics\n\n"
    text += f"Model: {stats['model']}\n\nSelected membership: canonical keyword `nb_err > 0`.\n\n"
    text += "| Metric | Full validation | Keyword-error subset | Remaining records |\n|---|---:|---:|---:|\n"
    for label, fn in [
        ("Records", lambda a: f"{a['records']:,}"),
        ("Hours", lambda a: f"{a['audio_hours']:.2f}"),
        ("WER", lambda a: f"{a['wer']:.2%}"),
        ("Keyword word error rate", lambda a: f"{a['keyword_word_error_rate']:.2%}"),
        ("Keyword errors / reference tokens", lambda a: f"{a['counts']['nb_err']:,} / {a['counts']['nb_ref']:,}"),
        ("Source-names word error rate", lambda a: f"{a['source_names_word_error_rate']:.2%}"),
        ("Mean duration, seconds", lambda a: f"{a['duration_seconds']['mean']:.2f}"),
    ]:
        text += "| " + " | ".join([label] + [fn(stats[k]) for k in ("full", "selected", "not_selected")]) + " |\n"
    text += (
        "\nKeyword edit counts: "
        + ", ".join(f"{k}: {v:,} ({v / full['counts']['nb_err']:.2%})" for k, v in edits.items())
        + ".\n\n"
    )
    text += (
        "Selected-record keyword-error histogram: "
        + ", ".join(f"{k} errors: {v} records" for k, v in sorted(hist.items()))
        + ".\n\n"
    )
    overlap = stats["source_names_selection_overlap"]
    text += (
        f"Of {selected['records']:,} selected records, "
        f"{overlap['keyword_and_source_name_errors']:,} also have errors scored on the source names field, "
        f"and {overlap['keyword_errors_only']:,} have keyword errors without a scored name error. "
        f"Across the full set, {stats['audit']['rows_without_reference_name_match']:,} rows have no "
        "reference match for any supplied name. Name lists are preserved source annotations, "
        "not manually verified labels.\n\n"
    )
    text += (
        "Most frequent normalized keyword confusions:\n\n"
        "| Reference | Hypothesis | Errors | Records |\n|---|---|---:|---:|\n"
    )
    for c in confusions[:15]:
        text += f"| `{c['reference']}` | `{c['hypothesis']}` | {c['keyword_token_errors']} | {c['records']} |\n"
    text += "\n" + "\n".join("- " + note for note in stats["limitations"]) + "\n"
    text += (
        "\nThe full keyword/reference audit and separate source-names overlap are in statistics.json. "
        "CSVs contain per-record results, per-keyword reference errors, normalized confusion pairs "
        "and source-group summaries.\n"
    )
    (out / "statistics.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="Directory with selection_summary.json and selected JSONL."
    )
    run(parser.parse_args().base)

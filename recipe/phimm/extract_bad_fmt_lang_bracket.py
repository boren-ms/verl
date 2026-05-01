"""Extract utterances where p_fmt or p_lang is not correct, or has brackets, from eval_openasr_ml results.

Computes WER and edge WER per dataset on the bad utterances and uploads them
to az://orngwus2cresco/data/boren/data/openasr_ml_jsonl/<corpus>/<lang>/bad_fmt_lang_bracket.jsonl
"""

import json
import re
import blobfile as bf

BRACKET_PATTERN = re.compile(r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}|\([^)]*\)")

VAL_DATA_ROOT = (
    "az://orngwus2cresco/data/boren/outputs/verl_repeat/"
    "eval_openasr_ml/val_data_gen"
)
UPLOAD_ROOT = "az://orngwus2cresco/data/boren/data/openasr_ml_jsonl"

# Map val_data_gen directory names to openasr_ml_jsonl subdirectory paths
DS_NAME_MAP = {
    "openasr_ml_fleurs_de": "fleurs/de",
    "openasr_ml_fleurs_fr": "fleurs/fr",
    "openasr_ml_fleurs_it": "fleurs/it",
    "openasr_ml_fleurs_es": "fleurs/es",
    "openasr_ml_fleurs_pt": "fleurs/pt",
    "openasr_ml_mcv_de": "mcv/de",
    "openasr_ml_mcv_fr": "mcv/fr",
    "openasr_ml_mcv_it": "mcv/it",
    "openasr_ml_mcv_es": "mcv/es",
    "openasr_ml_mcv_pt": "mcv/pt",
    "openasr_ml_mls_fr": "mls/fr",
    "openasr_ml_mls_it": "mls/it",
    "openasr_ml_mls_es": "mls/es",
    "openasr_ml_mls_pt": "mls/pt",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with bf.BlobFile(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def has_bracket(r: dict) -> bool:
    """Return True if clean_output contains bracketed text."""
    return bool(BRACKET_PATTERN.search(r.get("clean_output", "")))


def is_bad(r: dict) -> bool:
    """Return True if utterance has bad fmt, bad lang, or brackets in clean_output."""
    return (
        r.get("p_fmt", 1.0) < 1.0
        or r.get("p_lang", 1.0) < 1.0
        or has_bracket(r)
    )


def main():
    datasets = sorted(bf.listdir(VAL_DATA_ROOT))
    print(f"Found {len(datasets)} datasets: {datasets}\n")

    total_bad = 0
    summary_rows = []

    for ds in datasets:
        ds_path = bf.join(VAL_DATA_ROOT, ds, "0.jsonl")
        rows = load_jsonl(ds_path)

        bad_rows = [r for r in rows if is_bad(r)]

        n_total = len(rows)
        n_bad = len(bad_rows)

        # Overall WER
        n_err_all = sum(r.get("n_err", 0) for r in rows)
        n_ref_all = sum(r.get("n_ref", 1) for r in rows)
        n_edge_all = sum(r.get("n_edge", 0) for r in rows)
        wer_all = n_err_all / n_ref_all * 100 if n_ref_all else 0
        edge_wer_all = n_edge_all / n_ref_all * 100 if n_ref_all else 0

        if n_bad == 0:
            summary_rows.append({
                "dataset": ds,
                "total": n_total,
                "bad_count": 0,
                "bad_pct": 0.0,
                "bad_wer": "-",
                "bad_edge_wer": "-",
                "all_wer": f"{wer_all:.2f}%",
                "all_edge_wer": f"{edge_wer_all:.2f}%",
            })
            print(f"{ds}: {n_total} utterances, 0 bad p_fmt/p_lang/bracket")
            continue

        # Compute WER on bad rows
        n_err_bad = sum(r.get("n_err", 0) for r in bad_rows)
        n_ref_bad = sum(r.get("n_ref", 1) for r in bad_rows)
        n_edge_bad = sum(r.get("n_edge", 0) for r in bad_rows)
        wer_bad = n_err_bad / n_ref_bad * 100 if n_ref_bad else 0
        edge_wer_bad = n_edge_bad / n_ref_bad * 100 if n_ref_bad else 0

        # Count breakdown
        n_bad_fmt = sum(1 for r in bad_rows if r.get("p_fmt", 1.0) < 1.0)
        n_bad_lang = sum(1 for r in bad_rows if r.get("p_lang", 1.0) < 1.0)
        n_bad_bracket = sum(1 for r in bad_rows if has_bracket(r))

        print(f"{ds}: {n_total} total, {n_bad} bad ({n_bad/n_total*100:.1f}%) "
              f"[fmt={n_bad_fmt}, lang={n_bad_lang}, bracket={n_bad_bracket}]")
        print(f"  Bad WER: {wer_bad:.2f}%, Bad Edge WER: {edge_wer_bad:.2f}%")
        print(f"  All WER: {wer_all:.2f}%, All Edge WER: {edge_wer_all:.2f}%")

        summary_rows.append({
            "dataset": ds,
            "total": n_total,
            "bad_count": n_bad,
            "bad_fmt": n_bad_fmt,
            "bad_lang": n_bad_lang,
            "bad_bracket": n_bad_bracket,
            "bad_pct": round(n_bad / n_total * 100, 2),
            "bad_wer": f"{wer_bad:.2f}%",
            "bad_edge_wer": f"{edge_wer_bad:.2f}%",
            "all_wer": f"{wer_all:.2f}%",
            "all_edge_wer": f"{edge_wer_all:.2f}%",
        })

        total_bad += n_bad

        # Upload bad cases
        upload_subdir = DS_NAME_MAP.get(ds, ds)
        upload_path = bf.join(UPLOAD_ROOT, upload_subdir, "bad_fmt_lang_bracket.jsonl")
        print(f"  Uploading {n_bad} rows to {upload_path}")
        with bf.BlobFile(upload_path, "w") as out:
            for r in bad_rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary table
    print(f"\n{'='*110}")
    print(f"SUMMARY: {total_bad} total bad utterances across all datasets")
    print(f"{'='*110}")
    header = f"{'Dataset':<20} {'Total':>7} {'Bad':>6} {'Bad%':>7} {'Bad WER':>10} {'Bad EdgeWER':>12} {'All WER':>10} {'All EdgeWER':>12}"
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        print(f"{s['dataset']:<20} {s['total']:>7} {s['bad_count']:>6} {s['bad_pct']:>6.1f}% {s['bad_wer']:>10} {s['bad_edge_wer']:>12} {s['all_wer']:>10} {s['all_edge_wer']:>12}")


if __name__ == "__main__":
    main()

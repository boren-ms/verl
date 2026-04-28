"""Extract utterances where p_fmt or p_lang is not correct from eval_openasr results.

Computes WER and edge WER per dataset on the bad utterances and uploads them
to az://orngwus2cresco/data/boren/data/openasr_jsonl/<dataset>/bad_fmt_lang.jsonl
"""

import json
import blobfile as bf

VAL_DATA_ROOT = (
    "az://orngwus2cresco/data/boren/outputs/verl_repeat/"
    "eval_openasr/val_data_gen"
)
UPLOAD_ROOT = "az://orngwus2cresco/data/boren/data/openasr_jsonl"

# Map data_source names to openasr_jsonl subdirectory names
DS_NAME_MAP = {
    "ls_clean": "librispeech",
    "ls_other": "librispeech",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with bf.BlobFile(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    datasets = sorted(bf.listdir(VAL_DATA_ROOT))
    print(f"Found {len(datasets)} datasets: {datasets}\n")

    total_bad = 0
    summary_rows = []

    for ds in datasets:
        ds_path = bf.join(VAL_DATA_ROOT, ds, "0.jsonl")
        rows = load_jsonl(ds_path)

        bad_rows = [r for r in rows if r.get("p_fmt", 1.0) < 1.0 or r.get("p_lang", 1.0) < 1.0]

        n_total = len(rows)
        n_bad = len(bad_rows)

        if n_bad == 0:
            n_err = sum(r.get("n_err", 0) for r in rows)
            n_ref = sum(r.get("n_ref", 1) for r in rows)
            n_edge = sum(r.get("n_edge", 0) for r in rows)
            wer_all = n_err / n_ref * 100 if n_ref else 0
            edge_wer_all = n_edge / n_ref * 100 if n_ref else 0
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
            print(f"{ds}: {n_total} utterances, 0 bad p_fmt/p_lang")
            continue

        # Compute WER on bad rows
        n_err_bad = sum(r.get("n_err", 0) for r in bad_rows)
        n_ref_bad = sum(r.get("n_ref", 1) for r in bad_rows)
        n_edge_bad = sum(r.get("n_edge", 0) for r in bad_rows)
        wer_bad = n_err_bad / n_ref_bad * 100 if n_ref_bad else 0
        edge_wer_bad = n_edge_bad / n_ref_bad * 100 if n_ref_bad else 0

        # Overall WER for comparison
        n_err_all = sum(r.get("n_err", 0) for r in rows)
        n_ref_all = sum(r.get("n_ref", 1) for r in rows)
        n_edge_all = sum(r.get("n_edge", 0) for r in rows)
        wer_all = n_err_all / n_ref_all * 100 if n_ref_all else 0
        edge_wer_all = n_edge_all / n_ref_all * 100 if n_ref_all else 0

        # Count breakdown
        n_bad_fmt = sum(1 for r in bad_rows if r.get("p_fmt", 1.0) < 1.0)
        n_bad_lang = sum(1 for r in bad_rows if r.get("p_lang", 1.0) < 1.0)
        n_bad_both = sum(1 for r in bad_rows if r.get("p_fmt", 1.0) < 1.0 and r.get("p_lang", 1.0) < 1.0)

        print(f"{ds}: {n_total} total, {n_bad} bad ({n_bad/n_total*100:.1f}%) "
              f"[fmt={n_bad_fmt}, lang={n_bad_lang}, both={n_bad_both}]")
        print(f"  Bad WER: {wer_bad:.2f}%, Bad Edge WER: {edge_wer_bad:.2f}%")
        print(f"  All WER: {wer_all:.2f}%, All Edge WER: {edge_wer_all:.2f}%")

        summary_rows.append({
            "dataset": ds,
            "total": n_total,
            "bad_count": n_bad,
            "bad_fmt": n_bad_fmt,
            "bad_lang": n_bad_lang,
            "bad_both": n_bad_both,
            "bad_pct": round(n_bad / n_total * 100, 2),
            "bad_wer": f"{wer_bad:.2f}%",
            "bad_edge_wer": f"{edge_wer_bad:.2f}%",
            "all_wer": f"{wer_all:.2f}%",
            "all_edge_wer": f"{edge_wer_all:.2f}%",
        })

        total_bad += n_bad

        # Upload bad cases
        upload_ds = DS_NAME_MAP.get(ds, ds)
        upload_path = bf.join(UPLOAD_ROOT, upload_ds, "bad_fmt_lang.jsonl")
        print(f"  Uploading {n_bad} rows to {upload_path}")
        with bf.BlobFile(upload_path, "w") as out:
            for r in bad_rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY: {total_bad} total bad utterances across all datasets")
    print(f"{'='*100}")
    header = f"{'Dataset':<20} {'Total':>7} {'Bad':>6} {'Bad%':>7} {'Bad WER':>10} {'Bad EdgeWER':>12} {'All WER':>10} {'All EdgeWER':>12}"
    print(header)
    print("-" * len(header))
    for s in summary_rows:
        print(f"{s['dataset']:<20} {s['total']:>7} {s['bad_count']:>6} {s['bad_pct']:>6.1f}% {s['bad_wer']:>10} {s['bad_edge_wer']:>12} {s['all_wer']:>10} {s['all_edge_wer']:>12}")


if __name__ == "__main__":
    main()

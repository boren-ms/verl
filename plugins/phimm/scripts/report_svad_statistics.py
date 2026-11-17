"""Create reproducible distribution tables, workbook, PDF and figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

NAMES = ["earn_fy26_batch1", "earn_fy26_batch2", "quartr_1st23_long", "quartr_2nd48_long", "quartr_3rd38_long"]
LABELS = dict(zip(NAMES, ["FY26 B1", "FY26 B2", "Quartr 1st23", "Quartr 2nd48", "Quartr 3rd38"], strict=True))
COLORS = ["#2463a6", "#16857d", "#b87713", "#a54c81", "#7b61a8"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "statistics"
    output.mkdir(exist_ok=True)
    samples, segments = [], []
    for name in NAMES:
        rows = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        for index, row in enumerate(rows):
            lengths = [end - start for start, end in row["svad_bounds"]]
            samples.append(
                {
                    "batch": name,
                    "sample_index": index,
                    "id": row["id"],
                    "segment_count": len(row["audio_path"]),
                    "audio_seconds": row["svad_parent_duration"],
                    "retained_seconds": sum(lengths),
                    "word_count": len(row["text"].split()),
                    "svad_method": row["svad_method"],
                    "source": row["svad_source"],
                }
            )
            for seg_index, ((start, end), path) in enumerate(zip(row["svad_bounds"], row["audio_path"], strict=True)):
                segments.append(
                    {
                        "batch": name,
                        "sample_index": index,
                        "id": row["id"],
                        "segment_index": seg_index,
                        "start_seconds": start,
                        "end_seconds": end,
                        "audio_seconds": end - start,
                        "audio_path": path,
                    }
                )
    parents = pd.DataFrame(samples)
    children = pd.DataFrame(segments)
    assert len(parents) == 739 and len(children) == int(parents.segment_count.sum())
    assert children.audio_seconds.max() <= 30 + 1e-8
    groups = [*NAMES, "ALL"]
    summary = []
    for name in groups:
        p = parents if name == "ALL" else parents[parents.batch == name]
        s = children if name == "ALL" else children[children.batch == name]
        summary.append(
            {
                "batch": name,
                "sample_count": len(p),
                "segment_count": len(s),
                "segments_per_sample_mean": p.segment_count.mean(),
                "segments_per_sample_median": p.segment_count.median(),
                "audio_seconds_per_sample_mean": p.audio_seconds.mean(),
                "audio_seconds_per_sample_median": p.audio_seconds.median(),
                "audio_seconds_per_segment_mean": s.audio_seconds.mean(),
                "audio_seconds_per_segment_median": s.audio_seconds.median(),
                "words_per_sample_mean": p.word_count.mean(),
                "words_per_sample_median": p.word_count.median(),
                "source_audio_hours": p.audio_seconds.sum() / 3600,
                "retained_audio_hours": s.audio_seconds.sum() / 3600,
                "retained_audio_percent": s.audio_seconds.sum() / p.audio_seconds.sum() * 100,
                "total_words": int(p.word_count.sum()),
                "fallback_parents": int((p.svad_method == "fixed_fallback").sum()),
            }
        )
    summary = pd.DataFrame(summary)
    metrics = [
        (
            "segments_per_sample",
            parents,
            "segment_count",
            "Segments per sample",
            "count",
            [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, np.inf],
        ),
        (
            "audio_per_sample",
            parents,
            "audio_seconds",
            "Audio length per sample",
            "seconds",
            [0, 60, 120, 300, 600, 900, 1200, 1800, 3600, np.inf],
        ),
        (
            "audio_per_segment",
            children,
            "audio_seconds",
            "Audio length per segment",
            "seconds",
            [0, 0.5, 1, 2, 5, 10, 15, 20, 25, 30 + 1e-8, np.inf],
        ),
        (
            "words_per_sample",
            parents,
            "word_count",
            "Words per sample",
            "words",
            [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, np.inf],
        ),
    ]
    quantiles = []
    distributions = {}
    for key, frame, col, title, unit, bins in metrics:
        records = []
        for name in groups:
            values = frame[col].to_numpy() if name == "ALL" else frame.loc[frame.batch == name, col].to_numpy()
            q = np.percentile(values, [0, 5, 25, 50, 75, 95, 99, 100])
            quantiles.append(
                {
                    "metric": key,
                    "unit": unit,
                    "batch": name,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=0),
                    **dict(zip(["min", "p05", "p25", "median", "p75", "p95", "p99", "max"], q, strict=True)),
                }
            )
            counts, _ = np.histogram(values, bins)
            for left, right, count in zip(bins, bins[1:], counts, strict=False):
                label = f"[{left:g}, {right:g})"
                if key == "audio_per_segment" and 25 == left:
                    label = "[25, 30]"
                if key == "audio_per_segment" and left > 30:
                    label = ">30"
                records.append({"batch": name, "bin": label, "count": int(count), "percent": count / len(values) * 100})
        distributions[key] = pd.DataFrame(records)
    quantiles = pd.DataFrame(quantiles)
    frames = {
        "summary": summary,
        "quantiles": quantiles,
        "sample_metrics": parents,
        "segment_metrics": children,
        **{f"bins_{key}": frame for key, frame in distributions.items()},
    }
    definitions = pd.DataFrame(
        [
            ["sample", "One original long-form parent recording, preserved as one output row."],
            ["segment", "One ordered audio_path selector; no segment audio files are created."],
            ["segments/sample", "Segment count per parent. Combined mean = total segments / total parents."],
            ["audio/sample", "Full source recording duration; silence removed by VAD remains included here."],
            ["audio/segment", "Retained selector duration, end minus start; gaps excluded, overlaps removed."],
            [
                "word count/sample",
                "len(text.split()) after existing Base3a clean_tagged_text; words, not model tokens.",
            ],
            ["combined statistics", "Weighted over individual parents or segments, not an average of batch means."],
            ["quantiles", "numpy linear percentiles; std uses population denominator (ddof=0)."],
            ["retained audio", "Sum of nonoverlapping segment durations; VAD may omit silence gaps."],
            ["fallback", "Parents without retained SVAD spans use explicitly marked fixed windows."],
            ["boxplots", "Median/IQR boxes and 1.5-IQR whiskers; outliers hidden. Histograms/ECDFs include all rows."],
        ],
        columns=["term", "definition"],
    )
    frames["definitions"] = definitions
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.9f")
    workbook = output / "earnings_long_svad30_statistics.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            sheet.freeze_panes = "B2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="183650")
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            sheet.row_dimensions[1].height = 34
            for j, col in enumerate(frame.columns, 1):
                sheet.column_dimensions[get_column_letter(j)].width = min(45, max(15, len(col) + 2))
                if pd.api.types.is_float_dtype(frame[col]):
                    for row in sheet.iter_rows(min_row=2, min_col=j, max_col=j):
                        row[0].number_format = "0.00"
            if name == "definitions":
                sheet.column_dimensions["B"].width = 110
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    pdf = PdfPages(output / "earnings_long_svad30_statistics.pdf")
    figure_paths = []

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figure_paths.append(name + ".png")
        plt.close(fig)

    def title(fig, text):
        fig.suptitle(text, fontsize=18, fontweight="bold", y=1.02)
        fig.text(
            0.01,
            -0.025,
            "Earnings/Quartr long-form | SVAD max 30 s | full parent reference | metadata only",
            fontsize=9,
            color="#526578",
        )

    def boxes(ax, frame, col, label, scale=1):
        arrays = [frame.loc[frame.batch == n, col].to_numpy() / scale for n in NAMES]
        box = ax.boxplot(
            arrays,
            tick_labels=[LABELS[n] for n in NAMES],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#162331", "linewidth": 1.5},
        )
        for patch, color in zip(box["boxes"], COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    batch_summary = summary[summary.batch != "ALL"]
    for ax, col, heading in [
        (axes[0, 0], "sample_count", "1. Sample count"),
        (axes[0, 1], "segment_count", "2. Segment count"),
    ]:
        vals = batch_summary[col].to_numpy()
        ax.bar([LABELS[n] for n in NAMES], vals, color=COLORS)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, vals.max() * 1.16)
        ax.set_title(heading)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("Count")
    boxes(axes[0, 2], parents, "segment_count", "Segments / sample")
    axes[0, 2].set_title("3. Segment count per sample")
    boxes(axes[1, 0], parents, "audio_seconds", "Minutes / sample", 60)
    axes[1, 0].set_title("4. Audio length per sample")
    boxes(axes[1, 1], children, "audio_seconds", "Seconds / segment")
    axes[1, 1].set_title("5. Audio length per segment")
    boxes(axes[1, 2], parents, "word_count", "Words / sample")
    axes[1, 2].set_title("6. Word count per sample")
    total = summary.iloc[-1]
    title(fig, f"SVAD long-form dataset: {len(parents):,} samples / {len(children):,} segments")
    fig.text(
        0.01,
        -0.05,
        "Boxplots: median/IQR, 1.5-IQR whiskers; outliers hidden. Histograms and ECDFs include every observation.",
        fontsize=9,
        color="#526578",
    )
    save(fig, "01_overview_six_metrics")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, (key, frame, col, heading, unit, bins) in zip(axes.flat, metrics, strict=True):
        values = frame[col].to_numpy()
        ax.hist(values, bins="fd", color="#2463a6", edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(values), color="#b87713", ls="--", label=f"Median {np.median(values):.2f}")
        ax.axvline(values.mean(), color="#993c65", ls=":", label=f"Mean {values.mean():.2f}")
        ax.set_title(f"{heading} (n={len(values):,})")
        ax.set_xlabel(unit)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    title(fig, "Combined distributions across all five batches")
    save(fig, "02_combined_distributions")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, (key, frame, col, heading, unit, bins) in zip(axes.flat, metrics, strict=True):
        for name, color in zip(NAMES, COLORS, strict=True):
            values = np.sort(frame.loc[frame.batch == name, col].to_numpy())
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, label=LABELS[name])
        ax.set_title(heading)
        ax.set_xlabel(unit)
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    title(fig, "Per-batch distribution comparison (ECDF)")
    save(fig, "03_batch_ecdf_distributions")

    def table_figure(frame, name, heading, width=17, fontsize=10):
        fig, ax = plt.subplots(figsize=(width, max(3.0, 0.48 * len(frame) + 1.5)))
        ax.axis("off")
        t = ax.table(cellText=frame.astype(str).values, colLabels=frame.columns, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(fontsize)
        t.scale(1, 1.8)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#e2e8ef")
            if r == 0:
                cell.set_facecolor("#183650")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f0f5fa")
        ax.set_title(heading, fontsize=17, fontweight="bold", pad=18)
        fig.tight_layout()
        save(fig, name)

    visible = pd.DataFrame(
        {
            "Batch": summary.batch.map(lambda x: "ALL" if x == "ALL" else LABELS[x]),
            "Samples": summary.sample_count.map(lambda x: f"{x:,}"),
            "Segments": summary.segment_count.map(lambda x: f"{x:,}"),
            "Segments/sample\nmean": summary.segments_per_sample_mean.map(lambda x: f"{x:.2f}"),
            "Audio/sample\nmean minutes": (summary.audio_seconds_per_sample_mean / 60).map(lambda x: f"{x:.2f}"),
            "Audio/segment\nmean seconds": summary.audio_seconds_per_segment_mean.map(lambda x: f"{x:.2f}"),
            "Words/sample\nmean": summary.words_per_sample_mean.map(lambda x: f"{x:,.2f}"),
        }
    )
    table_figure(visible, "04_summary_table", "All six requested measures", fontsize=11)
    for key, frame, col, heading, unit, bins in metrics:
        q = quantiles[quantiles.metric == key].copy()
        show = q[["batch", "min", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean"]].copy()
        show["batch"] = show.batch.map(lambda x: "ALL" if x == "ALL" else LABELS[x])
        for c in show.columns[1:]:
            show[c] = show[c].map(lambda v: f"{v:,.2f}")
        table_figure(show, "quantiles_" + key, f"{heading}: distribution statistics ({unit})", fontsize=10)
        distribution = distributions[key]
        display = pd.DataFrame({"Bin": distribution[distribution.batch == "ALL"]["bin"].to_list()})
        for group in groups:
            f = distribution[distribution.batch == group]
            display["ALL" if group == "ALL" else LABELS[group]] = [
                f"{int(r['count']):,} ({r['percent']:.1f}%)" for _, r in f.iterrows()
            ]
        table_figure(display, "bins_" + key, f"{heading}: binned counts and within-batch percentages", fontsize=9)
    pdf.close()
    html = [
        '<!doctype html><meta charset="utf-8"><title>Earnings long-form SVAD statistics</title>',
        "<style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#183650}"
        "img{width:100%;margin:16px 0 36px}table{border-collapse:collapse}"
        "td,th{padding:9px;border:1px solid #ccd7e2}</style>",
        "<h1>Earnings/Quartr SVAD long-form statistics</h1>",
        "<p>One sample = one long parent. Segment lists retain the original audio and full reference. "
        "Word counts use the normalized full transcript.</p>",
        definitions.to_html(index=False, escape=True),
        '<p><a href="earnings_long_svad30_statistics.xlsx">Excel workbook</a> · '
        '<a href="earnings_long_svad30_statistics.pdf">PDF figures and tables</a></p>',
    ]
    for name in figure_paths:
        html.append(f'<img src="{name}" alt="{name}">')
    (output / "index.html").write_text("\n".join(html))
    summary.to_json(output / "summary.json", orient="records", indent=2)
    lines = [
        "# Earnings/Quartr SVAD long-form statistics",
        "",
        f"- Samples: {len(parents):,}",
        f"- Segments: {len(children):,}",
        f"- Segments per sample (mean): {total.segments_per_sample_mean:.4f}",
        f"- Original audio: {total.source_audio_hours:.6f} h; retained: {total.retained_audio_hours:.6f} h",
        f"- Mean audio per sample: {total.audio_seconds_per_sample_mean:.4f} s",
        f"- Mean audio per segment: {total.audio_seconds_per_segment_mean:.4f} s",
        f"- Mean words per sample: {total.words_per_sample_mean:.4f}",
        "",
        "Definitions are recorded in definitions.csv and the workbook. "
        "All CSV metrics are unrounded except serialization precision.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("Report:", output, flush=True)


if __name__ == "__main__":
    main()

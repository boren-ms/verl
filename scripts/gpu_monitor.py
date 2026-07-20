#!/usr/bin/env python3
"""Monitor per-GPU utilization with rolling-window averages.

Samples GPU utilization (and memory) via ``nvidia-smi`` at a fixed interval and
reports, for every GPU, the instantaneous value plus the average over several
time windows (1 min, 5 min, 30 min and the full session). A final row shows the
average across all GPUs for each window.

Usage:
    python scripts/gpu_monitor.py                 # live view, 1s sampling
    python scripts/gpu_monitor.py -i 2            # sample every 2 seconds
    python scripts/gpu_monitor.py --metric memory # track memory-util instead
    python scripts/gpu_monitor.py --once          # print one snapshot and exit
    python scripts/gpu_monitor.py --csv log.csv   # also append samples to CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field

# Window definitions: (label, seconds). ``None`` means "full session".
WINDOWS: list[tuple[str, float | None]] = [
    ("1min", 60.0),
    ("5min", 300.0),
    ("30min", 1800.0),
    ("full", None),
]


@dataclass
class GpuSeries:
    """Rolling history of (timestamp, value) samples for a single GPU."""

    index: int
    name: str
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    full_sum: float = 0.0
    full_count: int = 0

    def add(self, ts: float, value: float, max_age: float) -> None:
        self.samples.append((ts, value))
        self.full_sum += value
        self.full_count += 1
        # Drop samples older than the largest finite window to bound memory.
        cutoff = ts - max_age
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def window_average(self, now: float, seconds: float | None) -> float | None:
        if seconds is None:
            if self.full_count == 0:
                return None
            return self.full_sum / self.full_count
        cutoff = now - seconds
        total = 0.0
        count = 0
        # Iterate from the newest sample backwards; stop once out of window.
        for ts, value in reversed(self.samples):
            if ts < cutoff:
                break
            total += value
            count += 1
        if count == 0:
            return None
        return total / count

    @property
    def current(self) -> float | None:
        return self.samples[-1][1] if self.samples else None


def query_gpus(metric: str) -> list[tuple[int, str, float]]:
    """Return a list of (index, name, value) tuples from nvidia-smi."""
    query_field = "utilization.gpu" if metric == "gpu" else "utilization.memory"
    cmd = [
        "nvidia-smi",
        f"--query-gpu=index,name,{query_field}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows: list[tuple[int, str, float]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx = int(parts[0])
        name = parts[1]
        try:
            value = float(parts[2])
        except ValueError:
            value = 0.0
        rows.append((idx, name, value))
    return rows


def fmt(value: float | None) -> str:
    return f"{value:5.1f}" if value is not None else "  -  "


def render(series: dict[int, GpuSeries], now: float, metric: str) -> str:
    label = "GPU util %" if metric == "gpu" else "Mem util %"
    headers = ["GPU", "now"] + [w[0] for w in WINDOWS]
    col_w = 7
    lines: list[str] = []
    title = (
        f"GPU utilization monitor ({label}) on {socket.gethostname()}  "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(title)
    lines.append("-" * (col_w * len(headers)))
    lines.append("".join(h.rjust(col_w) for h in headers))
    lines.append("-" * (col_w * len(headers)))

    # Per-window running totals to compute the "all GPUs" average row.
    window_totals: list[tuple[float, int]] = [(0.0, 0) for _ in WINDOWS]
    cur_total = 0.0
    cur_count = 0

    for idx in sorted(series):
        s = series[idx]
        row_vals = [fmt(s.current)]
        if s.current is not None:
            cur_total += s.current
            cur_count += 1
        for wi, (_, secs) in enumerate(WINDOWS):
            avg = s.window_average(now, secs)
            row_vals.append(fmt(avg))
            if avg is not None:
                t, c = window_totals[wi]
                window_totals[wi] = (t + avg, c + 1)
        cells = [str(idx).rjust(col_w)] + [v.rjust(col_w) for v in row_vals]
        lines.append("".join(cells))

    lines.append("-" * (col_w * len(headers)))
    all_now = cur_total / cur_count if cur_count else None
    all_cells = ["ALL".rjust(col_w), fmt(all_now).rjust(col_w)]
    for t, c in window_totals:
        all_cells.append(fmt(t / c if c else None).rjust(col_w))
    lines.append("".join(all_cells))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="sampling interval in seconds (default: 1.0)")
    parser.add_argument(
        "--metric",
        choices=["gpu", "memory"],
        default="gpu",
        help="which utilization to track: 'gpu' compute or 'memory' (default: gpu)",
    )
    parser.add_argument("--once", action="store_true", help="print a single snapshot and exit")
    parser.add_argument("--csv", metavar="PATH", help="append raw per-GPU samples to this CSV file")
    args = parser.parse_args()

    if shutil.which("nvidia-smi") is None:
        print("error: nvidia-smi not found in PATH", file=sys.stderr)
        return 1

    metric = args.metric
    max_finite = max((s for _, s in WINDOWS if s is not None), default=0.0)

    series: dict[int, GpuSeries] = {}
    csv_writer = None
    csv_file = None
    if args.csv:
        new_file = not os.path.exists(args.csv)
        csv_file = open(args.csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(["timestamp", "gpu_index", "name", f"{metric}_util"])

    try:
        while True:
            now = time.time()
            try:
                rows = query_gpus(metric)
            except subprocess.CalledProcessError as exc:
                print(f"error: nvidia-smi failed: {exc}", file=sys.stderr)
                return 1

            for idx, name, value in rows:
                if idx not in series:
                    series[idx] = GpuSeries(index=idx, name=name)
                series[idx].add(now, value, max_finite)
                if csv_writer is not None:
                    csv_writer.writerow([f"{now:.3f}", idx, name, value])
            if csv_file is not None:
                csv_file.flush()

            output = render(series, now, metric)
            if args.once:
                print(output)
                break

            # Clear screen and redraw for a live view.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(output + "\n")
            sys.stdout.flush()

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    finally:
        if csv_file is not None:
            csv_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

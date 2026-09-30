#!/usr/bin/env python3
"""Find significant reward drops in verl training logs and rank likely drivers."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"(?:^| - )(?P<key>[A-Za-z0-9_.@/-]+):(?P<value>"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
    r"(?= - |$)",
    re.IGNORECASE,
)
PRIMARY_METRICS = (
    "critic/rewards/mean",
    "reward-core/score/mean",
    "critic/score/mean",
)


@dataclass(frozen=True)
class StepRecord:
    step: int
    line: int
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect when verl training rewards drop and rank evidence for why."
    )
    parser.add_argument("log", type=Path, help="verl training log")
    parser.add_argument(
        "--metric",
        help="Primary reward metric. Default: auto-detect critic/rewards/mean, reward-core/score/mean, or critic/score/mean.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of earlier steps used for the rolling-median baseline (default: 5).",
    )
    parser.add_argument(
        "--drop-abs",
        type=float,
        default=0.05,
        help="Minimum absolute decrease that triggers a drop (default: 0.05).",
    )
    parser.add_argument(
        "--drop-pct",
        type=float,
        default=5.0,
        help="Minimum relative decrease percentage that triggers a drop (default: 5).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of largest drops to report (default: 10).",
    )
    parser.add_argument("--json", type=Path, help="Optional path for a machine-readable report.")
    return parser.parse_args()


def finite(value: float) -> bool:
    return math.isfinite(value)


def parse_metric_line(line: str) -> dict[str, float]:
    clean = ANSI_RE.sub("", line).strip()
    metrics: dict[str, float] = {}
    for match in METRIC_RE.finditer(clean):
        value = float(match.group("value"))
        metrics[match.group("key")] = value
    return metrics


def load_records(path: Path) -> list[StepRecord]:
    by_step: dict[int, StepRecord] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if "step:" not in line:
                continue
            metrics = parse_metric_line(line)
            if not any(metric in metrics for metric in PRIMARY_METRICS):
                continue
            step_value = metrics.get("step")
            if step_value is None or not finite(step_value) or not step_value.is_integer():
                continue
            step = int(step_value)
            existing = by_step.get(step)
            if existing is None or len(metrics) > len(existing.metrics):
                by_step[step] = StepRecord(step=step, line=line_number, metrics=metrics)
    return sorted(by_step.values(), key=lambda record: record.step)


def select_metric(records: Iterable[StepRecord], requested: str | None) -> str:
    records = list(records)
    if requested:
        if not any(requested in record.metrics for record in records):
            raise ValueError(f"metric {requested!r} was not found in any training-step record")
        return requested
    for metric in PRIMARY_METRICS:
        if any(metric in record.metrics for record in records):
            return metric
    raise ValueError("no supported primary reward metric was found")


def median_metric(records: list[StepRecord], metric: str) -> float | None:
    values = [
        record.metrics[metric]
        for record in records
        if metric in record.metrics and finite(record.metrics[metric])
    ]
    return statistics.median(values) if values else None


def component_deltas(
    record: StepRecord, baseline_records: list[StepRecord]
) -> list[dict[str, float | str]]:
    components: list[dict[str, float | str]] = []
    for key, current in record.metrics.items():
        if not key.startswith("reward-core/") or not key.endswith("/mean") or key == "reward-core/score/mean":
            continue
        baseline = median_metric(baseline_records, key)
        if baseline is None or not finite(current):
            continue
        components.append(
            {
                "metric": key,
                "baseline": baseline,
                "current": current,
                "delta": current - baseline,
            }
        )
    return sorted(components, key=lambda item: float(item["delta"]))


def diagnostic_evidence(
    record: StepRecord, baseline_records: list[StepRecord]
) -> list[dict[str, float | str]]:
    rules = (
        ("actor/kl_loss", "KL divergence increased", "increase"),
        ("actor/grad_norm", "gradient norm increased", "increase"),
        ("actor/pg_clipfrac", "policy clipping increased", "increase"),
        ("response_length/clip_ratio", "response clipping increased", "increase"),
        ("response/aborted_ratio", "aborted responses increased", "increase"),
        ("train/num_gen_batches", "more generation batches were needed", "increase"),
        ("train/gen_kept_frac", "generation keep fraction fell", "decrease"),
    )
    evidence: list[dict[str, float | str]] = []
    for key, label, direction in rules:
        current = record.metrics.get(key)
        baseline = median_metric(baseline_records, key)
        if current is None or baseline is None or not finite(current):
            continue
        delta = current - baseline
        relevant = delta > 0 if direction == "increase" else delta < 0
        if not relevant:
            continue
        evidence.append(
            {
                "metric": key,
                "label": label,
                "baseline": baseline,
                "current": current,
                "delta": delta,
            }
        )
    return sorted(evidence, key=lambda item: abs(float(item["delta"])), reverse=True)


def persistence(records: list[StepRecord], index: int, metric: str, baseline: float) -> str:
    following = [
        record.metrics[metric]
        for record in records[index + 1 : index + 4]
        if metric in record.metrics and finite(record.metrics[metric])
    ]
    if not following:
        return "unknown (no later reward point)"
    recovered = sum(value >= baseline for value in following)
    if recovered:
        return f"transient ({recovered}/{len(following)} next points recovered to baseline)"
    return f"sustained across {len(following)} following points"


def analyze(
    records: list[StepRecord],
    metric: str,
    window: int,
    drop_abs: float,
    drop_pct: float,
    top: int,
) -> list[dict[str, object]]:
    drops: list[dict[str, object]] = []
    for index, record in enumerate(records):
        earlier = [
            candidate
            for candidate in records[max(0, index - window) : index]
            if metric in candidate.metrics and finite(candidate.metrics[metric])
        ]
        current = record.metrics.get(metric)
        baseline = median_metric(earlier, metric)
        if current is None or baseline is None or not finite(current):
            continue
        delta = current - baseline
        relative_pct = (delta / abs(baseline) * 100.0) if baseline else None
        absolute_trigger = -delta >= drop_abs
        relative_trigger = relative_pct is not None and relative_pct <= -drop_pct
        if not (absolute_trigger or relative_trigger):
            continue
        components = component_deltas(record, earlier)
        diagnostics = diagnostic_evidence(record, earlier)
        drops.append(
            {
                "step": record.step,
                "line": record.line,
                "metric": metric,
                "baseline": baseline,
                "current": current,
                "delta": delta,
                "relative_pct": relative_pct,
                "persistence": persistence(records, index, metric, baseline),
                "component_drivers": components[:5],
                "diagnostic_evidence": diagnostics[:5],
            }
        )
    drops.sort(key=lambda drop: float(drop["delta"]))
    return drops[:top]


def format_number(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.6g}"
    return str(value)


def render_markdown(path: Path, records: list[StepRecord], metric: str, drops: list[dict[str, object]]) -> str:
    lines = [
        "# Reward-drop analysis",
        "",
        f"- Log: `{path}`",
        f"- Parsed training steps: {len(records)}",
        f"- Primary metric: `{metric}`",
        f"- Significant drops: {len(drops)}",
        "",
    ]
    if not drops:
        lines.append("No drop crossed the configured absolute or relative thresholds.")
        return "\n".join(lines)

    lines.extend(
        [
            "| Rank | Step | Line | Baseline | Current | Delta | Relative | Behavior |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, drop in enumerate(drops, 1):
        relative = drop["relative_pct"]
        relative_text = "n/a" if relative is None else f"{float(relative):.2f}%"
        lines.append(
            f"| {rank} | {drop['step']} | {drop['line']} | "
            f"{format_number(drop['baseline'])} | {format_number(drop['current'])} | "
            f"{format_number(drop['delta'])} | {relative_text} | {drop['persistence']} |"
        )

    for rank, drop in enumerate(drops, 1):
        lines.extend(["", f"## {rank}. Step {drop['step']} (line {drop['line']})", ""])
        components = drop["component_drivers"]
        diagnostics = drop["diagnostic_evidence"]
        if components:
            lines.append("Likely reward-component drivers (largest decreases versus the rolling baseline):")
            for item in components:
                lines.append(
                    f"- `{item['metric']}`: {format_number(item['baseline'])} -> "
                    f"{format_number(item['current'])} (delta {format_number(item['delta'])})"
                )
        else:
            lines.append("No component reward decrease was available to explain this drop.")
        if diagnostics:
            lines.append("")
            lines.append("Correlated diagnostics (evidence, not proof of causation):")
            for item in diagnostics:
                lines.append(
                    f"- {item['label']} via `{item['metric']}`: "
                    f"{format_number(item['baseline'])} -> {format_number(item['current'])}"
                )
        else:
            lines.extend(["", "No configured optimization, length, or filtering diagnostic moved adversely."])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.window < 1:
        raise SystemExit("--window must be at least 1")
    if args.drop_abs < 0 or args.drop_pct < 0:
        raise SystemExit("--drop-abs and --drop-pct must be nonnegative")
    if not args.log.is_file():
        raise SystemExit(f"log does not exist: {args.log}")

    records = load_records(args.log)
    if not records:
        raise SystemExit("no verl training-step metric records were found")
    try:
        metric = select_metric(records, args.metric)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    drops = analyze(records, metric, args.window, args.drop_abs, args.drop_pct, args.top)
    print(render_markdown(args.log, records, metric, drops))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "log": str(args.log),
            "parsed_steps": len(records),
            "metric": metric,
            "window": args.window,
            "drop_abs": args.drop_abs,
            "drop_pct": args.drop_pct,
            "drops": drops,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

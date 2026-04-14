#!/usr/bin/env python3
"""
Diagnose idle ATS (bus-ats-prod) nodes: find root cause of 0% GPU utilization.

Usage:
    python3 diagnose_ats.py [--pattern GLOB] [--output-dir DIR] [--timeout SECS]

The script:
  1. Discovers Ready bus-ats-prod pools via rcall-brix ls.
  2. Checks GPU utilization on each node.
  3. For idle nodes, collects evidence from 7 sources (bus.log, processes,
     tmux, dmesg, disk, nvidia-smi pmon, systemd journal).
  4. Classifies the root cause and prints an actionable report.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent

NVIDIA_SMI_QUERY = (
    "index,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,temperature.gpu"
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Root-cause categories
ZOMBIE = "ZOMBIE"
CRASHED = "CRASHED"
OOM_KILLED = "OOM_KILLED"
DISK_FULL = "DISK_FULL"
NO_JOBS = "NO_JOBS"
STALE_SESSION = "STALE_SESSION"
PROVISIONING = "PROVISIONING"
UNKNOWN = "UNKNOWN"

CAUSE_ICONS = {
    ZOMBIE: "🔴",
    CRASHED: "🔴",
    OOM_KILLED: "🟠",
    DISK_FULL: "🟠",
    NO_JOBS: "🟡",
    STALE_SESSION: "🟡",
    PROVISIONING: "⏳",
    UNKNOWN: "❓",
}

CAUSE_ACTIONS = {
    ZOMBIE: "Kill orphan process or restart pool: rcall-brix restart <POOL>",
    CRASHED: "Inspect bus.log errors, fix config, resubmit the job",
    OOM_KILLED: "Reduce batch size / model size or request a larger node",
    DISK_FULL: "Clean up /tmp, stale checkpoints, or old logs",
    NO_JOBS: "Submit new jobs or check the scheduler queue",
    STALE_SESSION: "Clean up stale tmux sessions, restart pool if needed",
    PROVISIONING: "Wait for pod to finish provisioning",
    UNKNOWN: "SSH into node for manual investigation: rcall-brix tmux -s codex <NODE>",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose idle ATS nodes")
    p.add_argument(
        "--pattern", default="bus-ats-prod*",
        help="Glob pattern to filter pool names (default: bus-ats-prod*)",
    )
    p.add_argument("--output-dir", default="", help="Write JSON report to this directory")
    p.add_argument("--timeout", type=int, default=30, help="SSH timeout per node in seconds")
    p.add_argument("--workers", type=int, default=4, help="Parallel SSH workers")
    p.add_argument(
        "--all", action="store_true",
        help="Diagnose all Ready nodes, not just idle ones",
    )
    p.add_argument(
        "--restart-scheduled", action="store_true",
        help="Restart Scheduled pools matching the pattern",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pool discovery
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def list_pools_by_status(pattern: str, status_filter: str) -> list[dict]:
    """Return pools matching pattern with given status."""
    result = run_cmd(["rcall-brix", "ls"])
    pools = []
    for line in result.stdout.splitlines():
        clean = ANSI_RE.sub("", line)
        if status_filter not in clean:
            continue
        parts = clean.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        cluster = parts[1]
        size_match = re.search(r"\d+ x \d+ GPU", clean)
        size = size_match.group() if size_match else ""
        if pattern and not fnmatch.fnmatch(name, pattern):
            continue
        pools.append({"name": name, "cluster": cluster, "size": size})
    return pools


def restart_scheduled_pools(pools: list[dict]) -> list[dict]:
    results = []
    for pool in pools:
        name = pool["name"]
        try:
            r = run_cmd(["rcall-brix", "restart", name])
            results.append({"pool": name, "success": r.returncode == 0, "output": r.stdout.strip()})
        except Exception as e:
            results.append({"pool": name, "success": False, "output": str(e)})
    return results


# ---------------------------------------------------------------------------
# Evidence collection (per-node, via SSH)
# ---------------------------------------------------------------------------

def ssh_cmd(pool: str, remote_cmd: str, timeout: int) -> str:
    """Run a single command on a remote node. Returns stdout or error string."""
    try:
        r = run_cmd(["rcall-brix", "ssh", pool, "--", remote_cmd], timeout=timeout)
        # Filter out rcall-brix preamble lines (RUN:, timestamps)
        lines = []
        for line in r.stdout.splitlines():
            if line.startswith("RUN:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
    except subprocess.TimeoutExpired:
        return "ERROR:TIMEOUT"
    except Exception as e:
        return f"ERROR:{e}"


def collect_gpu_state(pool: str, timeout: int) -> dict:
    """nvidia-smi query for utilization."""
    raw = ssh_cmd(
        pool,
        f"nvidia-smi --query-gpu={NVIDIA_SMI_QUERY} --format=csv,noheader,nounits",
        timeout,
    )
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_gpu": int(parts[2]),
                    "util_mem": int(parts[3]),
                    "mem_used": int(parts[4]),
                    "mem_total": int(parts[5]),
                    "temp": int(parts[6]),
                })
            except (ValueError, IndexError):
                continue
    return {"gpus": gpus, "raw": raw}


def collect_gpu_processes(pool: str, timeout: int) -> str:
    """nvidia-smi pmon snapshot — shows which PIDs are using each GPU."""
    return ssh_cmd(pool, "nvidia-smi pmon -c 1 -s um 2>/dev/null | head -20", timeout)


def collect_bus_log(pool: str, timeout: int) -> str:
    """Last 50 lines of ~/bus.log."""
    return ssh_cmd(pool, "tail -50 ~/bus.log 2>/dev/null || echo NO_BUS_LOG_FOUND", timeout)


def collect_processes(pool: str, timeout: int) -> str:
    """Check for job-related processes."""
    cmds = [
        "pgrep -fa 'harmony_scripts.engines.start_engine' 2>/dev/null || true",
        "pgrep -fa 'run_asr_sft' 2>/dev/null || true",
        "pgrep -fa 'eval_audio' 2>/dev/null || true",
        "pgrep -fa 'python.*train' 2>/dev/null || true",
        "pgrep -fa 'python.*eval' 2>/dev/null || true",
    ]
    return ssh_cmd(pool, " && ".join(cmds), timeout)


def collect_tmux_sessions(pool: str, timeout: int) -> str:
    """List tmux sessions."""
    return ssh_cmd(pool, "tmux ls 2>/dev/null || echo NO_TMUX_SESSIONS", timeout)


def collect_dmesg(pool: str, timeout: int) -> str:
    """Recent kernel messages — OOM kills, GPU errors."""
    return ssh_cmd(
        pool,
        "dmesg --time-format iso 2>/dev/null | tail -30 || dmesg 2>/dev/null | tail -30 || echo NO_DMESG",
        timeout,
    )


def collect_disk_usage(pool: str, timeout: int) -> str:
    """Disk usage for key mount points."""
    return ssh_cmd(pool, "df -h / /tmp 2>/dev/null | tail -5", timeout)


def collect_journal(pool: str, timeout: int) -> str:
    """Recent systemd journal entries for container/kubelet issues."""
    return ssh_cmd(
        pool,
        "journalctl --no-pager -n 20 -p err 2>/dev/null | tail -20 || echo NO_JOURNAL",
        timeout,
    )


def collect_all_evidence(pool: str, timeout: int) -> dict:
    """Collect all diagnostic evidence from a single node."""
    evidence = {
        "pool": pool,
        "gpu_state": collect_gpu_state(pool, timeout),
        "gpu_processes": collect_gpu_processes(pool, timeout),
        "bus_log": collect_bus_log(pool, timeout),
        "processes": collect_processes(pool, timeout),
        "tmux": collect_tmux_sessions(pool, timeout),
        "dmesg": collect_dmesg(pool, timeout),
        "disk": collect_disk_usage(pool, timeout),
        "journal": collect_journal(pool, timeout),
    }
    return evidence


# ---------------------------------------------------------------------------
# Root-cause analysis
# ---------------------------------------------------------------------------

def analyze_root_cause(evidence: dict) -> dict:
    """Analyze evidence and return {cause, details, confidence}."""
    gpus = evidence["gpu_state"]["gpus"]
    bus_log = evidence["bus_log"]
    processes = evidence["processes"]
    tmux = evidence["tmux"]
    dmesg = evidence["dmesg"]
    disk = evidence["disk"]
    gpu_procs = evidence["gpu_processes"]

    # Calculate GPU stats
    total_mem_used = sum(g["mem_used"] for g in gpus) if gpus else 0
    avg_mem = total_mem_used / len(gpus) if gpus else 0
    all_idle = all(g["util_gpu"] == 0 for g in gpus) if gpus else True
    details = []

    # --- Check 1: OOM kills in dmesg ---
    if "ERROR:" not in dmesg:
        oom_lines = [l for l in dmesg.splitlines() if "Out of memory" in l or "oom-kill" in l.lower() or "OOM" in l]
        if oom_lines:
            details.append(f"OOM kill detected: {oom_lines[-1].strip()[:120]}")
            return {"cause": OOM_KILLED, "details": details, "confidence": "high"}

    # --- Check 2: Disk full ---
    if "ERROR:" not in disk:
        for line in disk.splitlines():
            parts = line.split()
            if len(parts) >= 5:
                use_pct = parts[4].rstrip("%")
                try:
                    if int(use_pct) >= 95:
                        details.append(f"Disk nearly full: {line.strip()}")
                        return {"cause": DISK_FULL, "details": details, "confidence": "high"}
                except ValueError:
                    pass

    # --- Check 3: Zombie — GPU memory held but 0% compute ---
    if all_idle and avg_mem > 5000:  # >5 GB average memory with 0% compute
        details.append(f"GPUs idle but avg memory = {avg_mem:.0f} MB (likely orphan process)")
        # Check if fake_eval is holding memory
        if "[fake_eval]" in bus_log:
            details.append("fake_eval pattern found in bus.log")
        if avg_mem > 50000:
            details.append("Very high residual memory (>50 GB avg) — strong zombie signal")
            return {"cause": ZOMBIE, "details": details, "confidence": "high"}
        return {"cause": ZOMBIE, "details": details, "confidence": "medium"}

    # --- Check 4: Pod provisioning ---
    if "Waiting until pod is ready" in bus_log:
        details.append("bus.log shows pod is still provisioning")
        return {"cause": PROVISIONING, "details": details, "confidence": "high"}

    # --- Check 5: Crashed job ---
    if "ERROR:" not in bus_log and bus_log != "NO_BUS_LOG_FOUND":
        error_patterns = [
            "Traceback", "RuntimeError", "CUDA error", "NCCL error",
            "Exception", "OOM", "killed", "Segmentation fault",
            "ConnectionError", "TimeoutError", "FileNotFoundError",
        ]
        crash_lines = []
        for line in bus_log.splitlines():
            for pat in error_patterns:
                if pat.lower() in line.lower():
                    crash_lines.append(line.strip()[:120])
                    break
        if crash_lines:
            details.append("Error patterns found in bus.log:")
            details.extend(crash_lines[-5:])  # last 5 error lines
            return {"cause": CRASHED, "details": details, "confidence": "medium"}

    # --- Check 6: Stale tmux session but no job process ---
    has_tmux = "NO_TMUX_SESSIONS" not in tmux and "ERROR:" not in tmux and tmux.strip()
    has_job_process = False
    if "ERROR:" not in processes:
        job_lines = [l for l in processes.splitlines() if l.strip()]
        has_job_process = len(job_lines) > 0

    if has_tmux and not has_job_process:
        details.append(f"tmux sessions exist but no job processes found")
        details.append(f"tmux: {tmux.strip()[:200]}")
        return {"cause": STALE_SESSION, "details": details, "confidence": "medium"}

    # --- Check 7: No jobs at all ---
    if not has_job_process and all_idle:
        details.append("No job processes found and all GPUs idle")
        if bus_log == "NO_BUS_LOG_FOUND":
            details.append("No bus.log present — node may have never received a job")
        return {"cause": NO_JOBS, "details": details, "confidence": "high"}

    # --- Fallback ---
    details.append("Could not determine a clear root cause from collected evidence")
    if has_job_process:
        job_lines = [l for l in processes.splitlines() if l.strip()]
        details.append(f"Job processes found: {'; '.join(l[:80] for l in job_lines[:3])}")
    return {"cause": UNKNOWN, "details": details, "confidence": "low"}


# ---------------------------------------------------------------------------
# Parallel collection
# ---------------------------------------------------------------------------

def collect_diagnoses(
    pools: list[dict], timeout: int, workers: int, diagnose_all: bool
) -> list[dict]:
    """Collect GPU state and diagnose idle nodes."""
    # Phase 1: GPU state for all pools (parallel)
    print(f"Phase 1: Checking GPU utilization on {len(pools)} node(s)...")
    gpu_results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(collect_gpu_state, p["name"], timeout): p["name"]
            for p in pools
        }
        for fut in as_completed(futures):
            name = futures[fut]
            gpu_results[name] = fut.result()

    # Identify idle nodes
    idle_pools = []
    busy_pools = []
    unreachable = []
    for p in pools:
        name = p["name"]
        gpus = gpu_results[name]["gpus"]
        if not gpus:
            unreachable.append(p)
            continue
        all_idle = all(g["util_gpu"] == 0 for g in gpus)
        half_idle = sum(1 for g in gpus if g["util_gpu"] == 0) >= len(gpus) // 2
        if all_idle or half_idle:
            idle_pools.append(p)
        else:
            busy_pools.append(p)

    targets = pools if diagnose_all else idle_pools
    if not targets:
        return _build_summary(pools, gpu_results, {}, busy_pools, unreachable)

    # Phase 2: Full evidence collection on targets (parallel)
    print(f"Phase 2: Collecting evidence from {len(targets)} node(s)...")
    evidence_map = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(collect_all_evidence, p["name"], timeout): p["name"]
            for p in targets
        }
        for fut in as_completed(futures):
            name = futures[fut]
            evidence_map[name] = fut.result()

    return _build_summary(pools, gpu_results, evidence_map, busy_pools, unreachable)


def _build_summary(
    pools: list[dict],
    gpu_results: dict,
    evidence_map: dict,
    busy_pools: list[dict],
    unreachable: list[dict],
) -> list[dict]:
    """Build structured diagnosis results."""
    results = []
    for p in pools:
        name = p["name"]
        gpus = gpu_results.get(name, {}).get("gpus", [])
        active = sum(1 for g in gpus if g["util_gpu"] > 0)
        idle = len(gpus) - active
        avg_util = sum(g["util_gpu"] for g in gpus) / len(gpus) if gpus else 0

        entry = {
            "pool": name,
            "cluster": p["cluster"],
            "size": p["size"],
            "gpu_count": len(gpus),
            "active_gpus": active,
            "idle_gpus": idle,
            "avg_util": round(avg_util, 1),
            "reachable": bool(gpus),
        }

        if name in evidence_map:
            ev = evidence_map[name]
            # Override gpu_state with full evidence
            ev["gpu_state"] = gpu_results[name]
            diagnosis = analyze_root_cause(ev)
            entry["diagnosis"] = diagnosis
            entry["evidence_summary"] = {
                "bus_log_lines": len(ev["bus_log"].splitlines()),
                "has_job_processes": bool(
                    ev["processes"].strip()
                    and "ERROR:" not in ev["processes"]
                    and any(l.strip() for l in ev["processes"].splitlines())
                ),
                "has_tmux": (
                    "NO_TMUX_SESSIONS" not in ev["tmux"]
                    and "ERROR:" not in ev["tmux"]
                ),
                "disk": ev["disk"][:200] if "ERROR:" not in ev["disk"] else "unavailable",
            }
        else:
            entry["diagnosis"] = None

        results.append(entry)

    results.sort(key=lambda r: (r.get("diagnosis") is None, r["cluster"], r["pool"]))
    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    total_pools = len(results)
    diagnosed = [r for r in results if r.get("diagnosis")]
    idle_count = sum(1 for r in results if r["idle_gpus"] == r["gpu_count"] and r["gpu_count"] > 0)
    total_gpus = sum(r["gpu_count"] for r in results)
    total_idle_gpus = sum(r["idle_gpus"] for r in results)
    unreachable_count = sum(1 for r in results if not r["reachable"])

    # Summary line
    print()
    print(f"{'='*100}")
    print(f"  ATS NODE DIAGNOSIS REPORT")
    print(f"  {total_pools} pool(s) checked | {idle_count} fully idle | "
          f"{total_gpus} GPUs total | {total_idle_gpus} GPUs idle | "
          f"{unreachable_count} unreachable")
    print(f"{'='*100}")

    # Pool overview table
    print()
    print(f"  {'Pool':<42} {'GPUs':>4} {'Active':>6} {'Idle':>4} {'Avg%':>5}  {'Root Cause'}")
    print(f"  {'-'*90}")
    for r in results:
        if not r["reachable"]:
            print(f"  {r['pool']:<42} {'—':>4} {'—':>6} {'—':>4} {'—':>5}  ⏳ unreachable")
            continue

        diag = r.get("diagnosis")
        if diag:
            icon = CAUSE_ICONS.get(diag["cause"], "")
            cause = diag["cause"]
            label = f"{icon} {cause}"
        elif r["idle_gpus"] == 0:
            label = "🟢 active"
        else:
            label = ""

        print(
            f"  {r['pool']:<42} {r['gpu_count']:>4} {r['active_gpus']:>6} "
            f"{r['idle_gpus']:>4} {r['avg_util']:>5.0f}%  {label}"
        )

    # Detailed diagnosis for each idle node
    if diagnosed:
        print()
        print(f"{'='*100}")
        print(f"  DETAILED DIAGNOSIS")
        print(f"{'='*100}")

        for r in diagnosed:
            diag = r["diagnosis"]
            icon = CAUSE_ICONS.get(diag["cause"], "")
            print()
            print(f"  ┌─ {r['pool']}")
            print(f"  │  Cause: {icon} {diag['cause']}  (confidence: {diag['confidence']})")
            print(f"  │  GPUs: {r['gpu_count']} total, {r['active_gpus']} active, {r['idle_gpus']} idle, avg {r['avg_util']:.0f}%")
            if r.get("evidence_summary"):
                es = r["evidence_summary"]
                print(f"  │  Evidence: bus.log={es['bus_log_lines']} lines, "
                      f"job_procs={'yes' if es['has_job_processes'] else 'no'}, "
                      f"tmux={'yes' if es['has_tmux'] else 'no'}")
            for detail in diag.get("details", []):
                print(f"  │  • {detail}")
            action = CAUSE_ACTIONS.get(diag["cause"], "")
            print(f"  │  ➤ Action: {action}")
            print(f"  └{'─'*60}")

    # Final summary
    if diagnosed:
        cause_counts: dict[str, int] = {}
        for r in diagnosed:
            c = r["diagnosis"]["cause"]
            cause_counts[c] = cause_counts.get(c, 0) + 1
        print()
        print("  Summary of root causes:")
        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
            icon = CAUSE_ICONS.get(cause, "")
            print(f"    {icon} {cause}: {count} node(s)")
    elif idle_count == 0 and unreachable_count == 0:
        print()
        print("  ✅ All nodes are active — no idle GPUs detected.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Optionally restart Scheduled pools
    if args.restart_scheduled:
        print("Checking for Scheduled pools...")
        scheduled = list_pools_by_status(args.pattern, "Scheduled")
        if scheduled:
            print(f"Found {len(scheduled)} Scheduled pool(s). Restarting...")
            for r in restart_scheduled_pools(scheduled):
                status = "✅ restarted" if r["success"] else "❌ failed"
                print(f"  {r['pool']}: {status}")
            print()
        else:
            print("No Scheduled pools found.\n")

    # Discover Ready pools
    print(f"Discovering Ready pools matching '{args.pattern}'...")
    pools = list_pools_by_status(args.pattern, "Ready")
    if not pools:
        print("No Ready pools found matching the pattern.")
        sys.exit(0)

    print(f"Found {len(pools)} Ready pool(s).\n")

    # Run diagnosis
    results = collect_diagnoses(pools, args.timeout, args.workers, args.all)

    # Print report
    print_report(results)

    # Optionally write JSON
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "diagnosis_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON report → {report_path}")


if __name__ == "__main__":
    main()

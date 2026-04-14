#!/usr/bin/env python3
"""
Check GPU utilization on Ready Brix nodes and produce a summary report.

Usage:
    python3 check_gpu.py [--pattern GLOB] [--output-dir DIR] [--timeout SECS]

The script:
  1. Runs list_ready_pools.sh to discover Ready pools.
  2. SSHes into each Ready pool with nvidia-smi.
  3. Prints a per-node summary table highlighting idle nodes/GPUs.
"""

import argparse
import os
import subprocess
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
LIST_SCRIPT = SCRIPT_DIR / "list_ready_pools.sh"

NVIDIA_SMI_QUERY = (
    "index,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,temperature.gpu"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check GPU utilization on Ready Brix nodes")
    p.add_argument("--pattern", default="", help="Glob pattern to filter pool names")
    p.add_argument("--output-dir", default="", help="Write JSON report to this directory")
    p.add_argument("--timeout", type=int, default=30, help="SSH timeout per node in seconds")
    p.add_argument("--workers", type=int, default=4, help="Parallel SSH workers")
    p.add_argument("--monitor", type=int, default=0, metavar="SECS",
                   help="Monitor mode: re-check GPU status for SECS seconds (0=single check)")
    p.add_argument("--interval", type=int, default=10,
                   help="Seconds between checks in monitor mode (default: 10)")
    return p.parse_args()


def list_ready_pools(pattern: str) -> list[dict]:
    """Return list of {name, cluster, size} for Ready pools."""
    cmd = ["bash", str(LIST_SCRIPT)]
    if pattern:
        cmd.append(pattern)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    pools = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            pools.append({
                "name": parts[0],
                "cluster": parts[1],
                "size": parts[2] if len(parts) > 2 else "",
            })
    return pools


def list_scheduled_pools(pattern: str) -> list[dict]:
    """Return list of {name, cluster, size} for Scheduled pools."""
    import re
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    result = subprocess.run(
        ["rcall-brix", "ls"], capture_output=True, text=True, timeout=60
    )
    pools = []
    for line in result.stdout.splitlines():
        clean = ansi_re.sub('', line)
        if 'Scheduled' not in clean:
            continue
        parts = clean.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        cluster = parts[1]
        size = ""
        size_match = re.search(r'\d+ x \d+ GPU', clean)
        if size_match:
            size = size_match.group()
        if pattern:
            import fnmatch
            if not fnmatch.fnmatch(name, pattern):
                continue
        pools.append({"name": name, "cluster": cluster, "size": size})
    return pools


def restart_scheduled_pools(pools: list[dict]) -> list[dict]:
    """Restart Scheduled pools and return results."""
    results = []
    for pool in pools:
        name = pool["name"]
        try:
            result = subprocess.run(
                ["rcall-brix", "restart", name],
                capture_output=True, text=True, timeout=60,
            )
            success = result.returncode == 0
            results.append({"pool": name, "success": success, "output": result.stdout.strip()})
        except Exception as e:
            results.append({"pool": name, "success": False, "output": str(e)})
    return results


def query_gpu(pool_name: str, timeout: int) -> dict:
    """SSH into a pool and return GPU info."""
    cmd = [
        "rcall-brix", "ssh", pool_name, "--",
        "nvidia-smi",
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            if line.startswith("RUN:") or line.startswith("20"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_gpu": int(parts[2]),
                    "util_mem": int(parts[3]),
                    "mem_used": int(parts[4]),
                    "mem_total": int(parts[5]),
                    "temp": int(parts[6]),
                })
        return {"status": "ok", "gpus": gpus}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "gpus": []}
    except Exception as e:
        return {"status": f"error: {e}", "gpus": []}


def classify_node(gpus: list[dict]) -> tuple[int, int, float]:
    """Return (active_count, idle_count, avg_util)."""
    if not gpus:
        return 0, 0, 0.0
    active = sum(1 for g in gpus if g["util_gpu"] > 0)
    idle = len(gpus) - active
    avg = sum(g["util_gpu"] for g in gpus) / len(gpus)
    return active, idle, avg


def query_bus_log(pool_name: str, timeout: int, tail_lines: int = 30) -> dict:
    """SSH into a pool and return the last N lines of ~/bus.log."""
    cmd = [
        "rcall-brix", "ssh", pool_name, "--",
        f"tail -{tail_lines} ~/bus.log 2>/dev/null || echo 'NO_BUS_LOG_FOUND'",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        lines = []
        for line in result.stdout.strip().splitlines():
            if line.startswith("RUN:") or line.startswith("20") and "INFO" in line:
                continue
            lines.append(line)
        return {"status": "ok", "lines": lines, "raw": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "lines": [], "raw": ""}
    except Exception as e:
        return {"status": f"error: {e}", "lines": [], "raw": ""}


def analyze_idle_reason(log_data: dict, gpus: list[dict]) -> str:
    """Analyze bus.log content and GPU memory to determine idle reason."""
    if log_data["status"] == "timeout":
        return "⏳ bus.log check timed out"
    if log_data["status"] != "ok":
        return f"⚠️  {log_data['status']}"

    lines = log_data["lines"]
    if not lines or (len(lines) == 1 and "NO_BUS_LOG_FOUND" in lines[0]):
        return "📄 no ~/bus.log found"

    # Check for fake_eval pattern
    has_fake_eval = any("[fake_eval]" in l for l in lines)

    # Compute average memory across GPUs
    avg_mem = 0
    if gpus:
        avg_mem = sum(g["mem_used"] for g in gpus) / len(gpus)

    if has_fake_eval:
        # Check if log shows a recent transition from active to idle
        # by looking for high util followed by low util in the log
        recent_high_util = False
        recent_low_util = False
        for line in lines:
            if "[fake_eval]" in line and "util" in line:
                try:
                    util_str = line.split("util")[1].split("%")[0].strip()
                    util_val = int(util_str)
                    if util_val >= 50:
                        recent_high_util = True
                    if util_val <= 1:
                        recent_low_util = True
                except (ValueError, IndexError):
                    pass

        if avg_mem > 50000:  # >50 GB average
            if recent_high_util and recent_low_util:
                return "🔴 zombie — job just finished, fake_eval holding GPU memory"
            else:
                return "🔴 zombie — fake_eval holding GPU memory (0% compute)"
        elif avg_mem > 5000:  # >5 GB average
            return "🟡 fake_eval running — moderate memory residue"
        else:
            return "🟢 fake_eval running — baseline memory (idle/available)"

    # Check for pod provisioning
    if any("Waiting until pod is ready" in l for l in lines):
        return "⏳ pod provisioning / not ready"

    # Check for other patterns — show last meaningful line
    last_line = lines[-1] if lines else ""
    if len(last_line) > 80:
        last_line = last_line[:77] + "..."
    return f"📋 {last_line}"


def collect_bus_logs(idle_pools: list[str], timeout: int, workers: int) -> dict[str, dict]:
    """Query bus.log on idle nodes in parallel. Returns {pool_name: log_data}."""
    results = {}
    if not idle_pools:
        return results
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(query_bus_log, name, timeout): name
            for name in idle_pools
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
    return results


def print_report(results: list[dict], bus_logs: dict[str, dict] | None = None) -> None:
    """Print a formatted report to stdout."""
    total_gpus = 0
    total_active = 0
    total_idle = 0
    unreachable = 0
    idle_nodes = []

    # Header
    print()
    print(f"{'Pool':<42} {'Cluster':<28} {'GPUs':>4}  {'Active':>6}  {'Idle':>4}  {'Avg%':>5}  {'Status'}")
    print("-" * 120)

    for r in results:
        pool = r["pool"]
        cluster = r["cluster"]
        gpus = r["gpus"]
        status = r["query_status"]

        if status != "ok":
            print(f"{pool:<42} {cluster:<28} {'—':>4}  {'—':>6}  {'—':>4}  {'—':>5}  ⏳ {status}")
            unreachable += 1
            continue

        active, idle, avg = classify_node(gpus)
        n = len(gpus)
        total_gpus += n
        total_active += active
        total_idle += idle

        # Highlight fully idle or mostly idle nodes
        if active == 0:
            marker = "🔴 FULLY IDLE"
            idle_nodes.append(r)
        elif idle >= n // 2:
            marker = f"🟡 {idle} idle"
            idle_nodes.append(r)
        elif avg >= 90:
            marker = "🟢 heavy"
        else:
            marker = ""

        print(f"{pool:<42} {cluster:<28} {n:>4}  {active:>6}  {idle:>4}  {avg:>5.0f}%  {marker}")

    # Summary
    print("-" * 120)
    print(
        f"Total: {total_gpus} GPUs | "
        f"{total_active} active ({100*total_active/max(total_gpus,1):.0f}%) | "
        f"{total_idle} idle ({100*total_idle/max(total_gpus,1):.0f}%) | "
        f"{unreachable} nodes unreachable"
    )

    # Idle node detail
    if idle_nodes:
        print()
        print("=== IDLE NODE DETAILS ===")
        for r in idle_nodes:
            pool = r["pool"]
            active, idle_count, avg = classify_node(r["gpus"])
            print(f"\n  {pool}  (active={active}, idle={idle_count}, avg_util={avg:.0f}%)")
            print(f"  {'GPU':>4}  {'Model':<28} {'Util%':>5}  {'MemUsed':>8}  {'MemTotal':>8}  {'Temp°C':>6}")
            for g in r["gpus"]:
                flag = " ⬅ IDLE" if g["util_gpu"] == 0 else ""
                print(
                    f"  {g['index']:>4}  {g['name']:<28} {g['util_gpu']:>5}  "
                    f"{g['mem_used']:>7}M  {g['mem_total']:>7}M  {g['temp']:>5}{flag}"
                )
            # Print idle reason from bus.log
            if bus_logs and pool in bus_logs:
                reason = analyze_idle_reason(bus_logs[pool], r["gpus"])
                print(f"  ➤ Reason: {reason}")


def collect_results(pools: list[dict], timeout: int, workers: int) -> list[dict]:
    """Query all pools in parallel and return sorted results."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(query_gpu, p["name"], timeout): p
            for p in pools
        }
        for future in as_completed(futures):
            pool = futures[future]
            data = future.result()
            results.append({
                "pool": pool["name"],
                "cluster": pool["cluster"],
                "size": pool["size"],
                "query_status": data["status"],
                "gpus": data["gpus"],
            })
    results.sort(key=lambda r: (r["cluster"], r["pool"]))
    return results


def main() -> None:
    args = parse_args()

    # Check for Scheduled pools and restart them
    print("Checking for Scheduled pools...")
    scheduled = list_scheduled_pools(args.pattern)
    if scheduled:
        print(f"Found {len(scheduled)} Scheduled pool(s). Restarting...")
        restart_results = restart_scheduled_pools(scheduled)
        for r in restart_results:
            status = "✅ restarted" if r["success"] else f"❌ failed"
            print(f"  {r['pool']}: {status}")
        print()
    else:
        print("No Scheduled pools found.\n")

    print("Discovering Ready pools...")
    pools = list_ready_pools(args.pattern)
    if not pools:
        print("No Ready pools found.")
        sys.exit(0)

    monitor_duration = args.monitor
    interval = max(args.interval, 1)

    if monitor_duration <= 0:
        # Single check
        print(f"Found {len(pools)} Ready pool(s). Checking GPU utilization...\n")
        results = collect_results(pools, args.timeout, args.workers)
        # Collect bus.log from idle nodes
        idle_pools = []
        for r in results:
            if r["query_status"] == "ok" and r["gpus"]:
                active, idle_count, _ = classify_node(r["gpus"])
                if active == 0 or idle_count >= len(r["gpus"]) // 2:
                    idle_pools.append(r["pool"])
        bus_logs = {}
        if idle_pools:
            print(f"Checking ~/bus.log on {len(idle_pools)} idle node(s)...")
            bus_logs = collect_bus_logs(idle_pools, args.timeout, args.workers)
        print_report(results, bus_logs)
    else:
        # Monitor mode: repeatedly check for monitor_duration seconds
        print(
            f"Found {len(pools)} Ready pool(s). "
            f"Monitoring GPU utilization for {monitor_duration}s "
            f"(interval={interval}s)...\n"
        )
        start = time.monotonic()
        iteration = 0
        all_snapshots = []
        while True:
            elapsed = time.monotonic() - start
            if iteration > 0 and elapsed >= monitor_duration:
                break
            iteration += 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{'='*120}")
            print(f"  [Snapshot {iteration}]  {ts}  (elapsed {elapsed:.0f}s / {monitor_duration}s)")
            print(f"{'='*120}")
            results = collect_results(pools, args.timeout, args.workers)
            # Collect bus.log from idle nodes
            idle_pools = []
            for r in results:
                if r["query_status"] == "ok" and r["gpus"]:
                    active, idle_count, _ = classify_node(r["gpus"])
                    if active == 0 or idle_count >= len(r["gpus"]) // 2:
                        idle_pools.append(r["pool"])
            bus_logs = {}
            if idle_pools:
                print(f"Checking ~/bus.log on {len(idle_pools)} idle node(s)...")
                bus_logs = collect_bus_logs(idle_pools, args.timeout, args.workers)
            print_report(results, bus_logs)
            all_snapshots.append({"timestamp": ts, "elapsed_s": round(elapsed, 1), "results": results})
            remaining = monitor_duration - (time.monotonic() - start)
            if remaining <= 0:
                break
            sleep_time = min(interval, remaining)
            print(f"\n⏳ Next check in {sleep_time:.0f}s ...\n")
            time.sleep(sleep_time)

        print(f"\n{'='*120}")
        print(f"  Monitor complete — {iteration} snapshot(s) over {monitor_duration}s")
        print(f"{'='*120}")
        # Use last snapshot for JSON output
        results = all_snapshots[-1]["results"] if all_snapshots else []

    # Optionally write JSON
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "gpu_report.json"
        payload = all_snapshots if monitor_duration > 0 else results
        with open(report_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON report -> {report_path}")


if __name__ == "__main__":
    main()

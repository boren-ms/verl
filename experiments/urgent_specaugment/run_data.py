#!/usr/bin/env python3
"""Remote CPU generation and independent loader audit, with a durable exit receipt."""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(exist_ok=True, parents=True)
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "experiments/urgent_specaugment/config"
    generator = [
        sys.executable,
        str(root / "scripts/build_ls_specaugment_dataset.py"),
        "--source-manifest",
        args.source_manifest,
        "--source-sha256",
        args.source_sha256,
        "--config",
        str(config_dir / "augment_mild.yaml"),
        "--output-dir",
        args.output_dir,
        "--work-dir",
        str(args.work_dir),
        "--workers",
        "24",
    ]
    receipt = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "RUNNING"}
    receipt_path = args.work_dir / "pipeline_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        # A separate smoke destination cannot accidentally publish full data.
        smoke = generator.copy()
        smoke[smoke.index("--output-dir") + 1] = args.output_dir + "/smoke"
        smoke[smoke.index("--work-dir") + 1] = str(args.work_dir / "smoke")
        subprocess.run([*smoke, "--max-examples", "16"], cwd=root, check=True)
        subprocess.run(generator, cwd=root, check=True)
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/verify_ls_specaugment_dataset.py"),
                "--dataset-config",
                str(config_dir / "data_specaug_mild.yaml"),
                "--work-dir",
                str(args.work_dir),
            ],
            cwd=root,
            check=True,
        )
        receipt.update(status="SUCCEEDED", verification=str(args.work_dir / "verification.json"))
    except Exception as error:
        receipt.update(status="FAILED", error=str(error))
        raise
    finally:
        receipt["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

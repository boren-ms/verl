#!/usr/bin/env python3
"""Independently audit the entire cohort through the pinned historical loader."""

import argparse
import hashlib
import json
from pathlib import Path

import blobfile as bf
import yaml
from recipe.phimm.data.dataset import create_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.dataset_config.read_text())
    manifest_path = config["jsonl_paths"]
    root = manifest_path.rsplit("/", 1)[0]
    original = (args.work_dir / "original_manifest.jsonl").read_bytes()
    summary = json.loads((args.work_dir / "summary.json").read_text())
    assert hashlib.sha256(original).hexdigest() == summary["source_manifest_sha256"]
    with bf.BlobFile(manifest_path, "rb") as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == summary["output_manifest_sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    old = [json.loads(line) for line in original.splitlines()]
    assert len(rows) == len(old) == 20424 and not summary["smoke_only"]
    assert len({r["specaugment_seed"] for r in rows}) == len(rows)
    for i, (source, row) in enumerate(zip(old, rows, strict=True)):
        for key in ("id", "text", "prompt", "gt_output", "keywords", "source_audio_path"):
            assert source.get(key) == row.get(key), (i, key)
        assert source.get("audio_chunk") is None and row["audio_chunk"] is None
        assert row["augmentation_order"] == ["specaugment"] and row["speed_factor"] == 1
        assert row["specaugment_config"] == summary["policy"]["specaugment"]
        assert all(row.get(key) is None for key in ("noise_path", "rir_path", "snr_db"))
        assert not any(key in row for key in ("response", "raw_response", "wer", "edge_wer"))
        assert json.loads((args.work_dir / "rows" / f"{i:08d}.json").read_text()) == row
        trace = row["specaugment_sampled"]
        warp = trace["warp"]
        if warp:
            assert abs(warp["destination_frame"] - warp["anchor_frame"]) <= 20
        else:
            assert trace["time_frames"] <= 41
        assert len(trace["masks"]) == 2
        for mask in trace["masks"]:
            length = trace["frequency_bins"] if mask["axis"] == "frequency" else trace["time_frames"]
            limit = 8 if mask["axis"] == "frequency" else min(10, int(0.02 * length))
            assert 0 <= mask["width"] <= limit and 0 <= mask["start"] <= length - mask["width"]
    loaded = create_datasets(config)
    assert len(loaded) == len(rows)
    for i, row in enumerate(rows):
        example = loaded[i]
        assert example["audio_path"] == row["audio_path"]
        assert example["data_source"] == "noisy_ls"
        assert example["reward_model"]["ground_truth"] == row["text"]
    proof = dict(
        status="PASSED",
        rows=len(rows),
        canonical_loader_rows=len(loaded),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        complete_labels_order_and_source_lineage_verified=True,
        independent_seeds_and_random_ranges_verified=True,
        uploaded_wav_sha256_verified=summary["uploaded_wav_sha256_verified"],
        all_audio_decoded_finite=summary["all_audio_decoded_finite"],
    )
    payload = json.dumps(proof, indent=2) + "\n"
    (args.work_dir / "verification.json").write_text(payload)
    with bf.BlobFile(root + "/verification.json", "w") as stream:
        stream.write(payload)
    for script in ("specaugment_wav.py", "build_ls_specaugment_dataset.py", "verify_ls_specaugment_dataset.py"):
        bf.copy(str(Path(__file__).with_name(script)), root + "/code/" + script, overwrite=False)
    print("VERIFICATION_JSON=" + json.dumps(proof))


if __name__ == "__main__":
    main()

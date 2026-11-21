"""Prepare AMI-IHM + LS training and evaluation-only DavidAI test rows."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from vibevoice_asr.decode import read_manifest

PAPER_TEST = {f"{stem}{suffix}" for stem in ("EN2002", "ES2004", "IS1009", "TS3003") for suffix in "abcd"}


def prepare_davidai_row(row):
    """Restore one complete mixed recording, keeping its original reference clock.

    Independent SVAD calls do not guarantee consistent speaker IDs. Native
    offline decoding therefore uses the common parent waveform in one call.
    """
    row = dict(row)
    paths = row["audio_path"]
    paths = paths if isinstance(paths, list) else [paths]
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("DavidAI requires nonempty audio paths")
    parents = {path.split("#", 1)[0] for path in paths}
    if len(parents) != 1:
        raise ValueError("DavidAI segments must belong to one parent recording")
    duration = row.get("evaluation_duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 901:
        raise ValueError("DavidAI requires a finite complete recording duration <= 901 seconds")
    if not row.get("reference_segments") or not row.get("reference_words"):
        raise ValueError("DavidAI evaluation requires attached speaker intervals and words")
    row.update(
        audio_path=parents.pop(),
        data_source="davidai_test",
        prompt="<audio>",
        language="English",
        validation_role="evaluation_only_no_tuning",
        decoding_mode="full_recording",
    )
    row.pop("audio", None)
    for key in ("svad_source", "svad_bounds", "svad_parent_duration", "svad_method"):
        row.pop(key, None)
    return row


def recording_id(row):
    for key in ("recording_id", "source_recording", "id"):
        match = re.search(r"[A-Z]{2}\d{4}[a-z]?", str(row.get(key, "")))
        if match:
            return match.group()
    raise ValueError("Cannot establish the AMI recording identity")


def validate_splits(train, dev):
    train_ids = {recording_id(row) for row in train}
    dev_ids = {recording_id(row) for row in dev}
    if train_ids & dev_ids or (train_ids | dev_ids) & PAPER_TEST:
        raise ValueError("AMI training/dev/test recording overlap")
    return {
        "train_recordings": sorted(train_ids),
        "dev_recordings": sorted(dev_ids),
        "test_recordings": sorted(PAPER_TEST),
    }


def prepare_row(row, source, *, sdm_map=None):
    row = dict(row)
    if source == "ami_dev_sdm":
        recording = recording_id(row)
        start, end = row["source_start_s"], row["source_end_s"]
        if not 0 <= start < end or recording not in sdm_map:
            raise ValueError("SDM audio and matching original clip bounds are required")
        row["audio_path"] = f"{sdm_map[recording]}#{start}:{end}"
        row.pop("audio", None)
        row["microphone"] = "Array1-01"
    elif source == "ls":
        from recipe.phimm.utils.audio import load_raw_audio

        waveform, sample_rate = load_raw_audio(row)
        duration = len(waveform) / sample_rate
        row.update(
            duration_s=duration,
            evaluation_duration_s=duration,
            reference_segments=[{"start": 0.0, "end": duration, "speaker": "ls_single"}],
            reference_words=[],
            annotation_kind="synthetic_single_speaker_full_audio",
        )
    else:
        row["microphone"] = "IHM"
    duration = row.get("evaluation_duration_s") or row.get("duration_s")
    if not isinstance(duration, (int, float)) or not 0 < duration <= 40:
        raise ValueError("RL clips require a known positive duration of at most 40 seconds")
    if not row.get("reference_segments"):
        raise ValueError("Diarization reward needs real reference speaker intervals")
    row.update(
        evaluation_duration_s=duration,
        data_source=source,
        prompt="<audio>",
        language="English",
    )
    row.setdefault("id", hashlib.sha256(row["audio_path"].encode()).hexdigest())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ami-train", required=True)
    parser.add_argument("--ami-dev", help="Optional AMI split audit only; not used for validation")
    parser.add_argument("--ls", required=True)
    parser.add_argument("--davidai-test", required=True, help="DavidAI English 45-recording internal-test manifest")
    parser.add_argument("--davidai-smoke", help="Optional disjoint 3-recording manifest for leakage auditing")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit-per-source", type=int, help="Explicit smoke subset; omitted uses all rows")
    args = parser.parse_args()
    if args.limit_per_source is not None and args.limit_per_source <= 0:
        parser.error("--limit-per-source must be positive")
    train, train_paths = read_manifest(args.ami_train)
    dev, dev_paths = read_manifest(args.ami_dev) if args.ami_dev else ([], [])
    ls, ls_paths = read_manifest(args.ls)
    davidai, davidai_paths = read_manifest(args.davidai_test)
    smoke, smoke_paths = read_manifest(args.davidai_smoke) if args.davidai_smoke else ([], [])
    audit = validate_splits(train, dev)
    if {row["id"] for row in davidai} & {row["id"] for row in smoke}:
        raise ValueError("DavidAI internal-test and smoke recordings overlap")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = {}
    for source, rows in (("ami_train_ihm", train), ("davidai_test", davidai), ("ls", ls)):
        selected = rows[: args.limit_per_source] if args.limit_per_source else rows
        prepared = [
            prepare_davidai_row(row) if source == "davidai_test" else prepare_row(row, source) for row in selected
        ]
        (args.output_dir / f"{source}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared)
        )
        counts[source] = len(prepared)
    audit.update(
        counts=counts,
        partial=args.limit_per_source is not None,
        source_paths={
            "train": train_paths,
            "ami_dev_audit_only": dev_paths,
            "ls": ls_paths,
            "davidai_test": davidai_paths,
            "davidai_smoke_audit_only": smoke_paths,
        },
        davidai_test_recordings=len(davidai),
        davidai_test_duration_s=sum(row["evaluation_duration_s"] for row in davidai),
        validation_policy=(
            "DavidAI internal-test evaluation only; no reward optimization, tuning or checkpoint selection"
        ),
        ls_reference_policy="One synthetic speaker over full decoded audio; no real diarization labels",
    )
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

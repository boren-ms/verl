"""Score complete full-recording transcriptions against published RTTM/UEM files."""

import argparse
import dataclasses
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from vibevoice_asr.transcription import NON_SPEECH_TAGS, close_truncated_json, parse_transcription


def hypothesis_rttm(details, expected_ids, *, allow_unassigned_speech=False, allow_truncated=False):
    """Require exact coverage and real speaker labels; never invent diarization."""
    ids = [row["id"] for row in details]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Predictions must cover every reference recording exactly once")
    lines = []
    for row in details:
        if "error" in row or "segments" not in row:
            raise ValueError(f"Unparsed prediction for {row['id']}")
        if row.get("truncated") and not allow_truncated:
            raise ValueError(f"Truncated full-recording prediction for {row['id']}")
        recording_lines = []
        for segment in row["segments"]:
            # Unintelligible speech is speech for DER even though text WER
            # excludes this tag. It must carry a model-emitted speaker label.
            if segment["text"].strip().lower() in NON_SPEECH_TAGS - {"[unintelligible speech]"}:
                continue
            speaker = segment["speaker"]
            if speaker is None and allow_unassigned_speech:
                # An absent speaker is an abstention: emit no hypothesis
                # speaker activity. The entire reference/UEM remains scored.
                continue
            if speaker is None or not str(speaker).strip() or any(c.isspace() for c in str(speaker)):
                raise ValueError(f"Missing/invalid speech speaker in {row['id']}")
            start, end = segment["start_s"], segment["end_s"]
            import math

            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError(f"Invalid speech timing in {row['id']}")
            if end > start:
                recording_lines.append(
                    f"SPEAKER {row['id']} 1 {start:.6f} {end - start:.6f} <NA> <NA> {speaker} <NA> <NA>\n"
                )
        if not recording_lines:
            # MeetEval intersects RTTM IDs, so reject rather than lose this
            # recording from the corpus denominator without warning.
            raise ValueError(f"No scorable speech for {row['id']}")
        lines.extend(recording_lines)
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--uem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reparse-raw", action="store_true", help="Parse saved raw responses without new inference")
    parser.add_argument("--allow-unassigned-speech", action="store_true", help="Score absent speakers as abstentions")
    parser.add_argument("--allow-truncated", action="store_true", help="Score emitted intervals against the full UEM")
    args = parser.parse_args()
    import meeteval.der
    from meeteval.io import RTTM, UEM

    reference = RTTM.load(args.reference)
    uem = UEM.load(args.uem)
    ids = set(reference.filenames())
    if set(uem.filenames()) != ids:
        raise ValueError("Reference RTTM and UEM recording sets differ")
    details = [json.loads(line) for line in args.details.read_text().splitlines() if line.strip()]
    if args.reparse_raw:
        for row in details:
            raw = row["raw_response"]
            if row.get("truncated") and args.allow_truncated:
                raw = close_truncated_json(raw)
            row["segments"] = parse_transcription(
                raw, allow_invalid_event_times=True, allow_unassigned_speech=args.allow_unassigned_speech
            )
            row.pop("error", None)
    payload = hypothesis_rttm(
        details, ids, allow_unassigned_speech=args.allow_unassigned_speech, allow_truncated=args.allow_truncated
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hyp = args.output_dir / "hypothesis.rttm"
    hyp.write_text(payload)
    result = {
        "recordings": sorted(ids),
        "meeteval_version": version("meeteval"),
        "truncated_recordings": [row["id"] for row in details if row.get("truncated")],
        "unassigned_speech": [
            {"recording": row["id"], **segment}
            for row in details
            for segment in row["segments"]
            if segment["speaker"] is None
            and segment["text"].strip().lower() not in NON_SPEECH_TAGS - {"[unintelligible speech]"}
        ],
        "allow_unassigned_speech": args.allow_unassigned_speech,
        "allow_truncated": args.allow_truncated,
        "abstention_policy": "No speaker activity emitted for unassigned speech; full reference and UEM always scored",
        "collar_definition": "md-eval -c excludes +/- collar seconds around each reference boundary",
        "inputs": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in {"details": args.details, "reference": args.reference, "uem": args.uem}.items()
        },
        "policy": "Full recordings; optimal speaker mapping per recording; overlap included; words-only references",
        "settings": {},
    }
    if args.reparse_raw:
        (args.output_dir / "reparsed_details.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details)
        )
    for collar in (0, 0.25):
        scores = meeteval.der.md_eval_22(args.reference, hyp, collar=collar, regions="all", uem=args.uem)
        if set(scores) != ids:
            raise ValueError("MeetEval omitted reference recordings")
        total = sum(scores.values())
        result["settings"][str(collar)] = {
            "collar_s": collar,
            "regions": "all",
            "aggregate": {k: float(v) for k, v in dataclasses.asdict(total).items()},
            "per_recording": {
                key: {k: float(v) for k, v in dataclasses.asdict(value).items()}
                for key, value in sorted(scores.items())
            },
        }
    (args.output_dir / "der.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Adapt existing Inhouse EWER VAD clips to complete longform evaluation rows.

No audio is segmented or rewritten. Each output row retains one original
recording, its ordered pre-segmented audio paths, and both reference forms.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blobfile as bf

SOURCE_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3"
OUTPUT_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_EWER_en_US_entity_v3_presegment_v1"
_CLIP = re.compile(r"_segment_(\d+)_(\d+)_(\d+)\.wav$")
_TIMING = re.compile(r"\[\s*-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s*\]")


def _without_timing(text):
    return " ".join(_TIMING.sub(" ", text).split())


def normalize_domain(original, clips, *, domain, source_root=SOURCE_ROOT):
    """Fail on missing parents, clips, ordering, or changed entity references."""
    parents = {row["UUID"]: row for row in original}
    if not parents or len(parents) != len(original):
        raise ValueError(f"Empty or duplicate original recording IDs: {domain}")
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["UUID"]].append(clip)
    if set(groups) != set(parents):
        raise ValueError(f"Pre-segmented recording coverage differs: {domain}")
    output = []
    for uid, parent in parents.items():
        reference = parent["Transcription"].strip()
        display_reference = parent.get("DisplayTranscription", "").strip() or reference
        if not reference:
            raise ValueError(f"Missing original reference: {domain}/{uid}")
        entries = []
        for clip in groups[uid]:
            match = _CLIP.search(clip["WavPath"])
            if not match or not Path(clip["WavPath"]).name.startswith(uid + "_"):
                raise ValueError(f"Unrecognized VAD clip identity: {clip['WavPath']}")
            index, start, end = map(int, match.groups())
            if not 0 <= start < end:
                raise ValueError(f"Invalid VAD clip bounds: {clip['WavPath']}")
            if _without_timing(clip.get("Transcription", "")) not in (
                "",
                _without_timing(reference),
                _without_timing(display_reference),
            ):
                raise ValueError(f"Changed entity reference: {domain}/{uid}")
            expected_prefix = f"{source_root}/{domain}/vad_segments/"
            if not clip["WavPath"].startswith(expected_prefix):
                raise ValueError(f"Unexpected pre-segmented audio root: {clip['WavPath']}")
            entries.append((index, start, end, clip["WavPath"]))
        entries.sort()
        if [entry[0] for entry in entries] != list(range(len(entries))):
            raise ValueError(f"Missing or duplicate VAD clip indices: {domain}/{uid}")
        # Existing VAD may trim leading silence and overlap adjacent clips.
        if any(b[1] < a[1] for a, b in zip(entries, entries[1:], strict=False)):
            raise ValueError(f"Invalid VAD clip timeline: {domain}/{uid}")
        if sum(bool(clip.get("Transcription", "").strip()) for clip in groups[uid]) != 1:
            raise ValueError(f"Expected exactly one full VAD reference: {domain}/{uid}")
        parent_audio = f"{source_root}/{domain}/wav/{Path(parent['WavPath']).name}"
        output.append(
            {
                "id": uid,
                "audio_path": [entry[3] for entry in entries],
                "parent_audio_path": parent_audio,
                "text": display_reference,
                "entity_transcription": reference,
                "language": "English",
                "data_source": "eer_"
                + domain.split("_DTEST_", 1)[0].replace("DomainSet", "").rstrip("_")
                + "_"
                + domain.split("_DTEST_", 1)[1].split("_Entity_", 1)[0],
                "n_segments": len(entries),
                "segment_starts_samples": [entry[1] for entry in entries],
                "segment_ends_samples": [entry[2] for entry in entries],
                "reference_scope": "parent",
                "source_manifest": f"{source_root}/{domain}/vad_segments.jsonl",
                "reference_manifest": f"{source_root}/{domain}/test.jsonl",
            }
        )
    return output


def read_rows(uri, cache):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with bf.BlobFile(uri, "rb") as stream:
            cache.write_bytes(stream.read())
    return [json.loads(line) for line in cache.read_bytes().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    summary = {"source_root": args.source_root, "output_root": args.output_root, "datasets": []}
    domains = sorted(bf.listdir(args.source_root))

    def prepare_domain(domain):
        domain = bf.basename(domain.rstrip("/"))
        rows = {
            name: read_rows(
                f"{args.source_root}/{domain}/{name}.jsonl", args.cache_dir / "source" / domain / f"{name}.jsonl"
            )
            for name in ("test", "vad_segments")
        }
        normalized = normalize_domain(rows["test"], rows["vad_segments"], domain=domain, source_root=args.source_root)
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized).encode()
        path = args.cache_dir / "normalized" / f"{domain}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        uri = f"{args.output_root}/{domain}.jsonl"
        if args.publish:
            if bf.exists(uri):
                with bf.BlobFile(uri, "rb") as stream:
                    if stream.read() != data:
                        raise ValueError(f"Refusing to overwrite changed manifest: {uri}")
            else:
                with bf.BlobFile(uri, "wb") as stream:
                    stream.write(data)
            with bf.BlobFile(uri, "rb") as stream:
                if stream.read() != data:
                    raise ValueError(f"Published manifest differs: {uri}")
        entry = {
            "domain": domain,
            "data_source": normalized[0]["data_source"],
            "manifest": uri,
            "parents": len(normalized),
            "segments": len(rows["vad_segments"]),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        print(json.dumps(entry), flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        summary["datasets"] = list(executor.map(prepare_domain, domains))
    (args.cache_dir / "preparation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

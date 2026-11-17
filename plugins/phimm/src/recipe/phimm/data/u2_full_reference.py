"""Publish U2 clips with original full-parent references, without decoding audio.

Run on an existing node with Blob access:
  python -m recipe.phimm.data.u2_full_reference --output-root az://.../version
Existing output files must match exactly; manifests are never overwritten.
"""

import argparse
import collections
import gzip
import hashlib
import json

import blobfile as bf

SOURCES = ("earnings21", "earnings22", "coraal", "earnings22_cleaned_aa")
EVALUATION_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation"
LEGACY_ROOT = EVALUATION_ROOT + "/OpenLong_all_segments_2parents_2609r2b_step400_v1"
FULL_ROOT = EVALUATION_ROOT + "/OpenLong_all_segments_2parents_fullref_2609r2b_step400_v1"


def read_rows(uri):
    with bf.BlobFile(uri, "rb") as stream:
        data = stream.read()
    if uri.endswith(".gz"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def reference_manifests(source, parents):
    if source == "earnings22_cleaned_aa":
        return [
            EVALUATION_ROOT + "/ArtificialAnalysis_seg_presegment_30s/Earnings22-Cleaned-AA-chunked/"
            "earnings22_cleaned_aa_chunked_v1_zip.jsonl"
        ]
    return sorted(
        {
            EVALUATION_ROOT
            + "/OpenASR_long_seg_presegment_30s/"
            + parent.split("/openasr_long_jsonl/", 1)[1].split("/audio/", 1)[0]
            + "/data_zip.jsonl"
            for parent in parents
        }
    )


def full_reference_rows(rows, reference_rows, normalize):
    """Retain the exact clips; replace inferred segment text with parent text."""
    groups = collections.defaultdict(list)
    references = collections.defaultdict(set)
    for row in rows:
        groups[row["parent_audio_path"]].append(row)
    for row in reference_rows:
        if row.get("parent_audio_path") in groups:
            references[row["parent_audio_path"]].add(row.get("text", row.get("transcript")))
    if set(groups) != set(references):
        raise ValueError("Missing original parent references")
    output, audit = [], []
    for parent, segments in groups.items():
        candidates = references[parent]
        if len(candidates) != 1 or not isinstance(next(iter(candidates)), str):
            raise ValueError(f"Inconsistent full reference for {parent}")
        reference = next(iter(candidates))
        segments = sorted(segments, key=lambda row: (float(row["seg_start"]), int(row["seg_index"])))
        if sorted(int(row["seg_index"]) for row in segments) != list(range(len(segments))):
            raise ValueError(f"Missing or duplicate segment indices for {parent}")
        if {int(row["n_segments"]) for row in segments} != {len(segments)}:
            raise ValueError(f"Incomplete parent group for {parent}")
        if normalize(" ".join(row["text"] for row in segments)) != normalize(reference):
            raise ValueError(f"Joined inferred reference differs from original parent for {parent}")
        for row in segments:
            new = {**row, "text": reference, "reference_scope": "parent", "legacy_segment_text": row["text"]}
            # These old diagnostics describe segment-reference scoring only.
            for key in ("baseline_n_err", "baseline_n_ref", "baseline_wer"):
                if key in new:
                    new["legacy_segment_" + key] = new.pop(key)
            output.append(new)
        canonical = [row for row in reference_rows if row.get("parent_audio_path") == parent]
        audit.append(
            {
                "parent_audio_path": parent,
                "segments": len(segments),
                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                "normalized_join_matches_original": True,
                "u2_span": [min(float(x["seg_start"]) for x in segments), max(float(x["seg_end"]) for x in segments)],
                "canonical_span": [
                    min(float(x["seg_start"]) for x in canonical),
                    max(float(x["seg_end"]) for x in canonical),
                ],
            }
        )
    return output, audit


def publish_identical(uri, data):
    if bf.exists(uri):
        with bf.BlobFile(uri, "rb") as stream:
            if hashlib.sha256(stream.read()).digest() != hashlib.sha256(data).digest():
                raise ValueError(f"Refusing to overwrite different data at {uri}")
    else:
        with bf.BlobFile(uri, "wb") as stream:
            stream.write(data)
    with bf.BlobFile(uri, "rb") as stream:
        if stream.read() != data:
            raise ValueError(f"Upload verification failed: {uri}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=FULL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import _HFEnglishTextNormalizer

    normalize = _HFEnglishTextNormalizer()
    summary = {"reference_scope": "parent", "legacy_root": LEGACY_ROOT, "datasets": {}}
    for source in SOURCES:
        rows = read_rows(LEGACY_ROOT + "/" + source + ".jsonl")
        paths = reference_manifests(source, {row["parent_audio_path"] for row in rows})
        reference_rows = [row for path in paths for row in read_rows(path)]
        output, audit = full_reference_rows(rows, reference_rows, normalize)
        assert len({row["id"] for row in output}) == len(output) == len(rows)
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output).encode()
        compressed = gzip.compress(payload, mtime=0)
        uri = args.output_root.rstrip("/") + "/" + source + ".jsonl.gz"
        if not args.dry_run:
            publish_identical(uri, compressed)
        summary["datasets"][source] = {
            "uri": uri,
            "segments": len(output),
            "parents": len(audit),
            "manifest_sha256": hashlib.sha256(compressed).hexdigest(),
            "compressed_bytes": len(compressed),
            "reference_manifests": paths,
            "parent_audit": audit,
        }
        print(
            json.dumps(
                {"source": source, "segments": len(output), "parents": len(audit), "published": not args.dry_run}
            ),
            flush=True,
        )
    if not args.dry_run:
        # Publish evidence only after every immutable manifest has been verified.
        publish_identical(
            args.output_root.rstrip("/") + "/manifest_verified.json",
            (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode(),
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

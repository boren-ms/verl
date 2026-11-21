"""Prepare MoE's exact en-US EER or TER suite from existing segments."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import blobfile as bf

REFERENCE_ROOT = "az://orngwus2cresco/data/speech/users/ruchaofan/Evaluation/InhouseASR_2605/en-US"
SEGMENT_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment/en-US"
SEGMENT_30S_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/InhouseASR_2605_seg_presegment_30s/en-US"
OUTPUT_ROOT = "az://orngwus2cresco/data/boren/data/Evaluation/dtest_enus_eer_v1"
# Ordered exactly as MoE/scripts/audio_eval/result_orders/2611_eval_rows.tsv.
EER_DATASETS = (
    "CustomerSpeechDomainSet_DTEST_Banking_Entity_FY23Q4_en-US_DTEST",
    "CustomerSpeechDomainSet_DTEST_Medical_Entity_FY23Q4_en-US_DTEST",
    "OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST",
    "OnlineMeetings_CS_Product_FY22_en-US_DTEST",
    "Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection",
)
TER_DATASETS = (
    "Conversation_DTEST_FY21Q1_en-US",
    "Conversation_OnlineMeetings_DTEST_FY25Q3_en-US_DTEST_OfflineDataCollection",
    "OnlineMeetings_CS_Product_FY22_en-US_DTEST",
    "OnlineMeetings_CS_Shiproom_FY22_en-US_DTEST",
    "Dictation_Commonset_OfficeOffline_FY24Q3_en-US_DTEST_OfflineDataCollection",
)
TER_OUTPUT_ROOT = OUTPUT_ROOT.replace("_eer", "_ter")
SUITES = {"eer": EER_DATASETS, "ter": TER_DATASETS}
OUTPUT_ROOTS = {"eer": OUTPUT_ROOT, "ter": TER_OUTPUT_ROOT}
# Short stable routing/report names; canonical source names remain in manifests.
DATA_SOURCES = {
    "eer": dict(zip(EER_DATASETS, ("banking", "medical", "shiproom", "product", "meetings"), strict=True)),
    "ter": dict(
        zip(
            TER_DATASETS,
            ("conversation", "meetings", "product", "shiproom", "dictation"),
            strict=True,
        )
    ),
}
WORKBOOK_S2 = (7.62, 6.51, 46.77, 31.07, 23.77)


def recording_statistics(rows, segments=()):
    """Count display words once per recording; clips have no aligned references."""
    lengths = []
    words = []
    segments_per_recording = []
    source_durations = {clip["WavPath"]: float(clip["seg_end"]) - float(clip["seg_start"]) for clip in segments}
    for row in rows:
        text = re.sub(r"\[\s*-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s*\]|<[^>]*>", " ", row["text"])
        words.append(sum(any(char.isalnum() for char in token) for token in text.split()))
        segments_per_recording.append(len(row["audio_path"]))
        for path in row["audio_path"]:
            if "#" in path:
                start, end = map(float, path.rsplit("#", 1)[1].split(":"))
                lengths.append(end - start)
            else:
                lengths.append(source_durations[path])

    def distribution(values):
        return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}

    return {
        "display_words": sum(words),
        "words_per_recording": distribution(words),
        "segments_per_recording": distribution(segments_per_recording),
        "segment_seconds": distribution(lengths),
        "summed_segment_hours": sum(lengths) / 3600,
        "segment_word_counts": None,
        "word_count_definition": (
            "DisplayTranscription whitespace tokens with letters/digits, tags/times removed; not TER tokens"
        ),
        "segment_word_count_note": "Existing segments copy full-recording references; no aligned segment word counts",
    }


def normalize_recordings(original, segments, *, domain, source_manifest, suite="eer"):
    """Keep exact source references, parent IDs, clip selectors and ordering."""
    if domain not in SUITES[suite]:
        raise ValueError(f"Dataset outside workbook suite: {domain}")
    parents = {row["UUID"]: row for row in original}
    if not parents or len(parents) != len(original):
        raise ValueError("Empty or duplicate parent IDs")
    groups = defaultdict(list)
    for row in segments:
        groups[row["UUID"]].append(row)
    if set(groups) != set(parents):
        raise ValueError("Segment and original recording coverage differ")
    output = []
    for uid, parent in parents.items():
        clips = sorted(groups[uid], key=lambda row: int(row["seg_index"]))
        if [int(row["seg_index"]) for row in clips] != list(range(len(clips))):
            raise ValueError(f"Missing or duplicate clips: {uid}")
        paths = []
        previous_start = -1
        for clip in clips:
            if int(clip["n_segments"]) != len(clips):
                raise ValueError(f"Incomplete segment group: {uid}")
            for key in ("Transcription", "DisplayTranscription"):
                if clip[key] != parent[key] or not isinstance(parent[key], str) or not parent[key].strip():
                    raise ValueError(f"Changed or empty {key}: {uid}")
            start, end = float(clip["seg_start"]), float(clip["seg_end"])
            if not 0 <= start < end or start < previous_start or end - start > 60:
                raise ValueError(f"Invalid or over-60s existing clip: {uid}")
            previous_start = start
            audio, _, selector = clip["WavPath"].partition("#")
            if audio != clip["parent_audio_path"] or Path(audio).name != Path(parent["WavPath"]).name:
                raise ValueError(f"Changed source recording: {uid}")
            if selector and list(map(float, selector.split(":"))) != [start, end]:
                raise ValueError(f"Changed clip bounds: {uid}")
            if not selector and (len(clips) != 1 or start != 0):
                raise ValueError(f"Unbounded path is not a single complete recording: {uid}")
            if not audio.startswith(f"{REFERENCE_ROOT}/{domain}/wav/"):
                raise ValueError(f"Unexpected source audio root: {uid}")
            paths.append(clip["WavPath"])
        output.append(
            {
                "id": uid,
                "audio_path": paths,
                "parent_audio_path": clips[0]["parent_audio_path"],
                "text": parent["DisplayTranscription"],
                "entity_transcription": parent["Transcription"],
                "language": "English",
                "data_source": DATA_SOURCES[suite][domain],
                "n_segments": len(clips),
                "reference_scope": "parent",
                "source_manifest": source_manifest,
                "reference_manifest": f"{REFERENCE_ROOT}/{domain}/test.jsonl",
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="eer")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root or OUTPUT_ROOTS[args.suite]
    inventory = {"suite": args.suite, "locale": "en-US", "datasets": []}
    for domain in SUITES[args.suite]:
        root = SEGMENT_ROOT if domain.startswith("CustomerSpeech") else SEGMENT_30S_ROOT
        clip_uri = f"{root}/{domain}/test.jsonl"
        source = []
        for uri in (f"{REFERENCE_ROOT}/{domain}/test.jsonl", clip_uri):
            with bf.BlobFile(uri, "r") as stream:
                source.append([json.loads(line) for line in stream if line.strip()])
        rows = normalize_recordings(*source, domain=domain, source_manifest=clip_uri, suite=args.suite)
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        name = domain + ".jsonl"
        (args.cache_dir / name).write_bytes(data)
        uri = f"{output_root}/{name}"
        if args.publish:
            if bf.exists(uri):
                with bf.BlobFile(uri, "rb") as stream:
                    if stream.read() != data:
                        raise ValueError(f"Refusing changed manifest: {uri}")
            else:
                with bf.BlobFile(uri, "wb") as stream:
                    stream.write(data)
            with bf.BlobFile(uri, "rb") as stream:
                if stream.read() != data:
                    raise ValueError(f"Published bytes differ: {uri}")
        entry = dict(
            domain=domain,
            data_source=rows[0]["data_source"],
            manifest=uri,
            parents=len(rows),
            segments=sum(row["n_segments"] for row in rows),
            sha256=hashlib.sha256(data).hexdigest(),
            source_manifest=clip_uri,
            statistics=recording_statistics(rows, source[1]),
        )
        inventory["datasets"].append(entry)
        print(json.dumps(entry), flush=True)
    (args.cache_dir / "preparation.json").write_text(json.dumps(inventory, indent=2) + "\n")
    if args.publish:
        with bf.BlobFile(f"{output_root}/preparation.json", "w") as stream:
            json.dump(inventory, stream, indent=2)


if __name__ == "__main__":
    main()

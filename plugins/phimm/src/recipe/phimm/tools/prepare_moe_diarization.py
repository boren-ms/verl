"""Export speaker-tagged SFT windows from the Cascades downstream blob chunks."""

import argparse
import concurrent.futures
import json
import math
import re
import struct
import subprocess
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from urllib.request import Request, urlopen

import soundfile as sf
import yaml
from scipy.signal import resample_poly

ROOT = "https://tsstd01uks.blob.core.windows.net/data/users/xioxiao/am_data/en_long"
SOURCES = {
    "ami_train": ("ami_ihm_mix/train", "train"),
    "ami_dev": ("ami_ihm_mix/dev", "validation"),
    "icsi": ("icsi_ihm_mix_profiles/train_dev_eval", "icsi_audit"),
    "fisher": ("Fisher/train_display_meta", "train"),
    "apptek": ("AppTekTwoSidesTrainEn", "train"),
    "definedai": ("DefinedAIEn", "train"),
    "podcast": ("podcast3870hr", "train"),
    "vox": (
        "vox_simulated/vox12dt_tr_ns5_5_nsent3_5_ovlp0.0_0.01_0.5s_gap0_2.0_rir0.2s_profile_nsent2_2",
        "train",
    ),
}
PROMPT = (
    "Transcribe the English audio and identify speaker changes. "
    "Prefix each change with speaker1, speaker2, and so on, numbering by first appearance. "
    "Return only the speaker-tagged transcript, without timestamps.<audio>"
)
EXTRA = [
    "recording_id",
    "source_recording",
    "speaker_map",
    "duration_s",
    "evaluation_duration_s",
    "source_start_s",
    "source_end_s",
    "reference_segments",
    "reference_words",
    "reference_turns",
    "annotation_kind",
    "source_chunk",
    "source_record_index",
    "split",
    "speaker_count",
]


def request(url, token, byte_range=None):
    headers = {"Authorization": f"Bearer {token}", "x-ms-version": "2021-12-02"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    response = urlopen(Request(url, headers=headers), timeout=90)
    if byte_range and (
        response.status != 206
        or not response.headers["Content-Range"].startswith(f"bytes {byte_range[0]}-{byte_range[1]}/")
    ):
        response.close()
        raise ValueError(f"Incorrect range response for {url}")
    return response


def read_info(raw, expected):
    if raw[:8] != b"Info\x01\x00\x00\x00":
        raise ValueError("Unexpected Info chunk header")
    records, offset = {}, 8
    while offset < len(raw):
        index, size = struct.unpack_from("<II", raw, offset)
        offset += 8
        if index in records or offset + size > len(raw):
            raise ValueError("Duplicate or incomplete Info record")
        records[index] = json.loads(raw[offset : offset + size])
        offset += size
    if len(records) != expected:
        raise ValueError(f"Info record count {len(records)} != manifest count {expected}")
    return records


def annotations(record):
    if "segmentation_with_spkid" in record:
        items = record["segmentation_with_spkid"]
        kind = "source_word_intervals"
    else:
        meta = record.get("Meta") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        items = []
        for turn in meta.get("segmentation", []):
            speaker, sep, body = turn.get("transcription", "").partition(":")
            inline_speakers = set(
                re.findall(r"<SPK_CHANGE>(.*?)</SPK_CHANGE>", turn.get("original_transcription_with_spk_change", ""))
            )
            if sep and speaker.strip() in {"A", "B"} and inline_speakers <= {speaker.strip()}:
                items.append([*turn["time"], body, speaker.strip()])
            else:
                raise ValueError("A Fisher turn could not be attributed to one source speaker")
        kind = "source_turn_intervals"

    result = []
    for lo, hi, text, speaker in items:
        lo, hi = float(lo), float(hi)
        text = " ".join(text.split())
        if text and speaker and math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi:
            result.append((lo, hi, text, str(speaker)))
    return sorted(result, key=lambda x: (x[0], x[1])), kind


def windows(items, duration):
    """Choose cuts at gaps in all speakers' labels; never split a word/turn."""
    pend = []
    end = 0.0
    for item in items:
        start, stop, *_ = item
        # Drop annotations beyond the waveform; the caller records the count.
        if stop > duration:
            continue
        if pend and start >= end and (end - pend[0][0] >= 15 or stop - pend[0][0] > 20):
            yield pend
            pend, end = [], 0.0
        pend.append(item)
        end = max(end, stop)
    if pend:
        yield pend


def make_row(items, kind, start, end, source, split, record, chunk, index, audio):
    speakers = {}
    labels = []
    segments = defaultdict(list)
    turns = []
    previous = None
    for lo, hi, text, source_speaker in items:
        speaker = speakers.setdefault(source_speaker, f"speaker{len(speakers) + 1}")
        local_lo, local_hi = round(lo - start, 6), round(hi - start, 6)
        if speaker != previous:
            labels.append(speaker)
        labels.append(text)
        previous = speaker
        turns.append({"start": local_lo, "end": local_hi, "speaker": speaker, "text": text})
        segments[speaker].append((local_lo, local_hi))
    active = []
    for speaker, intervals in segments.items():
        merged = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        active.extend({"start": lo, "end": hi, "speaker": speaker} for lo, hi in merged)
    recording = Path(record["filename"].replace("\\", "/")).stem
    return {
        "id": audio.stem,
        "recording_id": f"{source.removesuffix('_train').removesuffix('_dev')}:{recording}",
        "source_recording": record["filename"],
        "source_chunk": chunk,
        "source_record_index": index,
        "split": split,
        "data_source": "moe_" + source,
        "language": "English",
        "language_code": "en",
        "audio_path": str(audio),
        "prompt": PROMPT,
        "text": " ".join(item[2] for item in items),
        "gt_output": " ".join(labels),
        "duration_s": round(end - start, 6),
        "evaluation_duration_s": round(end - start, 6),
        "source_start_s": round(start, 6),
        "source_end_s": round(end, 6),
        "speaker_count": len(speakers),
        "speaker_map": [{"source": key, "local": val} for key, val in speakers.items()],
        "annotation_kind": kind,
        "reference_segments": sorted(active, key=lambda x: (x["start"], x["end"], x["speaker"])),
        "reference_words": turns if kind == "source_word_intervals" else None,
        "reference_turns": turns if kind == "source_turn_intervals" else None,
    }


def export_record(wav, record, source, split, chunk, index, out):
    items, kind = annotations(record)
    duration = wav.frames / wav.samplerate
    stats = {"input_spans": len(items), "out_of_waveform": sum(x[1] > duration for x in items), "skipped_spans": 0}
    rows = []
    for window in windows(items, duration):
        # Quantize outwards so the crop contains all labels.
        first = max(0, math.floor(window[0][0] * wav.samplerate))
        last = min(wav.frames, math.ceil(max(item[1] for item in window) * wav.samplerate))
        start, end = first / wav.samplerate, last / wav.samplerate
        if not 5 <= end - start <= 30:
            stats["skipped_spans"] += len(window)
            continue
        wav.seek(first)
        wave = wav.read(last - first, dtype="float32", always_2d=True).mean(axis=1)
        if wav.samplerate != 16000:
            gcd = math.gcd(wav.samplerate, 16000)
            wave = resample_poly(wave, 16000 // gcd, wav.samplerate // gcd)
        audio = out / "audio" / f"{source}_{chunk}_{index}_{first}_{last}.wav"
        sf.write(audio, wave, 16000, subtype="PCM_16")
        rows.append(make_row(window, kind, start, end, source, split, record, chunk, index, audio))
    return rows, stats


def export_chunk(source, path, split, entry, token, out, record_limit):
    chunk = entry["name"]
    base = f"{ROOT}/{path}/chunks/{chunk}"
    with request(base + ".info", token) as response:
        records = read_info(response.read(), entry["count"])
    audio_url = base + ".audio"
    with request(audio_url, token, (0, 8)) as response:
        if response.read() != b"Audio\x01\x00\x00\x00":
            raise ValueError(f"Unexpected Audio chunk header: {chunk}")
    rows, audit, seen, offset = [], [], set(), 9
    for _ in range(min(record_limit, len(records)) if record_limit else len(records)):
        with request(audio_url, token, (offset, offset + 7)) as response:
            index, size = struct.unpack("<II", response.read())
        offset += 8
        if index not in records or index in seen or not size:
            raise ValueError(f"Invalid audio index in {chunk}: {index}")
        seen.add(index)
        with tempfile.TemporaryFile() as tmp, request(audio_url, token, (offset, offset + size - 1)) as response:
            while block := response.read(1024 * 1024):
                tmp.write(block)
            if tmp.tell() != size:
                raise ValueError(f"Truncated audio record in {chunk}: {index}")
            tmp.seek(0)
            with sf.SoundFile(tmp) as wav:
                subset, stats = export_record(wav, records[index], source, split, chunk, index, out)
        rows.extend(subset)
        audit.append({"source": source, "chunk": chunk, "index": index, "rows": len(subset), **stats})
        offset += size
    return split, rows, audit


def recording_group(row):
    """Keep parts of an AMI meeting series together across splits."""
    identifier = row["recording_id"]
    if identifier.startswith("ami:"):
        match = re.search(r"[A-Z]{2}\d{4}", identifier, flags=re.IGNORECASE)
        if match:
            return "ami:" + match[0].upper()
    return identifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks-per-source", type=int, default=1, help="0 = all chunks")
    parser.add_argument("--records-per-chunk", type=int, default=2, help="0 = all records")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.chunks_per_source, args.records_per_chunk) < 0:
        parser.error("limits must be nonnegative")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=False)
    token = subprocess.check_output(
        [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://storage.azure.com",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ],
        text=True,
    ).strip()
    calls = []
    for source in args.sources:
        path, split = SOURCES[source]
        with request(f"{ROOT}/{path}/chunks/file_set.json", token) as response:
            manifest = json.load(response)["fileInfo"]
        for entry in manifest[: args.chunks_per_source or None]:
            calls.append((source, path, split, entry, token, out, args.records_per_chunk))
    groups, streams, audits = defaultdict(set), {}, []
    with ExitStack() as stack, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        # Bound completed results in memory even if a large chunk is slow.
        for offset in range(0, len(calls), args.workers):
            for split, rows, audit in pool.map(lambda args: export_chunk(*args), calls[offset : offset + args.workers]):
                audits.extend(audit)
                if rows and split not in streams:
                    streams[split] = stack.enter_context((out / f"{split}.jsonl.incomplete").open("w"))
                for row in rows:
                    groups[split].add(recording_group(row))
                    streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                print(audit[0]["source"], audit[0]["chunk"], len(rows), flush=True)

    if groups["train"] & groups["validation"]:
        raise ValueError("Train and validation share recording groups")
    for split in streams:
        data = out / f"{split}.jsonl"
        (out / f"{split}.jsonl.incomplete").rename(data)
        package = "train_data" if split == "train" else "val_data"
        conf = {
            "dataset_name": "jsonl",
            "jsonl_schema": "diarization",
            "jsonl_paths": str(data),
            "post_process": {"verl_format": {"prompt_key": "prompt", "extra_keys": EXTRA}},
        }
        if split != "train":
            conf["val_batch_size"] = -1
        with (out / f"{split}.yaml").open("w") as f:
            f.write(f"# @package data.{package}\n")
            yaml.safe_dump([conf], f, sort_keys=False)
    (out / "preparation.json").write_text(json.dumps(audits, indent=2) + "\n")


if __name__ == "__main__":
    main()

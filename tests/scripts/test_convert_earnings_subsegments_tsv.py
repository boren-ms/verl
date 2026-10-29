import csv
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "convert_earnings_subsegments_tsv",
    Path(__file__).parents[2] / "scripts/convert_earnings_subsegments_tsv.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
convert = module.convert


FIELDS = [
    "Audio",
    "Transcription",
    "Duration_Seconds",
    "Original_Audio_URL",
    "Start_Seconds",
    "End_Seconds",
    "Parent_Audio",
    "Part_Start_Seconds",
    "Part_End_Seconds",
]


def _write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_convert_normalizes_paths_and_preserves_tagged_text(tmp_path):
    source = tmp_path / "results_subsegments.tsv"
    output = tmp_path / "train.jsonl"
    _write_tsv(
        source,
        [
            {
                "Audio": r"audio\event_1_seg0.wav",
                "Transcription": "Hello <NE> Contoso </NE> .",
                "Duration_Seconds": "2.5",
                "Original_Audio_URL": "https://example.test/event_1.wav",
                "Start_Seconds": "1",
                "End_Seconds": "3.5",
                "Parent_Audio": r"..\segmented\audio\event_1_seg0.wav",
                "Part_Start_Seconds": "0",
                "Part_End_Seconds": "2.5",
            }
        ],
    )

    summary = convert(source, output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert summary["records"] == 1
    assert summary["tagged_rows"] == 1
    assert record["id"] == "event_1_seg0"
    assert record["audio_path"] == "audio/event_1_seg0.wav"
    assert record["display_ne_tagged"] == "Hello <NE> Contoso </NE> ."
    assert record["parent_audio"] == "../segmented/audio/event_1_seg0.wav"
    assert record["audio_duration_seconds"] == 2.5


def test_convert_rejects_duplicate_audio(tmp_path):
    source = tmp_path / "results_subsegments.tsv"
    output = tmp_path / "train.jsonl"
    row = {
        "Audio": r"audio\event_1_seg0.wav",
        "Transcription": "Hello.",
        "Duration_Seconds": "2.5",
        "Original_Audio_URL": "https://example.test/event_1.wav",
        "Start_Seconds": "1",
        "End_Seconds": "3.5",
        "Parent_Audio": r"..\segmented\audio\event_1_seg0.wav",
        "Part_Start_Seconds": "0",
        "Part_End_Seconds": "2.5",
    }
    _write_tsv(source, [row, row])

    with pytest.raises(ValueError, match="Duplicate audio_path"):
        convert(source, output)

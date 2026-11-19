import json

import pytest
from recipe.phimm.data.speaker_jsonl import format_speaker_jsonl, is_speaker_jsonl


def test_segment_metadata_is_optional_and_independent():
    row = {
        "reference_turns": [
            {"text": "Speaker unknown.", "start": 0.2, "end": 0.5},
            {"text": "Timing unknown.", "speaker": "A"},
            {"text": "Both unknown."},
        ],
    }
    output = format_speaker_jsonl(row)
    assert list(map(json.loads, output.splitlines())) == [
        {"text": "Speaker unknown.", "start": 20, "end": 50},
        {"spk": 1, "text": "Timing unknown."},
        {"text": "Both unknown."},
    ]
    assert is_speaker_jsonl(output)
    assert json.loads(format_speaker_jsonl({"text": "Plain reference."})) == {"text": "Plain reference."}


def test_segment_text_keeps_ambiguous_words_separate():
    row = {
        "reference_segments": [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "B", "start": 0.5, "end": 1.5},
        ],
        "reference_words": [
            {"speaker": "A", "start": 0.0, "end": 0.3, "text": "One"},
            {"start": 0.6, "end": 0.8, "text": "Overlap"},
            {"speaker": "B", "start": 1.1, "end": 1.3, "text": "Two"},
        ],
    }
    assert list(map(json.loads, format_speaker_jsonl(row).splitlines())) == [
        {"spk": 1, "text": "One", "start": 0, "end": 100},
        {"spk": 2, "text": "Two", "start": 50, "end": 150},
        {"text": "Overlap", "start": 60, "end": 80},
    ]


def test_off_grid_boundaries_round_to_ticks_and_short_final_segment_keeps_one_tick():
    output = format_speaker_jsonl(
        {
            "duration_s": 0.216,
            "reference_turns": [
                {"text": "First.", "start": 0.123, "end": 0.176},
                {"text": "Last.", "start": 0.2151, "end": 0.2155},
            ],
        }
    )
    assert list(map(json.loads, output.splitlines())) == [
        {"text": "First.", "start": 12, "end": 18},
        {"text": "Last.", "start": 21, "end": 22},
    ]


@pytest.mark.parametrize(
    "target",
    [
        '{"spk":0,"text":"Hi"}',
        '{"spk":true,"text":"Hi"}',
        '{"text":"Hi","start":0,"end":0}',
        '{"text":"Hi","start":0.0,"end":1}',
        '{"text":"Hi","start":0,"end":true}',
        '{"text":"Hi","start":0}',
        '{"text":"Hi","range":[0,1]}',
        '{"text":"Hi","text":"Hello"}',
        '{"spk":1,"start":0,"end":1}',
    ],
)
def test_invalid_segment_targets(target):
    assert not is_speaker_jsonl(target)

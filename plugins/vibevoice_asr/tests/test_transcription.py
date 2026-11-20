import json

import pytest
from vibevoice_asr.transcription import (
    close_truncated_json,
    collapse_audio_tokens,
    parse_transcription,
    transcription_text,
)


def test_official_field_variants_and_repeated_speaker():
    segments = parse_transcription(
        json.dumps(
            [
                {"Start": 0, "End": 1, "Speaker": 0, "Content": "Hello"},
                {"Start time": 1, "End time": 2, "Speaker ID": 1, "Content": "there"},
                {"Start": 2, "End": 3, "Speaker": 0, "Content": "again"},
                {"Start": 3, "End": 4, "Speaker": 0, "Content": "[Silence]"},
            ]
        )
    )
    assert [s["speaker"] for s in segments] == ["0", "1", "0", "0"]
    assert transcription_text(segments) == "Hello there again"


@pytest.mark.parametrize(
    "raw",
    [
        '[{"Start":0,"End":1,"Speaker":0,"Content":"truncated"}',
        '[{"Start":2,"End":1,"Speaker":0,"Content":"backwards"}]',
        '[{"Start":0,"End":1,"Content":"no identity"}]',
        '[{"Start":0,"End":NaN,"Speaker":0,"Content":"bad time"}]',
        '{"Start":0,"End":1,"Speaker":0,"Content":"wrong shape"}',
    ],
)
def test_invalid_generation_is_not_silently_scored(raw):
    with pytest.raises(ValueError):
        parse_transcription(raw)


def test_fenced_json_and_empty_audio():
    assert parse_transcription("```json\n[]\n```") == []


def test_official_generated_assistant_role_wrapper():
    raw = 'assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hello."}]\n'
    assert transcription_text(parse_transcription(raw)) == "Hello."
    with pytest.raises(ValueError):
        parse_transcription('Here is your result:\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hello."}]')


def test_official_acoustic_events_have_no_speaker():
    segments = parse_transcription(
        '[{"Start":0,"End":1,"Content":"[Silence]"},'
        '{"Start":1,"End":2,"Speaker":0,"Content":"Hello."},'
        '{"Start":2,"End":3,"Content":"[Human Sounds]"},'
        '{"Start":3,"End":4,"Content":"[Sigh]"}]'
    )
    assert [s["speaker"] for s in segments] == [None, "0", None, None]
    assert transcription_text(segments) == "Hello."


def test_rescore_rejects_missing_or_mismatched_samples(tmp_path):
    from vibevoice_asr.decode import read_generations

    rows = [{"id": "a", "text": "hello", "audio_path": "a.wav"}]
    path = tmp_path / "details.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="exactly once"):
        read_generations(path, rows)
    path.write_text(json.dumps({"id": "a", "reference": "wrong", "audio_path": "a.wav"}))
    with pytest.raises(ValueError, match="does not match"):
        read_generations(path, rows)


def test_truncated_wer_keeps_all_repetitions_and_partial_word():
    raw = 'assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"I know, I know, I kn'
    with pytest.raises(ValueError):
        parse_transcription(raw)
    fixed = close_truncated_json(raw)
    assert transcription_text(parse_transcription(fixed)) == "I know, I know, I kn"
    assert fixed.endswith('I know, I know, I kn"}]')
    with pytest.raises(ValueError, match="escape"):
        close_truncated_json(raw + "\\")


def test_wer_can_flag_inverted_excluded_event_without_changing_times():
    raw = '[{"Start":16.0,"End":15.97,"Content":"[Silence]"}]'
    with pytest.raises(ValueError, match="time"):
        parse_transcription(raw)
    segments = parse_transcription(raw, allow_invalid_event_times=True)
    assert segments[0]["start_s"] == 16.0 and segments[0]["end_s"] == 15.97
    assert segments[0]["timing_valid"] is False
    assert transcription_text(segments) == ""


def test_placeholder_expansion_round_trip_keeps_prompt_boundaries():
    original = [10, 11, 100, 102, 102, 102, 101, 198, 20, 21]
    collapsed = collapse_audio_tokens(original, start_id=100, end_id=101, audio_id=200)
    assert collapsed == [10, 11, 200, 20, 21]
    expanded = []
    for token in collapsed:
        expanded.extend([100, 102, 102, 102, 101, 198] if token == 200 else [token])
    assert expanded == original


def test_unexpected_audio_prompt_layout_fails_closed():
    with pytest.raises(ValueError):
        collapse_audio_tokens([10, 100, 102, 101, 20], start_id=100, end_id=101, audio_id=200)

import json

import pytest
from vibevoice_asr.prepare_rl_data import prepare_davidai_row, prepare_row, validate_splits
from vibevoice_asr.reward import diarization_reward, transcript_text


def response(*segments):
    return "assistant\n" + json.dumps(
        [{"Start": start, "End": end, "Speaker": speaker, "Content": text} for start, end, speaker, text in segments]
    )


@pytest.fixture
def reference():
    return {
        "evaluation_duration_s": 4.0,
        "reference_segments": [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 2.0, "end": 4.0, "speaker": "B"},
        ],
        "reference_words": [
            {"start": 0.0, "end": 2.0, "speaker": "A", "text": "hello"},
            {"start": 2.0, "end": 4.0, "speaker": "B", "text": "world"},
        ],
    }


def test_native_response_uses_seconds_and_permutation_invariant_speakers(reference):
    raw = response((0, 2, 9, "hello"), (2, 4, 3, "world"))
    result = diarization_reward([raw], "hello world", extra_info=reference)
    assert result["score"] == 1
    assert result["der_n_ref"] == 4
    assert result["der"] == result["wer"] == result["wder"] == 0
    assert transcript_text([raw]) == "hello world"


def test_correct_text_with_merged_speakers_loses_diarization_credit(reference):
    raw = response((0, 4, "one", "hello world"))
    result = diarization_reward(raw, "hello world", extra_info=reference)
    assert result["wer"] == 0
    assert result["der"] == 0.5
    assert result["score"] == 0.75


@pytest.mark.parametrize("raw", ["not json", '[{"Content":"hello"}]', "assistant\n[", response((0, 2, None, "hello"))])
def test_invalid_predictions_count_as_misses_and_never_earn_reward(reference, raw):
    result = diarization_reward(raw, "hello world", extra_info=reference)
    assert result["score"] == result["n_json_valid"] == 0
    assert result["der_n_err"] == result["der_n_ref"] == 4
    assert result["n_ref"] == 2
    assert result["n_err"] == 2
    assert result["invalid_output"] == 1


def test_bounded_reward_does_not_clip_reported_der(reference):
    raw = response(*[(0, 4, speaker, "hello world") for speaker in range(5)])
    result = diarization_reward(raw, "hello world", extra_info=reference)
    assert result["der"] > 1
    assert result["der_n_err"] > result["der_n_ref"]
    assert 0 <= result["score"] <= 1


def test_bad_references_raise_even_for_invalid_model_output(reference):
    reference["reference_segments"][0]["end"] = -1
    with pytest.raises(ValueError):
        diarization_reward("invalid", "hello world", extra_info=reference)


def test_disjoint_split_check_rejects_reproduction_test_meetings():
    with pytest.raises(ValueError, match="overlap"):
        validate_splits([{"recording_id": "EN2002a"}], [{"recording_id": "ES2003a"}])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits([{"recording_id": "ES2003a"}], [{"recording_id": "ES2003a"}])


def test_dev_sdm_uses_same_clock_and_reference_without_target_prompt(reference):
    row = {
        **reference,
        "id": "ES2003a_1",
        "recording_id": "ES2003a",
        "text": "hello world",
        "audio_path": "ihm.wav",
        "source_start_s": 5.0,
        "source_end_s": 9.0,
    }
    mapped = prepare_row(row, "ami_dev_sdm", sdm_map={"ES2003a": "ES2003a.Array1-01.wav"})
    assert mapped["audio_path"] == "ES2003a.Array1-01.wav#5.0:9.0"
    assert mapped["reference_segments"] == row["reference_segments"]
    assert mapped["evaluation_duration_s"] == 4
    assert mapped["prompt"] == "<audio>"


def test_davidai_validation_restores_complete_mixed_recording(reference):
    row = {
        **reference,
        "id": "davidai-recording",
        "audio_path": ["mixed.wav#0.0:1.0", "mixed.wav#2.0:4.0"],
        "svad_bounds": [[0, 1], [2, 4]],
        "svad_method": "smartvad",
        "text": "hello world",
    }
    mapped = prepare_davidai_row(row)
    assert mapped["audio_path"] == "mixed.wav"
    assert "svad_bounds" not in mapped
    assert mapped["data_source"] == "davidai_test"
    assert mapped["validation_role"] == "evaluation_only_no_tuning"
    assert mapped["reference_segments"] == row["reference_segments"]
    assert mapped["reference_words"] == row["reference_words"]
    assert mapped["evaluation_duration_s"] == 4
    assert mapped["prompt"] == "<audio>"
    assert len(row["audio_path"]) == 2  # Do not mutate the source manifest.


def test_davidai_rejects_mixed_parent_recordings(reference):
    with pytest.raises(ValueError, match="one parent"):
        prepare_davidai_row({**reference, "audio_path": ["a.wav#0:1", "b.wav#1:2"]})


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1, 0, 902])
def test_davidai_never_silently_truncates_reference_coverage(reference, duration):
    with pytest.raises(ValueError, match="duration"):
        prepare_davidai_row({**reference, "audio_path": "a.wav", "evaluation_duration_s": duration})

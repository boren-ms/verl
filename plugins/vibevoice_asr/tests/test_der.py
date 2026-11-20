import pytest
from vibevoice_asr.der import hypothesis_rttm


def prediction(recording="meeting", **changes):
    row = {
        "id": recording,
        "truncated": False,
        "segments": [{"start_s": 1.0, "end_s": 3.0, "speaker": "2", "text": "hello"}],
    }
    row.update(changes)
    return row


def test_exact_coverage_required_before_meeteval_intersection():
    with pytest.raises(ValueError, match="every reference"):
        hypothesis_rttm([prediction()], {"meeting", "missing"})
    with pytest.raises(ValueError, match="every reference"):
        hypothesis_rttm([prediction(), prediction()], {"meeting"})
    with pytest.raises(ValueError, match="Truncated"):
        hypothesis_rttm([prediction(truncated=True)], {"meeting"})


def test_unintelligible_speech_is_scored_with_real_speaker():
    row = prediction()
    row["segments"][0]["text"] = "[Unintelligible Speech]"
    assert "1.000000 2.000000 <NA> <NA> 2" in hypothesis_rttm([row], {"meeting"})
    row["segments"][0]["speaker"] = None
    with pytest.raises(ValueError, match="speech speaker"):
        hypothesis_rttm([row], {"meeting"})


def test_overlapping_turns_preserved_and_acoustic_events_excluded():
    row = prediction()
    row["segments"] += [
        {"start_s": 2.0, "end_s": 4.0, "speaker": "1", "text": "yes"},
        {"start_s": 4.0, "end_s": 5.0, "speaker": None, "text": "[Noise]"},
    ]
    result = hypothesis_rttm([row], {"meeting"})
    assert len(result.splitlines()) == 2
    assert "2.000000 2.000000 <NA> <NA> 1" in result


def test_abstention_keeps_reference_recording_and_never_invents_speaker():
    row = prediction()
    row["segments"].append({"start_s": 4.0, "end_s": 5.0, "speaker": None, "text": "[Speech]"})
    with pytest.raises(ValueError, match="speech speaker"):
        hypothesis_rttm([row], {"meeting"})
    rttm = hypothesis_rttm([row], {"meeting"}, allow_unassigned_speech=True)
    assert len(rttm.splitlines()) == 1
    assert "1.000000 2.000000" in rttm
    assert "4.000000" not in rttm


def test_truncation_requires_explicit_full_timeline_scoring_opt_in():
    row = prediction(truncated=True)
    with pytest.raises(ValueError, match="Truncated"):
        hypothesis_rttm([row], {"meeting"})
    assert "meeting 1 1.000000 2.000000" in hypothesis_rttm([row], {"meeting"}, allow_truncated=True)

"""Pre-segmented entity evaluation must preserve complete parent references."""

from copy import deepcopy

import pytest
from recipe.phimm.data.inhouse_entity import SOURCE_ROOT, normalize_domain

DOMAIN = "Conversation_DomainSet_DTEST_Gaming_Entity_FY24Q1_en-US_DTEST"


def examples():
    original = [
        {
            "UUID": "u",
            "WavPath": "/old/wav/u_0.wav",
            "Transcription": "hello <NE> World </NE>",
            "DisplayTranscription": "Hello <NE> World </NE>.",
        }
    ]
    clips = [
        {
            "UUID": "u",
            "WavPath": f"{SOURCE_ROOT}/{DOMAIN}/vad_segments/u_0_segment_0000_0_16000.wav",
            "Transcription": original[0]["Transcription"],
        },
        {
            "UUID": "u",
            "WavPath": f"{SOURCE_ROOT}/{DOMAIN}/vad_segments/u_0_segment_0001_16000_32000.wav",
            "Transcription": "",
        },
    ]
    return original, clips


def test_parent_reference_and_source_clips_preserved():
    original, clips = examples()
    before = deepcopy((original, clips))
    result = normalize_domain(original, list(reversed(clips)), domain=DOMAIN)
    assert len(result) == len(original)
    row = result[0]
    assert row["audio_path"] == [clip["WavPath"] for clip in clips]
    assert row["entity_transcription"] == original[0]["Transcription"]
    assert row["text"] == original[0]["DisplayTranscription"]
    assert row["n_segments"] == 2
    assert row["data_source"] == "eer_Conversation_Gaming"
    assert (original, clips) == before


def test_vad_display_reference_matches_original_without_timestamps():
    original, clips = examples()
    original[0]["DisplayTranscription"] = "[0 1] Hello <NE> World </NE>."
    clips[0]["Transcription"] = "Hello <NE> World </NE>."
    row = normalize_domain(original, clips, domain=DOMAIN)[0]
    assert row["entity_transcription"] == original[0]["Transcription"]
    assert row["text"] == original[0]["DisplayTranscription"]


@pytest.mark.parametrize(
    "corruption", ["missing_parent", "missing_segment", "duplicate_segment", "changed_ref", "lost_ref"]
)
def test_incomplete_or_changed_sources_rejected(corruption):
    original, clips = examples()
    if corruption == "missing_parent":
        clips[1]["UUID"] = "other"
    elif corruption == "missing_segment":
        clips = clips[1:]
    elif corruption == "duplicate_segment":
        clips.append(deepcopy(clips[1]))
    elif corruption == "changed_ref":
        clips[0]["Transcription"] = "changed <NE> reference </NE>"
    else:
        clips[0]["Transcription"] = ""
    with pytest.raises(ValueError):
        normalize_domain(original, clips, domain=DOMAIN)

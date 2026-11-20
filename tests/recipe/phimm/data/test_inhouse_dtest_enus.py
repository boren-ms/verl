"""Workbook scope and original references must survive segment adaptation."""

from copy import deepcopy

import pytest
import yaml
from recipe.phimm.data.inhouse_dtest_enus import EER_DATASETS as DATASETS
from recipe.phimm.data.inhouse_dtest_enus import REFERENCE_ROOT, normalize_recordings


def examples():
    domain = DATASETS[2]
    parent = dict(
        UUID="u",
        WavPath="/old/u_0.wav",
        Transcription="<PName> Alex </PName>",
        DisplayTranscription="[0 1] <PName> Alex </PName>.",
    )
    audio = f"{REFERENCE_ROOT}/{domain}/wav/u_0.wav"
    clips = [
        dict(
            parent,
            WavPath=f"{audio}#{i * 20}:{(i + 1) * 20}",
            parent_audio_path=audio,
            seg_index=str(i),
            n_segments="2",
            seg_start=str(i * 20),
            seg_end=str((i + 1) * 20),
        )
        for i in range(2)
    ]
    return domain, [parent], clips


def test_existing_clips_and_full_display_entities_preserved():
    domain, original, clips = examples()
    before = deepcopy((original, clips))
    row = normalize_recordings(original, list(reversed(clips)), domain=domain, source_manifest="source")[0]
    assert row["text"] == original[0]["DisplayTranscription"]
    assert row["entity_transcription"] == original[0]["Transcription"]
    assert row["audio_path"] == [clip["WavPath"] for clip in clips]
    assert row["data_source"] == "shiproom"
    assert (original, clips) == before


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reference", "bounds", "parent", "unselected"])
def test_incomplete_or_changed_inputs_fail(corruption):
    domain, original, clips = examples()
    if corruption == "missing":
        clips.pop()
    elif corruption == "duplicate":
        clips.append(deepcopy(clips[0]))
    elif corruption == "reference":
        clips[0]["DisplayTranscription"] = "different entity reference"
    elif corruption == "bounds":
        clips[0]["WavPath"] += "1"
    elif corruption == "parent":
        clips[0]["parent_audio_path"] = "another recording"
    else:
        domain = "unselected_dataset"
    with pytest.raises(ValueError):
        normalize_recordings(original, clips, domain=domain, source_manifest="source")


def test_yaml_explicitly_selects_exact_workbook_datasets():
    from pathlib import Path

    from recipe.phimm.data import inhouse_dtest_enus

    path = Path(inhouse_dtest_enus.__file__).parent.parent / "config/data/val_data/inhouse/eer.yaml"
    specs = yaml.safe_load(path.read_text())
    assert len(specs) == 5
    assert [Path(spec["jsonl_paths"]).stem for spec in specs] == list(DATASETS)
    assert not any("*" in spec["jsonl_paths"] for spec in specs)


def test_ter_suite_reuses_meetings_but_excludes_banking_and_medical():
    from recipe.phimm.data.inhouse_dtest_enus import SUITES

    assert len(SUITES["eer"]) == len(SUITES["ter"]) == 5
    assert len(set(SUITES["eer"]) | set(SUITES["ter"])) == 7
    domain, original, clips = examples()
    row = normalize_recordings(original, clips, domain=domain, source_manifest="source", suite="ter")[0]
    assert row["data_source"] == "shiproom"
    with pytest.raises(ValueError, match="outside workbook"):
        normalize_recordings(original, clips, domain=DATASETS[0], source_manifest="source", suite="ter")


def test_yaml_selects_exact_datasets_and_shared_english_task():
    from pathlib import Path

    from recipe.phimm.data import inhouse_dtest_enus

    root = Path(inhouse_dtest_enus.__file__).parent.parent / "config/data/val_data/inhouse"
    for suite, datasets in inhouse_dtest_enus.SUITES.items():
        specs = yaml.safe_load((root / f"{suite}.yaml").read_text())
        assert len(specs) == 5
        assert [Path(spec["jsonl_paths"]).stem for spec in specs] == list(datasets)
        sources = []
        for spec in specs:
            assert spec["add_task_info"] == {"style": "display", "language": "English", "lang_hint": True}
            assert "pre_process" not in spec
            sources.append(spec["post_process"]["add_field"]["fields"]["data_source"])
        assert sources == list(inhouse_dtest_enus.DATA_SOURCES[suite].values())
        assert len(set(sources)) == 5 and all(len(source) <= 16 for source in sources)
        preset = root.parents[3] / "config/v2611_dev/eval" / f"eval_enus_{suite}.yaml"
        routing = yaml.safe_load(preset.read_text())["_inhouse_reward_map"]
        assert list(routing) == sources


def test_statistics_do_not_count_full_reference_for_each_segment():
    from recipe.phimm.data.inhouse_dtest_enus import recording_statistics

    domain, original, clips = examples()
    rows = normalize_recordings(original, clips, domain=domain, source_manifest="source")
    stats = recording_statistics(rows)
    assert stats["display_words"] == 1  # Only Alex; tags, times and punctuation-only tokens are excluded.
    assert stats["segments_per_recording"] == dict(min=2, mean=2, max=2)
    assert stats["segment_seconds"] == dict(min=20, mean=20, max=20)
    assert stats["segment_word_counts"] is None


def test_single_existing_whole_recording_path_is_preserved():
    from recipe.phimm.data.inhouse_dtest_enus import recording_statistics

    domain, original, clips = examples()
    clips = clips[:1]
    clips[0]["n_segments"] = 1
    clips[0]["WavPath"] = clips[0]["parent_audio_path"]
    rows = normalize_recordings(original, clips, domain=domain, source_manifest="source")
    assert rows[0]["audio_path"] == [clips[0]["WavPath"]]
    assert recording_statistics(rows, clips)["segment_seconds"]["mean"] == 20


def test_unbounded_path_in_multisegment_recording_is_rejected():
    domain, original, clips = examples()
    clips[0]["WavPath"] = clips[0]["parent_audio_path"]
    with pytest.raises(ValueError, match="single complete recording"):
        normalize_recordings(original, clips, domain=domain, source_manifest="source")

"""Short and offline multi-segment rows remain usable in the same training mix."""

import json

import pytest
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset


@pytest.mark.parametrize("adjustment", ["original", None])
def test_short_and_contiguous_mix_preserves_audio_metadata_and_segments(tmp_path, adjustment):
    rows = [
        {
            "audio_path": "short.wav",
            "audio_adjustment": adjustment,
            "audio_duration_seconds": 1.5,
            "text": "short text",
        },
        {
            "audio_path": ["first.wav", "second.wav"],
            "audio_adjustment": ["first adjustment", "second adjustment"],
            "audio_duration_seconds": [2.0, 3.0],
            "text": "first text second text",
        },
    ]
    sources = []
    for index, row in enumerate(rows):
        path = tmp_path / f"source_{index}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        sources.append(
            {
                "dataset_name": "jsonl",
                "jsonl_paths": str(path),
                "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
                "post_process": {"verl_format": {"prompt_key": "prompt"}},
            }
        )
    config = OmegaConf.create({"train_files": "train", "train_data": sources})
    data = ASRDataset("train", None, config)

    assert len(data) == 2
    assert [segment["audio_path"] for segment in data[0]["asr_segments"]] == ["short.wav"]
    assert [segment["audio_path"] for segment in data[1]["asr_segments"]] == ["first.wav", "second.wav"]
    assert data.ds[0]["audio_adjustment"] == ([adjustment] if adjustment is not None else None)
    assert data.ds[1]["audio_adjustment"] == rows[1]["audio_adjustment"]
    assert data.ds[0]["audio_duration_seconds"] == [1.5]
    assert data.ds[1]["audio_duration_seconds"] == [2.0, 3.0]
    assert [data[i]["reward_model"]["ground_truth"] for i in range(2)] == [row["text"] for row in rows]


@pytest.mark.parametrize("approval", ["approved", None])
def test_short_and_merged_fl_preserve_provenance_lists(tmp_path, approval):
    rows = [
        {"audio_path": "short.wav", "text": "short", "approval": approval, "source_row": 12},
        {
            "audio_path": ["first.wav", "second.wav"],
            "text": "first second",
            "approval": ["approved", "reviewed"],
            "source_row": ["12", "15"],
        },
    ]
    sources = []
    for index, row in enumerate(rows):
        path = tmp_path / f"fl_{index}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        sources.append(
            {
                "dataset_name": "jsonl",
                "jsonl_paths": str(path),
                "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
                "post_process": {"verl_format": {"prompt_key": "prompt", "extra_keys": ["approval", "source_row"]}},
            }
        )
    config = OmegaConf.create({"train_files": "train", "version": 2609, "train_data": sources})
    data = ASRDataset("train", None, config)

    assert len(data) == 2
    assert data.ds[0]["extra_info"]["approval"] == ([approval] if approval is not None else None)
    assert data.ds[0]["extra_info"]["source_row"] == ["12"]
    assert data.ds[1]["extra_info"]["approval"] == rows[1]["approval"]
    assert data.ds[1]["extra_info"]["source_row"] == rows[1]["source_row"]
    assert [data[i]["extra_info"]["language"] for i in range(2)] == ["English", "English"]
    assert [s["audio_path"] for s in data[1]["asr_segments"]] == rows[1]["audio_path"]
    assert [data[i]["reward_model"]["ground_truth"] for i in range(2)] == [row["text"] for row in rows]

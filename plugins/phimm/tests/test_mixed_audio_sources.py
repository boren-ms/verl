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
    config = OmegaConf.create({"train_files": "train", "version": 2609, "train_data": sources})
    data = ASRDataset("train", None, config)

    assert len(data) == 2
    assert [segment["audio_path"] for segment in data[0]["asr_segments"]] == ["short.wav"]
    assert [segment["audio_path"] for segment in data[1]["asr_segments"]] == ["first.wav", "second.wav"]
    assert data.ds[0]["audio_adjustment"] == ([adjustment] if adjustment is not None else None)
    assert data.ds[1]["audio_adjustment"] == rows[1]["audio_adjustment"]
    assert data.ds[0]["audio_duration_seconds"] == [1.5]
    assert data.ds[1]["audio_duration_seconds"] == [2.0, 3.0]
    assert [data[i]["reward_model"]["ground_truth"] for i in range(2)] == [row["text"] for row in rows]

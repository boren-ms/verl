from datasets import Dataset

from recipe.phimm.data import dataset as dataset_module


def test_process_ds_cache_audio_runs_after_rename_fields(monkeypatch):
    ds = Dataset.from_dict(
        {
            "WavPath": [
                "az://orngwus2cresco/data/Evaluation/sample.wav",
                "az://orngwus2cresco/data/Evaluation/second.wav",
            ],
            "DisplayTranscription": ["hello", "world"],
        }
    )
    submissions = []

    def fake_submit(dataset, fields, max_workers):
        submissions.append((dataset, fields, max_workers))

    monkeypatch.setattr(
        dataset_module,
        "submit_audio_cache_dataset",
        fake_submit,
    )

    result = dataset_module.process_ds(
        ds,
        rename_fields={"mappings": {"audio_path": "WavPath", "text": "DisplayTranscription"}},
        cache_audio={},
    )

    assert result[0]["audio_path"] == "az://orngwus2cresco/data/Evaluation/sample.wav"
    assert result[0]["text"] == "hello"
    assert result[1]["audio_path"] == "az://orngwus2cresco/data/Evaluation/second.wav"
    assert result[1]["text"] == "world"
    assert submissions == [(result, ["audio_path", "audio_chunk"], 16)]


def test_cache_audio_submits_fields_without_updating_dataset(monkeypatch):
    ds = Dataset.from_dict(
        {
            "audio_path": ["remote/a.wav", ""],
            "audio_chunk": ["remote/chunk.audio:2:0", "remote/chunk.audio:2:1"],
        }
    )
    submissions = []
    monkeypatch.setattr(
        dataset_module,
        "submit_audio_cache_dataset",
        lambda dataset, fields, max_workers: submissions.append((dataset, fields, max_workers)),
    )

    result = dataset_module.cache_audio(ds, batch_size=2, max_workers=8)

    assert result is ds
    assert result["audio_path"] == ["remote/a.wav", ""]
    assert result["audio_chunk"] == ["remote/chunk.audio:2:0", "remote/chunk.audio:2:1"]
    assert submissions == [(ds, ["audio_path", "audio_chunk"], 8)]


def test_cache_audio_uses_configured_fields(monkeypatch):
    ds = Dataset.from_dict(
        {
            "audio_path": ["remote/a.wav", "remote/a.wav"],
            "audio_chunk": ["remote/chunk.audio:2:0", "remote/chunk.audio:2:1"],
        }
    )
    submissions = []
    monkeypatch.setattr(
        dataset_module,
        "submit_audio_cache_dataset",
        lambda dataset, fields, max_workers: submissions.append((dataset, fields)),
    )

    result = dataset_module.cache_audio(ds, fields=["audio_chunk"])

    assert result is ds
    assert submissions == [(ds, ["audio_chunk"])]
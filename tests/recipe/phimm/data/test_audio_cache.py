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
    cached_sources = []

    def fake_cache(source):
        cached_sources.append(source)
        return source.replace("az://orngwus2cresco/data/", "/home/test/data/")

    monkeypatch.setattr(
        dataset_module,
        "cache_audio_source",
        fake_cache,
    )

    result = dataset_module.process_ds(
        ds,
        rename_fields={"mappings": {"audio_path": "WavPath", "text": "DisplayTranscription"}},
        cache_audio={},
    )

    assert result[0]["audio_path"] == "/home/test/data/Evaluation/sample.wav"
    assert result[0]["text"] == "hello"
    assert result[1]["audio_path"] == "/home/test/data/Evaluation/second.wav"
    assert result[1]["text"] == "world"
    assert cached_sources == ds["WavPath"]


def test_cache_audio_batches_audio_path_and_chunk(monkeypatch):
    ds = Dataset.from_dict(
        {
            "audio_path": ["remote/a.wav", ""],
            "audio_chunk": ["remote/chunk.audio:2:0", "remote/chunk.audio:2:1"],
        }
    )
    monkeypatch.setattr(dataset_module, "cache_audio_source", lambda source: f"local/{source}")

    result = dataset_module.cache_audio(ds, batch_size=2)

    assert result["audio_path"] == ["local/remote/a.wav", ""]
    assert result["audio_chunk"] == ["local/remote/chunk.audio:2:0", "local/remote/chunk.audio:2:1"]


def test_cache_audio_uses_parallel_workers_and_deduplicates_sources(monkeypatch):
    ds = Dataset.from_dict(
        {
            "audio_path": ["remote/a.wav", "remote/a.wav"],
            "audio_chunk": ["remote/chunk.audio:2:0", "remote/chunk.audio:2:1"],
        }
    )
    executor_workers = []
    cached_sources = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            executor_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, sources):
            return [function(source) for source in sources]

    def fake_cache(source):
        cached_sources.append(source)
        return f"local/{source}"

    monkeypatch.setattr(dataset_module, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(dataset_module, "cache_audio_source", fake_cache)

    result = dataset_module.cache_audio(ds, batch_size=4, max_workers=8)

    assert executor_workers == [3]
    assert set(cached_sources) == {"remote/a.wav", "remote/chunk.audio:2:0", "remote/chunk.audio:2:1"}
    assert result["audio_path"] == ["local/remote/a.wav", "local/remote/a.wav"]
"""Check full-reference random segments from dataset preparation to audio reads."""

import io
import json

import datasets
import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.data.random_segment import random_segment
from recipe.phimm.utils.audio import _is_chunk_spec, get_audio_duration, load_audio


def test_random_segments_preserve_the_single_full_reference():
    ds = datasets.Dataset.from_list([{"audio_path": "A.wav", "text": "full original reference"}])
    result = random_segment(ds, segment_count=[2, 2])
    assert len(result) == 1
    paths = result[0]["audio_path"]
    assert len(paths) == 2
    assert paths[0].startswith("A.wav#0%:")
    assert paths[1].endswith(":100%")
    assert paths[0].rsplit(":", 1)[1] == paths[1].rsplit("#", 1)[1].split(":")[0]
    assert paths != ["A.wav#0%:50%", "A.wav#50%:100%"]
    assert result[0]["text"] == "full original reference"


def test_random_segment_counts_and_boundaries_are_reproducible_complete_partitions():
    ds = datasets.Dataset.from_list([{"audio_path": f"{i}.wav"} for i in range(40)])
    first = random_segment(ds, segment_count=[1, 4], seed=7)
    assert first.to_list() == random_segment(ds, segment_count=[1, 4], min_segment_fraction=0.1, seed=7).to_list()
    assert first.to_list() == random_segment(ds, segment_count=[1, 4], seed=7).to_list()
    assert first.to_list() != random_segment(ds, segment_count=[1, 4], seed=8).to_list()
    assert {len(row["audio_path"]) for row in first} == {1, 2, 3, 4}
    for row in first:
        ranges = [tuple(float(value[:-1]) for value in path.rsplit("#", 1)[1].split(":")) for path in row["audio_path"]]
        assert ranges[0][0] == 0 and ranges[-1][1] == 100
        assert all(end - start >= 10 - 1e-8 for start, end in ranges)
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:], strict=False))


def test_default_minimum_fraction_rejects_infeasible_count_range():
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav"}])
    with pytest.raises(ValueError, match="min_segment_fraction"):
        random_segment(ds, segment_count=[1, 20])


@pytest.mark.parametrize(
    "options",
    [
        {"segment_count": [0, 4]},
        {"segment_count": [1, True]},
        {"segment_count": [4, 2]},
        {"segment_count": []},
        {"segment_count": [1]},
        {"segment_count": [1, 2, 3]},
        {"segment_count": [1, 2.5]},
        {"segment_count": "1,4"},
        {"segment_count": 4},
        {"min_segment_fraction": -0.1},
        {"min_segment_fraction": 0.4},
        {"seed": 1.5},
    ],
)
def test_random_segment_rejects_invalid_settings(options):
    ds = datasets.Dataset.from_list([{"audio_path": "A.wav"}])
    with pytest.raises(ValueError):
        random_segment(ds, **({"segment_count": [1, 4]} | options))


@pytest.mark.parametrize("mode", [{"segment_count": [2, 2]}, {"segment_len": [20, 20]}])
def test_dataset_block_expands_paths_without_splitting_reference_or_loading_audio(tmp_path, mode):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"audio_path": "A.wav", "audio_length": 40, "text": "full reference", "prompt": "task<audio>"})
        + "\n"
    )
    config = OmegaConf.create(
        {
            "train_files": "train",
            "val_files": "val",
            "version": 2609,
            "train_data": [
                {
                    "dataset_name": "jsonl",
                    "jsonl_paths": str(path),
                    "post_process": {"verl_format": {"prompt_key": "prompt"}},
                }
            ],
            "val_data": [
                {
                    "dataset_name": "jsonl",
                    "jsonl_paths": str(path),
                    "post_process": {"verl_format": {"prompt_key": "prompt"}},
                }
            ],
            "random_segment": {
                **mode,
            },
        }
    )
    train = ASRDataset("train", object(), config)
    assert len(train) == 1
    row = train[0]
    assert len(row["audio_path"]) == 2
    assert row["audio_path"][0].startswith("A.wav#0%:")
    assert row["audio_path"][-1].endswith(":100%")
    assert row["reward_model"]["ground_truth"] == "full reference"
    assert [segment["audio_path"] for segment in row["asr_segments"]] == row["audio_path"]
    assert all(segment["prompt"] == row["prompt"] for segment in row["asr_segments"])
    assert all(segment["reward_model"]["ground_truth"] == "full reference" for segment in row["asr_segments"])
    assert not row["asr_validate"]
    # Global training augmentation must not change validation inputs.
    val = ASRDataset("val", object(), config)
    assert val[0]["audio_path"] == "A.wav"


def test_segmented_and_unsegmented_sources_share_a_mix(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"audio_path": "A.wav", "text": "reference", "prompt": "task<audio>"}) + "\n")
    sources = []
    for name in ("segmented", "original"):
        post = {"verl_format": {"prompt_key": "prompt"}, "add_field": {"fields": {"data_source": name}}}
        if name == "segmented":
            post["random_segment"] = {"segment_count": [2, 2]}
        sources.append({"dataset_name": "jsonl", "jsonl_paths": str(path), "post_process": post})
    config = OmegaConf.create(
        {
            "train_files": "train",
            "version": 2609,
            "train_data": sources,
            "shuffle": False,
            "train_batch_size": 1,
            "mix": [{"datasource": [{"name": name, "sample": 1} for name in ("segmented", "original")]}],
        }
    )
    train = ASRDataset("train", object(), config, max_samples=2)
    assert len(train) == 2
    assert len(train[0]["asr_segments"]) == 2
    assert len(train[1]["asr_segments"]) == 1
    assert train[1]["asr_segments"][0]["audio_path"] == "A.wav"


@pytest.mark.parametrize("mode", [{"segment_count": [4, 4]}, {"segment_len": [0.1, 0.3]}])
@pytest.mark.parametrize("source_suffix", ["", "#0.2:0.8"])
def test_percent_segments_reassemble_exact_waveform_without_gaps_or_overlap(tmp_path, source_suffix, mode):
    path = tmp_path / "A.wav"
    waveform = np.linspace(-0.9, 0.9, 16000, dtype=np.float32)
    sf.write(path, waveform, 16000, subtype="FLOAT")
    source = str(path) + source_suffix
    ds = random_segment(datasets.Dataset.from_list([{"audio_path": source}]), **mode)
    chunks = [load_audio({"audio_path": item}, min_dur=0)[0] for item in ds[0]["audio_path"]]
    original, _ = load_audio({"audio_path": source}, min_dur=0)
    np.testing.assert_array_equal(np.concatenate(chunks), original)


def test_original_percent_selector_and_existing_chunk_grammar(tmp_path):
    path = tmp_path / "A.wav"
    sf.write(path, np.arange(16000, dtype=np.float32) / 16000, 16000, subtype="FLOAT")
    first, _ = load_audio({"audio_path": f"{path}#0%:50%"}, min_dur=0)
    second, _ = load_audio({"audio_path": f"{path}#50%:100%"}, min_dur=0)
    assert len(first) == len(second) == 8000
    assert _is_chunk_spec("archive.chunk:100:2")
    with pytest.raises(ValueError, match="full reference"):
        load_audio({"audio_path": f"{path}#0%:100%", "asr_full_reference": True}, max_dur=0.1)


def test_random_partitions_support_archive_entries_and_concatenated_ranges(tmp_path):
    waveform = np.linspace(-0.9, 0.9, 16000, dtype=np.float32)
    payload = io.BytesIO()
    sf.write(payload, waveform, 16000, format="WAV", subtype="FLOAT")
    encoded = payload.getvalue()
    archive = tmp_path / "audio.zip"
    archive.write_bytes(b"header" + encoded)
    wav = tmp_path / "audio.wav"
    sf.write(wav, waveform, 16000, subtype="FLOAT")
    sources = [f"{archive}!6:{len(encoded)}", f"{wav}#0:0.5;{wav}#0.5:1"]
    for source in sources:
        ds = random_segment(datasets.Dataset.from_list([{"audio_path": source}]), segment_count=[2, 2])
        parts = [load_audio({"audio_path": path}, min_dur=0)[0] for path in ds[0]["audio_path"]]
        np.testing.assert_array_equal(np.concatenate(parts), waveform)


def test_length_mode_prefers_audio_length_without_audio_io(monkeypatch):
    import importlib

    module = importlib.import_module("recipe.phimm.data.random_segment")
    monkeypatch.setattr(module, "get_audio_duration", lambda source: pytest.fail("must use audio_length"))
    ds = datasets.Dataset.from_list(
        [{"audio_path": f"missing-{i}.wav", "audio_length": 39.0, "text": "full reference"} for i in range(40)]
    )
    result = random_segment(ds, segment_len=[10, 20], seed=7)
    assert result.to_list() == random_segment(ds, segment_len=[10, 20], seed=7).to_list()
    assert result.to_list() != random_segment(ds, segment_len=[10, 20], seed=8).to_list()
    for row in result:
        ranges = [tuple(float(value[:-1]) for value in path.rsplit("#", 1)[1].split(":")) for path in row["audio_path"]]
        lengths = [(end - start) * 0.39 for start, end in ranges]
        assert ranges[0][0] == 0 and ranges[-1][1] == 100
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:], strict=False))
        assert all(10 - 1e-8 <= length <= 20 + 1e-8 for length in lengths[:-1])
        assert 0 < lengths[-1] <= 20 + 1e-8
        assert sum(lengths) == pytest.approx(39)
        assert row["text"] == "full reference"


def test_length_mode_draws_a_different_random_length_for_each_segment():
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav", "audio_length": 200}])
    result = random_segment(ds, segment_len=[10, 20], seed=7)
    ranges = [
        tuple(float(value[:-1]) for value in path.rsplit("#", 1)[1].split(":")) for path in result[0]["audio_path"]
    ]
    lengths = [(end - start) * 2 for start, end in ranges]
    assert len(lengths) >= 10
    assert all(10 <= length <= 20 for length in lengths[:-1])
    assert len({round(length, 6) for length in lengths[:-1]}) == len(lengths) - 1
    assert 0 < lengths[-1] <= 20
    assert sum(lengths) == pytest.approx(200)


@pytest.mark.parametrize("duration", [2, 10, 15, 20, 21, 39, 40, 75, 81, 220, 221, 1000])
def test_fixed_length_mode_keeps_the_complete_recording_and_short_tail(duration):
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav", "audio_length": duration}])
    options = {"segment_len": [20, 20]}
    result = random_segment(ds, **options)
    row = result[0]
    assert len(row["audio_path"]) == (duration + 19) // 20
    ranges = [tuple(float(value[:-1]) for value in path.rsplit("#", 1)[1].split(":")) for path in row["audio_path"]]
    lengths = [(end - start) * duration / 100 for start, end in ranges]
    assert all(length == pytest.approx(20) for length in lengths[:-1])
    assert 0 < lengths[-1] <= 20 + 1e-8
    assert sum(lengths) == pytest.approx(duration)


def test_segment_modes_are_mutually_exclusive():
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav"}])
    with pytest.raises(ValueError, match="exactly one"):
        random_segment(ds, segment_count=[1, 2], segment_len=[10, 20])


@pytest.mark.parametrize("options", [{}, {"segment_len": None}, {"segment_count": None, "segment_len": None}])
def test_null_duration_defaults_to_count_mode_without_duration_io(monkeypatch, options):
    import importlib

    module = importlib.import_module("recipe.phimm.data.random_segment")
    monkeypatch.setattr(
        module, "get_audio_duration", lambda source: pytest.fail("default mode must not inspect duration")
    )
    ds = datasets.Dataset.from_list([{"audio_path": f"missing-{i}.wav", "audio_length": -1} for i in range(40)])
    result = random_segment(ds, **options)
    assert result.to_list() == random_segment(ds, segment_count=[1, 2], min_segment_fraction=0.1).to_list()
    assert {len(row["audio_path"]) for row in result} == {1, 2}


@pytest.mark.parametrize(
    "lengths",
    [
        True,
        False,
        [],
        [10],
        [10, 20, 30],
        [20, 10],
        [0, 20],
        [-1, 20],
        [True, 20],
        [10, float("nan")],
        [10, float("inf")],
        "10,20",
        20,
    ],
)
def test_length_mode_rejects_invalid_settings(lengths):
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav"}])
    with pytest.raises(ValueError, match="segment_len"):
        random_segment(ds, segment_len=lengths)


@pytest.mark.parametrize("duration", [True, 0, -1, float("nan"), float("inf"), "20"])
def test_length_mode_rejects_invalid_audio_length(duration):
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav", "audio_length": duration}])
    with pytest.raises(ValueError, match="audio_length"):
        random_segment(ds, segment_len=[10, 20])


def test_count_mode_never_inspects_duration_and_supports_twenty_segments(monkeypatch):
    import importlib

    module = importlib.import_module("recipe.phimm.data.random_segment")
    monkeypatch.setattr(
        module, "get_audio_duration", lambda source: pytest.fail("count mode must not inspect duration")
    )
    ds = datasets.Dataset.from_list([{"audio_path": f"missing-{i}.wav", "audio_length": -1} for i in range(100)])
    result = random_segment(ds, segment_count=[1, 20], min_segment_fraction=0.01)
    counts = {len(row["audio_path"]) for row in result}
    assert min(counts) == 1 and max(counts) == 20
    assert result.to_list() == random_segment(ds, segment_count=[1, 20], min_segment_fraction=0.01).to_list()
    assert random_segment(ds, enabled=False) is ds


def test_length_mode_rejects_count_only_minimum_fraction():
    ds = datasets.Dataset.from_list([{"audio_path": "missing.wav"}])
    with pytest.raises(ValueError, match="only to segment_count"):
        random_segment(ds, segment_len=[10, 20], min_segment_fraction=0.1)


@pytest.mark.parametrize("suffix", ["", "#0.2:0.8", "#0.2:0.8#0%:50%"])
def test_missing_audio_length_uses_headers_without_decoding(tmp_path, monkeypatch, suffix):
    import recipe.phimm.utils.audio as audio_module

    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(16000), 16000)
    monkeypatch.setattr(sf, "read", lambda *args, **kwargs: pytest.fail("must not decode audio"))
    monkeypatch.setattr(audio_module, "_load_chunk", lambda *args: pytest.fail("must not load a chunk"))
    ds = datasets.Dataset.from_list([{"audio_path": str(path) + suffix, "audio_length": None}])
    result = random_segment(ds, segment_len=[0.1, 0.3])
    duration = get_audio_duration(str(path) + suffix)
    for part in result[0]["audio_path"]:
        assert get_audio_duration(part) <= 0.3 + 1e-10
    assert sum(get_audio_duration(part) for part in result[0]["audio_path"]) == pytest.approx(duration)


def test_legacy_chunk_duration_reads_only_indexed_entry(tmp_path, monkeypatch):
    from recipe.phimm.data import indexed_chunk

    payload = io.BytesIO()
    sf.write(payload, np.zeros(16000), 16000, format="WAV")
    encoded = payload.getvalue()
    path = tmp_path / "audio.audio"
    path.write_bytes(b"header" + encoded)
    calls = []

    def references(source, count):
        calls.append((source, count))
        return [f"{path}!6:{len(encoded)}"]

    monkeypatch.setattr(indexed_chunk, "chunk_audio_byte_references", references)
    monkeypatch.setattr(sf, "read", lambda *args, **kwargs: pytest.fail("must not decode audio"))
    assert get_audio_duration(f"{path}:1:0") == 1.0
    assert calls == [(str(path), 1)]


@pytest.mark.parametrize("mode", [{"segment_count": [2, 2]}, {"segment_len": [1, 1]}])
@pytest.mark.parametrize("frames", [(16000, 16000), (12000, 20000), (20000, 12000)])
def test_plain_files_are_concatenated_before_segmenting(tmp_path, mode, frames):
    paths = [tmp_path / name for name in ("a.wav", "b.wav")]
    waveforms = [np.linspace(-0.9, -0.1, frames[0]), np.linspace(0.1, 0.9, frames[1])]
    for path, waveform in zip(paths, waveforms, strict=True):
        sf.write(path, waveform, 16000, subtype="FLOAT")
    source = ";".join(map(str, paths))
    ds = datasets.Dataset.from_list([{"audio_path": source}])
    result = random_segment(ds, **mode)
    parts = [load_audio({"audio_path": part}, min_dur=0)[0] for part in result[0]["audio_path"]]
    original, _ = load_audio({"audio_path": source}, min_dur=0)
    assert len(parts) == 2
    assert sum(map(len, parts)) == len(original)
    if "segment_len" in mode:
        assert [len(part) for part in parts] == [16000, 16000]
    # Compare against a slice of the merged waveform, including file crossings.
    split = len(parts[0])
    np.testing.assert_array_equal(parts[0], original[:split])
    np.testing.assert_array_equal(parts[1], original[split:])
    np.testing.assert_array_equal(np.concatenate(parts), original)


def test_random_segments_partition_the_entire_concatenated_waveform(tmp_path):
    paths = [tmp_path / name for name in ("a.wav", "b.wav")]
    for path, value in zip(paths, (-0.5, 0.5), strict=True):
        sf.write(path, np.full(16000, value), 16000, subtype="FLOAT")
    source = ";".join(map(str, paths))
    ds = datasets.Dataset.from_list([{"audio_path": source}])
    result = random_segment(ds, segment_count=[4, 4], seed=7)
    original, _ = load_audio({"audio_path": source}, min_dur=0)
    parts = [load_audio({"audio_path": part}, min_dur=0)[0] for part in result[0]["audio_path"]]
    assert sum(map(len, parts)) == len(original)
    np.testing.assert_array_equal(np.concatenate(parts), original)


@pytest.mark.parametrize("duration_mode", [False, True])
def test_multisegment_config_keeps_sample_counts_and_per_segment_budgets(duration_mode):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    import recipe.phimm

    directory = Path(recipe.phimm.__file__).parent / "config/v2611_dev/main"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        overrides = ["+asr_training=multi_segment"]
        if duration_mode:
            overrides += ["data.random_segment.segment_count=null", "data.random_segment.segment_len=[10,20]"]
        config = compose(config_name="remax_main_base3a_2611r2", overrides=overrides)
    OmegaConf.resolve(config)
    if duration_mode:
        assert config.data.random_segment.segment_count is None
        assert list(config.data.random_segment.segment_len) == [10, 20]
    else:
        assert list(config.data.random_segment.segment_count) == [1, 2]
        assert config.data.random_segment.segment_len is None
    assert config.data.max_audio_dur is None
    assert config.actor_rollout_ref.rollout.response_length == 768
    assert config.asr.expected_training_rows == 55664
    assert config.trainer.experiment_name is None

"""Verify the shared offline packing helper and downstream SL reconstruction."""

import datasets
import numpy as np
import pytest
import soundfile as sf
from recipe.phimm.tools.random_merge import _iter_random_merge, random_merge
from recipe.phimm.tools.random_segment import random_segment
from recipe.phimm.utils.audio import load_audio


def rows(count, duration=20):
    return [dict(audio_path=f"{i}.wav", text=f"word{i}", audio_length=duration, id=i) for i in range(count)]


def test_packing_strict_cap_reproducibility_and_exact_conservation():
    ds = datasets.Dataset.from_list(rows(1000))
    result = random_merge(ds, seed=7)
    assert len(result) == 35
    assert result.to_list() == random_merge(ds, seed=7).to_list()
    assert result.to_list() != random_merge(ds, seed=8).to_list()
    indices = []
    for row in result:
        order = row["random_merge_source_indices"]
        indices.extend(order)
        assert row["audio_path"] == ";".join(f"{i}.wav" for i in order)
        assert row["text"] == " ".join(f"word{i}" for i in order)
        assert row["id"] == ";".join(str(i) for i in order)
        assert row["audio_length"] == 20 * len(order) < 600
        assert row["random_merge_count"] == len(order)
    assert sorted(indices) == list(range(1000))
    assert sum(result["audio_length"]) == 20000
    assert result["audio_length"][:-1] == [580] * 34


def test_buffer_lookahead_is_bounded_and_refilled_during_packing():
    class Source:
        consumed = 0

        def __iter__(self):
            for row in rows(10000, 200):
                self.consumed += 1
                yield row

    source = Source()
    packs = _iter_random_merge(source, max_audio_length=600, buffer_size=100, seed=42, text_separator=" ", group_by=())
    first = next(packs)
    assert first["random_merge_count"] == 2
    assert source.consumed == 102
    ds = datasets.Dataset.from_list(
        [dict(row, audio_length=d) for row, d in zip(rows(5), [8, 1, 1, 1, 1], strict=True)]
    )
    result = random_merge(ds, max_audio_length=10, buffer_size=2, seed=1)
    assert result[0]["audio_length"] == 9
    assert sum(result["random_merge_count"]) == 5


@pytest.mark.parametrize("primary", ["audio_path", "audio_file", "audio_chunk"])
def test_unused_null_audio_aliases_accept_merged_paths(primary):
    data = []
    for i in range(3):
        row = dict.fromkeys(("audio_path", "audio_file", "audio_chunk"))
        row.update({primary: f"{i}.wav", "text": f"word{i}", "audio_length": 20})
        data.append(row)
    result = random_merge(datasets.Dataset.from_list(data))
    assert len(result) == 1
    order = result[0]["random_merge_source_indices"]
    expected = ";".join(f"{i}.wav" for i in order)
    for field in ("audio_path", "audio_file", "audio_chunk"):
        assert result.features[field].dtype == "string"
        assert result[0][field] == expected


def test_language_groups_keywords_components_and_zero_id():
    data = rows(6)
    for i, row in enumerate(data):
        row.update(
            language="Chinese" if i % 2 else "English",
            keywords=[f"word{i}", "shared"],
            components=[{"language": "Chinese" if i % 2 else "English", "text": row["text"]}],
        )
    result = random_merge(datasets.Dataset.from_list(data), text_separator="")
    assert len(result) == 2
    for row in result:
        order = row["random_merge_source_indices"]
        assert len({data[i]["language"] for i in order}) == 1
        assert row["text"] == "".join(data[i]["text"] for i in order)
        assert row["keywords"] == list(dict.fromkeys(w for i in order for w in data[i]["keywords"]))
        assert row["components"] == [data[i]["components"][0] for i in order]
        assert row["id"] == ";".join(str(i) for i in order)


@pytest.mark.parametrize("audio_field", ["audio_path", "audio_file", "audio_chunk"])
def test_merged_audio_and_random_segments_reconstruct_exact_waveform(tmp_path, audio_field):
    data = []
    waves = {}
    for i, frames in enumerate([12000, 20000, 24000]):
        path = tmp_path / f"{i}.wav"
        wave = np.full(frames, (i + 1) / 8, dtype=np.float32)
        sf.write(path, wave, 16000, subtype="FLOAT")
        data.append({audio_field: f"{path}#0%:100%", "text": f"word{i}"})
        waves[i] = wave
    merged = random_merge(datasets.Dataset.from_list(data), max_audio_length=10)
    assert len(merged) == 1
    order = merged[0]["random_merge_source_indices"]
    expected = np.concatenate([waves[i] for i in order])
    assert merged[0]["audio_length"] == len(expected) / 16000
    np.testing.assert_array_equal(load_audio({"audio_path": merged[0]["audio_path"]})[0], expected)
    segmented = random_segment(merged, segment_len=[0.3, 0.7])
    actual = np.concatenate([load_audio({"audio_path": path}, min_dur=0)[0] for path in segmented[0]["audio_path"]])
    np.testing.assert_array_equal(actual, expected)
    assert segmented[0]["text"] == " ".join(f"word{i}" for i in order)


def test_precomputed_duration_avoids_audio_io(monkeypatch):
    def fail(source):
        raise AssertionError("must use audio_length")

    monkeypatch.setattr("recipe.phimm.tools.random_merge.get_audio_duration", fail)
    result = random_merge(datasets.Dataset.from_list(rows(7)), seed=19)
    assert result[0]["audio_length"] == 140


def test_raw_manifest_formatted_reference_is_removed_before_regeneration():
    ds = datasets.Dataset.from_list([dict(row, gt_output="stale reference") for row in rows(3)])
    result = random_merge(ds)
    assert len(result) == 1
    assert "gt_output" not in result.column_names
    assert sorted(result[0]["text"].split()) == ["word0", "word1", "word2"]


@pytest.mark.parametrize(
    "options",
    [
        {"max_audio_length": 0},
        {"max_audio_length": True},
        {"max_audio_length": float("inf")},
        {"buffer_size": 0},
        {"buffer_size": 1.5},
        {"buffer_size": True},
        {"seed": 1.5},
        {"text_separator": None},
        {"group_by": "language"},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        random_merge(datasets.Dataset.from_list(rows(1)), **options)


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), 600, 601])
def test_invalid_or_oversized_utterance_fails_without_dropping(duration):
    with pytest.raises(datasets.exceptions.DatasetGenerationError) as error:
        random_merge(datasets.Dataset.from_list(rows(1, duration)))
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("source", ["a.wav;b.wav", "a.wav#0%:100%#0%:50%"])
def test_reject_already_merged_or_nested_segmented_sources(source):
    with pytest.raises(datasets.exceptions.DatasetGenerationError) as error:
        random_merge(datasets.Dataset.from_list([dict(rows(1)[0], audio_path=source)]))
    assert "individual sources" in str(error.value.__cause__)


def test_empty_and_disabled():
    ds = datasets.Dataset.from_list(rows(1))
    assert random_merge(ds, enabled=False) is ds
    empty = random_merge(ds.select([]))
    assert len(empty) == 0
    assert empty.features["audio_length"].dtype == "float64"

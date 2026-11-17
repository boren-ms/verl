"""Offline manifests preserve ordered content and are readable by audio loaders."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml
from recipe.phimm.data.dataset import augment, jsonl_dataset, process_ds
from recipe.phimm.data.offline_random_merge import merge_metadata, prepare_rows, run
from recipe.phimm.utils.audio import load_audio


def test_offline_order_provenance_schema_and_audio(tmp_path):
    originals = []
    waves = []
    for i in range(7):
        path = tmp_path / f"{i}.wav"
        wave = np.full(16000, i / 10, dtype=np.float32)
        sf.write(path, wave, 16000, subtype="FLOAT")
        waves.append(wave)
        originals.append(
            dict(
                audio_file=str(path),
                text=f"text{i}",
                id=i,
                language="English",
                audio_length=1.0,
                keywords=["shared", str(i)],
                source_row=i,
                start_seconds=float(i),
                display=f"display{i}",
                gt_output="stale",
                custom={"nullable": None if i % 2 else "present"},
            )
        )
    packed = list(prepare_rows(originals, originals, cap=3, seed=7, buffer_size=3))
    assert packed == list(prepare_rows(originals, originals, cap=3, seed=7, buffer_size=3))
    assert sorted(i for row in packed for i in row["random_merge_source_indices"]) == list(range(7))
    assert sum(row["audio_length"] for row in packed) == 7
    assert len(packed) == 4  # Strictly below three seconds; singleton retained.
    for row in packed:
        order = row["random_merge_source_indices"]
        assert row["audio_length"] < 3
        assert row["text"] == " ".join(originals[i]["text"] for i in order)
        assert row["display"] == " ".join(originals[i]["display"] for i in order)
        assert row["source_row"] == order
        assert row["start_seconds"] == [float(i) for i in order]
        assert [json.loads(value) for value in row["source_metadata_json"]] == [originals[i] for i in order]
        assert "gt_output" not in row
        audio, sr = load_audio(row)
        assert sr == 16000
        np.testing.assert_array_equal(audio, np.concatenate([waves[i] for i in order]))
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in packed))
    assert jsonl_dataset(str(path)).to_list() == packed


def test_incompatible_metadata_and_oversize_raise():
    rows = [
        dict(audio_path="a.wav", text="a", language="English", audio_length=180),
        dict(audio_path="b.wav", text="b", language="Chinese", audio_length=2),
    ]
    with pytest.raises(ValueError, match="Incompatible language"):
        merge_metadata(rows, [0, 1], rows)
    with pytest.raises(ValueError, match="must be below"):
        list(prepare_rows(rows, rows))


def test_nullable_metadata_across_arrow_json_blocks(tmp_path):
    # Producer fields can be absent in an entire first 1MB JSON parser block.
    # Keep optional values typed consistently rather than triggering Arrow's
    # null-list promotion failure when later blocks contain strings.
    rows = [
        dict(
            audio_path=f"{i}.wav",
            text="word",
            audio_length=1,
            keywords=[],
            evidence=None if i < 250 else "evidence",
            payload="x" * 8000,
        )
        for i in range(400)
    ]
    packed = list(prepare_rows(rows, rows, cap=3))
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in packed))
    loaded = jsonl_dataset(str(path)).to_list()
    assert loaded == packed
    for row in loaded:
        order = row["random_merge_source_indices"]
        assert [json.loads(value) for value in row["evidence"]] == [rows[i]["evidence"] for i in order]


@pytest.mark.parametrize("audio_format", ["semicolon", "list"])
def test_generated_yaml_loads_without_online_merge_transform(tmp_path, audio_format):
    source = tmp_path / "source.jsonl"
    rows = [dict(audio_path=f"{i}.wav", text=f"word{i}", audio_length=1.0, audience=f"audience{i}") for i in range(2)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    specs = tmp_path / "sources.json"
    specs.write_text(
        json.dumps(
            [
                dict(
                    name="example",
                    config=dict(dataset_name="jsonl", jsonl_paths=str(source), add_task_info={}, post_process={}),
                )
            ]
        )
    )
    output = tmp_path / "output"
    run(
        SimpleNamespace(
            sources=str(specs),
            output_dir=str(output),
            input_dir=None,
            blob_root=None,
            cap=3,
            buffer_size=100,
            seed=42,
            workers=1,
            audio_format=audio_format,
            merge_count=[2, 3] if audio_format == "list" else None,
        )
    )
    for path in [output / "train_data.yaml", output / "example/train_data.yaml"]:
        [block] = yaml.safe_load(path.read_text())
        assert "random_merge" not in block.get("pre_process", {})
        loaded = augment(jsonl_dataset(block["jsonl_paths"]), **block)
        assert len(loaded) == 1
        row = loaded[0]
        indices = row["random_merge_source_indices"]
        expected = [rows[i]["audience"] for i in indices]
        assert row["audience"] == ([json.dumps(value) for value in expected] if audio_format == "list" else expected)
        if audio_format == "list":
            assert row["audio_path"] == [rows[i]["audio_path"] for i in indices]
        assert row["text"] == " ".join(rows[i]["text"] for i in indices)
        assert row["audio_length"] == 2.0


def test_dataset_merge_block_emits_list_without_changing_offline_output(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(dict(audio_path="a.wav", text="a", audio_length=1.0)) + "\n")
    result = process_ds(jsonl_dataset(str(path)), random_merge={"max_audio_length": 180})
    assert result[0]["audio_path"] == ["a.wav"]
    assert result[0]["text"] == "a"

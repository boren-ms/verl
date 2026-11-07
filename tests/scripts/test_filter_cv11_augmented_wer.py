import pytest
import json
from argparse import Namespace

from scripts.filter_cv11_augmented_wer import blob_lines, join_row, keep_row, run


@pytest.mark.parametrize("errors,refs,expected", [(0, 10, False), (1, 100, True), (1, 101, False), (20, 10, True)])
def test_inclusive_one_percent_boundary_and_uncapped_wer(errors, refs, expected):
    assert keep_row({"n_err": errors, "n_ref": refs}) is expected


@pytest.mark.parametrize("errors,refs,expected", [
    (0, 20, False), (1, 20, False), (1, 21, False), (1, 19, True),
    (19, 20, True), (20, 20, False), (21, 20, False),
])
def test_strict_five_to_one_hundred_percent_boundaries(errors, refs, expected):
    assert keep_row({"n_err": errors, "n_ref": refs}, "wer_gt5pct_lt100pct") is expected


def example():
    source = {"id": "a", "text": "Original label.", "audio_path": "az://test/data/a.wav",
              "duration": 2.0, "sample_rate": 16000, "rir_sha256": "rir-hash", "snr_db": 4.5}
    generated = {"id": "a", "text": source["text"], "audio_path": source["audio_path"],
                 "audio_duration_seconds": 2.0, "response": "Different hypothesis", "wer": 0.5,
                 "n_err": 1, "n_ref": 2}
    return source, generated


def test_preserves_original_labels_and_augmentation_instead_of_pseudo_labels():
    source, generated = example()
    joined, _ = join_row(source, generated)
    assert joined["text"] == source["text"] != generated["response"]
    assert joined["rir_sha256"] == source["rir_sha256"]
    assert joined["snr_db"] == 4.5
    assert joined["wer"] == 0.5


def test_strict_range_has_accurate_metadata_and_preserves_source():
    source, generated = example()
    joined, _ = join_row(source, generated, "wer_gt5pct_lt100pct")
    assert joined["data_source"] == "noisy_cv11_en_train_2609r2_wer_gt5pct_lt100pct"
    assert joined["filter_wer_threshold_fraction"] == 0.05
    assert joined["filter_wer_upper_threshold_fraction"] == 1.0
    assert joined["filter_wer_lower_inclusive"] is False
    assert joined["filter_wer_upper_inclusive"] is False
    for key, value in source.items():
        assert joined[key] == value


def test_strict_range_publication_joins_all_sources_and_hashes_uploaded_output(tmp_path, monkeypatch):
    import hashlib
    import yaml
    import scripts.filter_cv11_augmented_wer as filtering

    source, generated = example()
    sources, generations = [], []
    for index, (errors, refs) in enumerate([(1, 20), (1, 19), (19, 20), (20, 20), (21, 20)]):
        sources.append({**source, "id": str(index), "audio_path": f"az://test/data/{index}.wav"})
        generations.append({**generated, "id": str(index), "audio_path": sources[-1]["audio_path"],
                            "n_err": errors, "n_ref": refs, "wer": errors / refs})
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources))
    monkeypatch.setattr(filtering, "SOURCE_SHA256", hashlib.sha256(source_path.read_bytes()).hexdigest())
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "part-000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generations))
    published = tmp_path / "published"
    training = tmp_path / "training.yaml"
    training.write_text(yaml.safe_dump({"jsonl_paths": str(published / "data.jsonl"),
        "post_process": {"add_field": {"fields": {
            "data_source": "noisy_cv11_en_train_2609r2_wer_gt5pct_lt100pct"}}}}))
    remax = tmp_path / "remax.yaml"
    remax.write_text("trainer: {total_epochs: 4}\n")
    args = Namespace(generation_dir=str(generation), source_jsonl=str(source_path),
        output_dir=str(published), work_dir=str(tmp_path / "work"), expected_rows=5,
        training_yaml=str(training), remax_yaml=str(remax), filter_mode="wer_gt5pct_lt100pct")
    run(args)
    rows = [json.loads(line) for line in (published / "data.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["1", "2"]
    assert all(row["text"] == source["text"] for row in rows)
    complete = json.loads((published / "COMPLETE.json").read_text())
    assert complete["source_rows"] == 5
    assert complete["filtered_rows"] == 2
    assert complete["filtered_manifest_sha256"] == hashlib.sha256((published / "data.jsonl").read_bytes()).hexdigest()
    summary = json.loads((published / "summary.json").read_text())
    assert summary["filtered"]["total_words"] == 4
    assert summary["filtered"]["audio_hours"] == 4 / 3600


@pytest.mark.parametrize("field,value", [("text", "Wrong reference"), ("id", "b"),
                                        ("audio_path", "az://test/data/b.wav"), ("audio_duration_seconds", 1.0)])
def test_rejects_wrong_pairing_and_trimmed_audio(field, value):
    source, generated = example()
    generated[field] = value
    with pytest.raises(ValueError):
        join_row(source, generated)


@pytest.mark.parametrize("missing_row", [True, False])
def test_incomplete_generation_or_wrong_source_hash_does_not_publish(tmp_path, missing_row):
    source, generated = example()
    source_path = tmp_path / "source.jsonl"
    sources = [source]
    if missing_row:
        sources.append({**source, "id": "b", "audio_path": "az://test/data/b.wav"})
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources))
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "part-000.jsonl").write_text(json.dumps(generated) + "\n")
    published = tmp_path / "published"
    args = Namespace(generation_dir=str(generation), source_jsonl=str(source_path),
                     output_dir=str(published), work_dir=str(tmp_path / "work"),
                     expected_rows=len(sources), training_yaml="unused", remax_yaml="unused")
    with pytest.raises(ValueError, match="cover every source row|pinned manifest checksum"):
        run(args)
    assert not (published / "data.jsonl").exists()
    assert not (published / "COMPLETE.json").exists()


@pytest.mark.parametrize("ending", [b"", b"\n"])
def test_large_range_reader_preserves_utf8_jsonl_bytes_at_chunk_boundaries(tmp_path, ending):
    rows = [{"text": "English, 中文, café", "value": i} for i in range(3)]
    data = b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows) + ending
    path = tmp_path / "utf8.jsonl"
    path.write_bytes(data)
    lines = list(blob_lines(str(path), read_size=17))
    assert b"".join(lines) == data
    assert [json.loads(line) for line in lines] == rows

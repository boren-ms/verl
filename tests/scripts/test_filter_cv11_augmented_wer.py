import pytest
import json
from argparse import Namespace

from scripts.filter_cv11_augmented_wer import join_row, keep_row, run


@pytest.mark.parametrize("errors,refs,expected", [(0, 10, False), (1, 100, True), (1, 101, False), (20, 10, True)])
def test_inclusive_one_percent_boundary_and_uncapped_wer(errors, refs, expected):
    assert keep_row({"n_err": errors, "n_ref": refs}) is expected


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

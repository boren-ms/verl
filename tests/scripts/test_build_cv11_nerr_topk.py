import hashlib
import json
from argparse import Namespace

import pytest
import yaml

from scripts.build_cv11_nerr_topk import run, select_topk
from scripts.filter_cv11_augmented_wer import MODEL_PATH


def rows_bytes(rows):
    return [json.dumps(row).encode() + b"\n" for row in rows]


def examples():
    rows = []
    for index, (errors, refs) in enumerate([(8, 20), (6, 7), (8, 19), (10, 15), (8, 18)]):
        rows.append({"id": str(index), "text": "Original label.", "audio_path": f"az://test/data/{index}.wav",
            "duration": 2.0, "n_err": errors, "n_ref": refs, "wer": errors / refs,
            "response": "Other label", "filter_model_path": MODEL_PATH,
            "rir_sha256": "rir-hash", "snr_db": 4.5,
            "data_source": "noisy_cv11_en_train_2609r2_wer_gt5pct_lt100pct"})
    return rows


def select(rows, count=3, expected=None, sha=None):
    lines = rows_bytes(rows)
    return select_topk(lines, count, len(rows) if expected is None else expected,
        sha or hashlib.sha256(b"".join(lines)).hexdigest(), "wer_gt5pct_lt100pct")


def test_ranks_by_error_count_and_breaks_boundary_ties_by_source_order():
    selected, summary, counts = select(examples())
    assert [json.loads(item[2])["id"] for item in selected] == ["3", "0", "2"]
    assert summary["count"] == 5
    assert counts[8] == 3
    assert json.loads(selected[-1][2])["wer"] < examples()[1]["wer"]


@pytest.mark.parametrize("expected,sha", [(6, None), (None, "0" * 64)])
def test_requires_full_pinned_candidate_manifest(expected, sha):
    with pytest.raises(ValueError, match="count or pinned checksum"):
        select(examples(), expected=expected, sha=sha)


def test_rejects_duplicate_ids_and_invalid_wer():
    rows = examples()
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        select(rows)
    rows = examples()
    rows[0]["wer"] = float("nan")
    with pytest.raises(ValueError, match="WER disagrees"):
        select(rows)


@pytest.mark.parametrize("count", [0, 6])
def test_rejects_invalid_topk_size(count):
    with pytest.raises(ValueError, match="Top-K count"):
        select(examples(), count=count)


def test_publication_preserves_labels_audio_and_provenance_and_verifies_upload(tmp_path):
    rows = examples()
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "data.jsonl"
    source.write_bytes(b"".join(rows_bytes(rows)))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    (parent / "COMPLETE.json").write_text(json.dumps({"filtered_manifest_sha256": sha,
        "filtered_rows": 5, "remote_manifest_sha256_verified": True}))
    (parent / "summary.json").write_text(json.dumps({"filtered_manifest_sha256": sha,
        "filter_mode": "wer_gt5pct_lt100pct", "filter": "5% < WER < 100%", "scoring": "2609"}))
    destination = tmp_path / "published"
    training = tmp_path / "training.yaml"
    training.write_text(yaml.safe_dump({"jsonl_paths": str(destination / "data.jsonl"),
        "post_process": {"add_field": {"fields": {"data_source": "cv11_topk"}}}}))
    remax = tmp_path / "remax.yaml"
    remax.write_text("trainer: {total_epochs: 4}\n")
    run(Namespace(source_jsonl=str(source), source_sha256=sha, expected_rows=5,
        source_mode="wer_gt5pct_lt100pct", count=3, output_dir=str(destination),
        work_dir=str(tmp_path / "work"), training_yaml=str(training), remax_yaml=str(remax)))
    selected = [json.loads(line) for line in (destination / "data.jsonl").read_text().splitlines()]
    assert [row["id"] for row in selected] == ["3", "0", "2"]
    for rank, (row, source_index) in enumerate(zip(selected, [3, 0, 2]), 1):
        for key, value in rows[source_index].items():
            if key != "data_source":
                assert row[key] == value
        assert row["selection_nerr_rank"] == rank
        assert row["selection_source_row_index"] == source_index
    complete = json.loads((destination / "COMPLETE.json").read_text())
    assert complete["filtered_manifest_sha256"] == hashlib.sha256((destination / "data.jsonl").read_bytes()).hexdigest()
    assert complete["selection"]["cutoff_n_err"] == 8
    assert complete["selection"]["candidate_ties_at_cutoff"] == 3
    assert complete["selection"]["selected_ties_at_cutoff"] == 2
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["filtered"]["count"] == 3
    assert summary["filtered"]["total_words"] == 6
    assert summary["filtered"]["audio_hours"] == 6 / 3600

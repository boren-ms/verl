import json

import pytest
import yaml

from scripts import build_openlong_all_segments_with_baseline as builder


def test_keeps_all_segments_and_attaches_baseline(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.EXPECTED, "earnings21", (5, 2))
    rows = []
    for parent, errors, count in (("a", 3, 2), ("b", 9, 3)):
        for i in range(count):
            rows.append({"data_source": "earnings21", "parent_audio_path": parent,
                         "gts": "hello there friend", "seg_index": i, "n_segments": count,
                         "id": parent, "seg_start": i * 5, "seg_end": i * 5 + 5,
                         "audio_path": f"{parent}#{i * 5}:{i * 5 + 5}",
                         "clean_output": "hello" if i == 0 else "there friend",
                         "n_err": errors if i == 0 else None, "n_ref": 3 if i == 0 else None})
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = builder.build_dataset(source, output, "earnings21")
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["segments"] == 5
    assert [(r["parent_audio_path"], r["seg_index"]) for r in selected] == [
        ("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)
    ]
    assert all(r["baseline_hypothesis"] for r in selected)
    assert {r["parent_rank"] for r in selected if r["parent_audio_path"] == "b"} == {1}
    for parent in ("a", "b"):
        assert " ".join(r["text"] for r in selected if r["parent_audio_path"] == parent).split() == [
            "hello", "there", "friend"
        ]
    rows[0]["seg_index"] = 1
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete or duplicate segments"):
        builder.build_dataset(source, output, "earnings21")


def test_yaml_retains_hypothesis_without_grouping_segments():
    with open("recipe/phimm/config/data/val_data/openlong_all_segments_baseline_2609r2b.yaml") as handle:
        rows = yaml.safe_load(handle)
    assert len(rows) == 4
    for row in rows:
        keys = row["post_process"]["verl_format"]["extra_keys"]
        assert "baseline_hypothesis" in keys
        assert "parent_audio_path" not in keys and "audio_path" not in keys

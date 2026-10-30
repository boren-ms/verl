import json

import pytest
import yaml

from scripts import build_openlong_high_error_two_parents as builder


def test_selects_two_highest_error_parents_and_retains_all_segments(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.EXPECTED, "earnings21", (7, 3))
    rows = []
    for parent, errors, count in (("a", 9, 2), ("b", 17, 3), ("c", 12, 2)):
        for i in range(count):
            rows.append({"data_source": "earnings21", "parent_audio_path": parent,
                         "gts": "hello there friend", "seg_index": i, "n_segments": count,
                         "id": parent, "seg_start": i * 5, "seg_end": i * 5 + 5,
                         "audio_path": f"{parent}#{i * 5}:{i * 5 + 5}",
                         "clean_output": ("hello" if i == 0 else "there friend"),
                         "n_err": errors if i == 0 else None, "n_ref": 3 if i == 0 else None})
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary, examples = builder.build(source, output, "earnings21")
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert [p["parent"] for p in summary["parents"]] == ["b", "c"]
    assert [p["selected_segments"] for p in summary["parents"]] == [3, 2]
    assert {r["parent_audio_path"] for r in selected} == {"b", "c"}
    assert [(r["parent_audio_path"], r["seg_index"]) for r in selected] == [
        ("b", 0), ("b", 1), ("b", 2), ("c", 0), ("c", 1)
    ]
    assert all(len(group) == 2 for group in examples.values())
    rows[0]["seg_index"] = 1
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Duplicate segment index"):
        builder.build(source, output, "earnings21")


def test_yaml_scores_clips_without_parent_grouping():
    with open("recipe/phimm/config/data/val_data/openlong_all_2parents_error_2609r2b.yaml") as handle:
        rows = yaml.safe_load(handle)
    assert len(rows) == 4
    assert all(not r["post_process"]["verl_format"].get("extra_keys") for r in rows)

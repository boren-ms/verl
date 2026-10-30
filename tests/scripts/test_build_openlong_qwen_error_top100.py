import json

import pytest
import yaml

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts import build_openlong_qwen_error_top100 as builder


def make_row(ident, text, hyp):
    metric = measure_openasr_en_wer(hyp, text)
    return {"id": ident, "text": text, "baseline_hypothesis": hyp,
            "baseline_n_err": metric["n_err"], "baseline_n_ref": metric["n_ref"],
            "baseline_wer": metric["wer"], "alignment_status": "needs_review",
            "alignment_flags": ["test"]}


def test_ranks_total_errors_including_insertions_and_empty_reference(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.EXPECTED, "earnings21", (4, 1))
    rows = [make_row("empty", "", "alpha beta gamma"),
            make_row("insert", "cat", "cat red blue"),
            make_row("delete", "cat red blue", "cat"),
            make_row("match", "cat", "cat")]
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = builder.build_dataset(source, output, "earnings21", builder.sha256(source), limit=3)
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in selected] == ["empty", "insert", "delete"]
    assert selected[0]["baseline_n_ins"] == 3 and selected[0]["reference_word_count"] == 0
    assert selected[1]["baseline_n_ins"] == 2 and selected[2]["baseline_n_del"] == 2
    assert [row["error_count_rank"] for row in selected] == [1, 2, 3]
    assert summary["empty_normalized_references"] == 1
    with pytest.raises(FileExistsError):
        builder.build_dataset(source, output, "earnings21", builder.sha256(source), limit=3)


def test_mismatched_source_or_saved_metric_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.EXPECTED, "earnings21", (1, 1))
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    row = make_row("x", "cat", "cat")
    row["baseline_n_err"] = 9
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        builder.build_dataset(source, output, "earnings21", "wrong", limit=1)
    with pytest.raises(ValueError, match="metric mismatch"):
        builder.build_dataset(source, output, "earnings21", builder.sha256(source), limit=1)
    assert not output.exists()


def test_yaml_exposes_error_components_without_parent_grouping():
    with open("recipe/phimm/config/data/val_data/openlong_qwen_error_top100_2609r2b.yaml", encoding="utf-8") as handle:
        rows = yaml.safe_load(handle)
    assert len(rows) == 4
    for row in rows:
        keys = row["post_process"]["verl_format"]["extra_keys"]
        assert {"baseline_n_ins", "baseline_n_del", "baseline_n_sub", "reference_word_count"} <= set(keys)
        assert "parent_audio_path" not in keys and "audio_path" not in keys

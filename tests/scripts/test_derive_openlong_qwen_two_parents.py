import json

import pytest

from scripts.derive_openlong_qwen_two_parents import derive


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_subset_keeps_every_segment_from_both_parents(tmp_path):
    rows = [{"id": f"{parent}:{i}", "parent_audio_path": parent, "seg_index": i,
             "n_segments": count, "audio_path": f"{parent}#{i}", "baseline_hypothesis": "hyp",
             "alignment_status": "qwen_candidate"}
            for parent, count in (("a", 2), ("b", 3)) for i in range(count)]
    extra = {"id": "c:0", "parent_audio_path": "c", "seg_index": 0,
             "n_segments": 1, "audio_path": "c#0", "baseline_hypothesis": "hyp",
             "alignment_status": "needs_review"}
    full, selection, out = (tmp_path / name for name in ("full.jsonl", "selection.jsonl", "out.jsonl"))
    write_rows(full, rows + [extra])
    write_rows(selection, rows)
    summary = derive(full, selection, out)
    assert summary["segments"] == 5 and summary["parents"] == 2
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == [r["id"] for r in rows]


def test_missing_selected_segment_fails(tmp_path):
    rows = [{"id": f"{parent}:0", "parent_audio_path": parent, "seg_index": 0,
             "n_segments": 1, "audio_path": parent, "baseline_hypothesis": "hyp",
             "alignment_status": "qwen_candidate"} for parent in ("a", "b")]
    full, selection, out = (tmp_path / name for name in ("full.jsonl", "selection.jsonl", "out.jsonl"))
    write_rows(full, rows[:1])
    write_rows(selection, rows)
    with pytest.raises(ValueError, match="Missing"):
        derive(full, selection, out)
    assert not out.exists()

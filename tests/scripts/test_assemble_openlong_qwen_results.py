import json

import pytest

from scripts.assemble_openlong_qwen_results import assemble


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_window(dataset, ids):
    return {"dataset": dataset, "parent_audio_path": dataset, "core_segment_ids": ids,
            "audio_source": "audio.zip!1:2", "audio_start": 0, "audio_end": 2,
            "seed_text": "one two", "core_word_start": 0, "core_word_end": 2,
            "segments": [{"id": ids[0], "seg_start": 0, "seg_end": 1, "text": "one"},
                         {"id": ids[1], "seg_start": 1, "seg_end": 2, "text": "two"}]}


def test_replacement_and_retry_core_coverage(tmp_path):
    a, b = make_window("a", ["a0", "a1"]), make_window("b", ["b0", "b1"])
    retries = []
    for segment in b["segments"]:
        retry = dict(b, core_segment_ids=[segment["id"]], retry_original_core_segment_ids=b["core_segment_ids"],
                     retry_reason="collapsed_timestamps", audio_start=segment["seg_start"],
                     audio_end=segment["seg_end"], seed_text=segment["text"])
        retries.append({"window": retry, "status": "ok"})
    plan, shard, replacement = (tmp_path / name for name in ("plan", "shard", "replacement"))
    write_rows(plan, [a, b])
    write_rows(shard, [{"window": a, "status": "ok"}, {"window": b, "status": "ok"}])
    write_rows(replacement, retries)
    summary = assemble(plan, [shard], replacement, "b", tmp_path / "out")
    assert summary["planned_windows"] == 2 and summary["result_rows"] == 3
    assert summary["single_segment_retries"] == 2


def test_missing_retry_fails_without_final_output(tmp_path):
    a, b = make_window("a", ["a0", "a1"]), make_window("b", ["b0", "b1"])
    retry = dict(b, core_segment_ids=["b0"], retry_original_core_segment_ids=b["core_segment_ids"],
                 retry_reason="collapsed_timestamps", audio_start=0, audio_end=1, seed_text="one")
    plan, shard, replacement = (tmp_path / name for name in ("plan", "shard", "replacement"))
    write_rows(plan, [a, b])
    write_rows(shard, [{"window": a, "status": "ok"}])
    write_rows(replacement, [{"window": retry, "status": "ok"}])
    with pytest.raises(ValueError, match="Incomplete"):
        assemble(plan, [shard], replacement, "b", tmp_path / "out")
    assert not (tmp_path / "out/results.jsonl").exists()

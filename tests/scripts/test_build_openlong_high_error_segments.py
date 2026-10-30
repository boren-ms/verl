import json

import pytest
import yaml

from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from scripts import build_openlong_high_error_segments as builder


def test_build_ranks_independent_segments_and_checks_parent_score(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.EXPECTED, "earnings21", (2, 1))
    parent = "az://test/audio.wav"
    reference = "one two three four five six"
    hyps = ["one two", "three wrong wrong wrong"]
    grouped = measure_openasr_en_wer("\n".join(hyps), reference)
    rows = [
        {"data_source": "earnings21", "parent_audio_path": parent, "gts": reference,
         "seg_index": i, "seg_start": i * 10, "seg_end": i * 10 + 10,
         "n_segments": 2, "id": "0", "audio_path": f"{parent}#{i * 10}:{i * 10 + 10}",
         "clean_output": hyp, "n_err": grouped["n_err"] if i == 0 else None}
        for i, hyp in enumerate(hyps)
    ]
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = builder.build_dataset(source, output, "earnings21", 1)
    selected = json.loads(output.read_text())
    assert result["selected"] == 1
    assert selected["seg_index"] == 1
    assert selected["text"] == "three four five six"
    assert selected["baseline_n_err"] == measure_openasr_en_wer(hyps[1], selected["text"])["n_err"]
    assert selected["baseline_n_err"] > 0
    rows[0]["n_err"] = 999
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Grouped score mismatch"):
        builder.build_dataset(source, output, "earnings21", 1, verify_parent_scores=True)


def test_config_does_not_group_selected_clips_by_parent():
    path = "recipe/phimm/config/data/val_data/openlong_top100_error_2609r2b.yaml"
    config = yaml.safe_load(open(path, encoding="utf-8"))
    assert len(config) == 4
    assert all(not row["post_process"]["verl_format"].get("extra_keys") for row in config)

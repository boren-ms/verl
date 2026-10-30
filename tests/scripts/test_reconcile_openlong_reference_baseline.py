import json

import pytest

from scripts import reconcile_openlong_reference_baseline as hybrid
from scripts import validate_openlong_reference_baseline as validator


def segment(index, seed, hyp, qwen, flags=None):
    return {"id": f"s{index}", "parent_audio_path": "parent", "seg_index": index,
            "seg_start": index * 10, "seg_end": index * 10 + 10, "n_segments": 2,
            "prior_inferred_text": seed, "baseline_hypothesis": hyp, "text": qwen,
            "alignment_flags": flags or [], "alignment_status": "qwen_candidate"}


def test_rejects_pathological_qwen_and_preserves_original_reference():
    rows = [segment(0, "alpha beta", "alpha beta", "alpha beta " + "noise " * 200),
            segment(1, "gamma delta", "gamma delta", "gamma delta")]
    output, qa = hybrid.reconcile_parent(rows, "alpha beta gamma delta")
    assert [r["text"] for r in output] == ["alpha beta", "gamma delta"]
    assert qa["qwen_outlier_segments"] == 1
    assert "qwen_text_outlier_rejected" in output[0]["alignment_flags"]
    assert sum(r["baseline_n_err"] for r in output) == 0


def test_qwen_can_move_an_unanchored_word_to_adjacent_segment():
    rows = [segment(0, "alpha rare", "alpha", "alpha"),
            segment(1, "gamma delta", "gamma delta", "rare gamma delta")]
    output, _ = hybrid.reconcile_parent(rows, "alpha rare gamma delta")
    assert [r["text"] for r in output] == ["alpha", "rare gamma delta"]
    assert output[1]["alignment_diagnostics"]["qwen_near_seed"] == 1


def test_unreliable_qwen_placement_does_not_override_seed():
    rows = [segment(0, "alpha rare", "alpha", "alpha"),
            segment(1, "gamma delta", "gamma delta", "rare gamma delta", ["window_edge"])]
    output, _ = hybrid.reconcile_parent(rows, "alpha rare gamma delta")
    assert [r["text"] for r in output] == ["alpha rare", "gamma delta"]


def test_qwen_zero_duration_support_is_flagged_for_review():
    rows = [segment(0, "alpha rare", "alpha", "alpha"),
            segment(1, "gamma delta", "gamma delta", "rare gamma delta", ["zero_duration"])]
    output, _ = hybrid.reconcile_parent(rows, "alpha rare gamma delta")
    assert output[1]["text"] == "rare gamma delta"
    assert "qwen_zero_duration_support" in output[1]["alignment_flags"]
    assert output[1]["alignment_status"] == "needs_review"


def test_invalid_original_or_nonmonotonic_projection():
    rows = [segment(0, "alpha", "alpha", "alpha"), segment(1, "beta", "beta", "beta")]
    with pytest.raises(ValueError, match="original test reference"):
        hybrid.reconcile_parent(rows, "different reference")
    assert hybrid.monotone_owners([0, 2, 1, 3], [1, 1, 1, 1]) == [0, 2, 2, 3]


def test_build_fails_closed_on_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setitem(hybrid.EXPECTED, "test", (2, 1))
    source, original, out = (tmp_path / name for name in ("source", "original", "out"))
    rows = [segment(0, "alpha", "alpha", "alpha"), segment(1, "beta", "beta", "beta")]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original.write_text("".join(json.dumps(dict(r, gts="alpha beta", data_source="test")) + "\n" for r in rows))
    with pytest.raises(ValueError, match="checksum"):
        hybrid.build_dataset(source, original, out, "test", "wrong", hybrid.sha256(original))
    assert not out.exists()


def test_original_reference_and_qwen_provenance_validate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(hybrid.EXPECTED, "test", (2, 1))
    monkeypatch.setitem(validator.EXPECTED, "test", (2, 1))
    rows = [segment(0, "alpha", "alpha", "alpha"), segment(1, "beta", "beta", "beta")]
    for row in rows:
        row["audio_path"] = f"audio#{row['seg_index']}"
    source, original, out = (tmp_path / name for name in ("source", "original", "out"))
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original.write_text("".join(json.dumps(dict(r, gts="alpha beta", data_source="test")) + "\n" for r in rows))
    hybrid.build_dataset(source, original, out, "test", hybrid.sha256(source), hybrid.sha256(original))
    assert validator.validate(source, original, out, "test")["segments"] == 2
    result = [json.loads(line) for line in out.read_text().splitlines()]
    result[0]["text"] = "wrong"
    out.write_text("".join(json.dumps(r) + "\n" for r in result))
    with pytest.raises(ValueError, match="test text"):
        validator.validate(source, original, out, "test")

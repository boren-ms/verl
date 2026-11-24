"""Debug compaction must retain every raw field and scale with chunk count."""

import copy
import json

import pytest

from recipe.phimm.agentic.details import validation_detail


def sample(count):
    paths = [f"audio.zip!100:160044#{i * 2}:{(i + 1) * 2}" for i in range(count)]
    info = {"id": "parent", "audio_path": paths, "n_segments": count, "chunking_method": "hard_nonoverlapping"}
    original = {
        "prompt": [{"role": "user", "content": "Transcribe in English."}],
        "audio_path": paths,
        "reward_model": {"ground_truth": "Exact full-parent reference. " * 1000},
        "extra_info": info,
        "asr_full_reference": True,
    }
    rows = [{**original, "audio_path": path} for path in paths]
    # Metadata overrides must remain reversible even when one segment differs.
    rows[-1]["extra_info"] = {**info, "id": "last-segment"}
    return rows, info


def test_round_trip_retains_every_field_without_modifying_inputs():
    rows, info = sample(200)
    before = copy.deepcopy(rows)
    texts = [f"raw hypothesis {i}" for i in range(len(rows))]
    contexts = [""] * len(rows)
    detail = json.loads(json.dumps(validation_detail(rows, texts, info, contexts, validate=True)))
    rebuilt = [
        {
            **detail["parent_row"],
            **row,
            "extra_info": {**detail["parent_info"], **row["extra_info"]},
        }
        for row in detail["segment_rows"]
    ]
    assert rebuilt == before == rows
    assert detail["segment_outputs"] == texts
    assert detail["previous_contexts"] == contexts
    assert detail["parent_info"] == info
    assert [{**info, **entry} for entry in detail["segment_info"]] == [row["extra_info"] for row in rows]
    assert detail["parent_row"]["reward_model"]["ground_truth"] == rows[0]["reward_model"]["ground_truth"]


def test_ten_times_more_chunks_does_not_repeat_full_parent_ten_times_per_chunk():
    sizes = []
    for count in (100, 1000):
        rows, info = sample(count)
        detail = validation_detail(rows, [""] * count, info, [""] * count, validate=True)
        sizes.append(len(json.dumps(detail)))
    assert sizes[1] < sizes[0] * 10


@pytest.mark.parametrize("validate", [False, True])
def test_existing_scalar_segmentation_and_training_keep_their_format(validate):
    rows, info = sample(2)
    info["chunking_method"] = "original_processed_30s"
    detail = validation_detail(rows, ["a", "b"], info, ["", ""], validate=validate)
    assert "segment_rows_encoding" not in detail and "parent_row" not in detail
    assert detail["segment_info"] == [row["extra_info"] for row in rows]
    if validate:
        assert detail["segment_rows"] == rows
    else:
        assert "segment_rows" not in detail

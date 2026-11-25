"""Parent/segment transport must preserve inputs without repeating parent payloads."""

import copy
import json

import numpy as np

from recipe.phimm.agentic.details import segment_fields, segment_row, validation_detail


def sample(count):
    paths = [f"audio.zip!100:160044#{i * 2}:{(i + 1) * 2}" for i in range(count)]
    parent = {
        "prompt": [{"role": "user", "content": "Transcribe in English."}],
        "audio_path": paths,
        "reward_model": {"ground_truth": "Exact full-parent reference. " * 1000},
        "extra_info": {"id": "parent", "audio_path": paths, "n_segments": count},
        "asr_full_reference": True,
    }
    rows = [{**parent, "audio_path": path} for path in paths]
    rows[-1]["extra_info"] = {**parent["extra_info"], "id": "last-segment"}
    return parent, rows


def test_transport_keeps_shared_fields_on_parent_and_survives_json():
    parent, rows = sample(1000)
    before = copy.deepcopy(rows)
    segments = [segment_fields(parent, row) for row in rows]
    wire = json.dumps({**parent, "asr_segments": segments})
    loaded = json.loads(wire)
    assert isinstance(loaded["asr_segments"], list)
    assert segments[0] == {"audio_path": rows[0]["audio_path"]}
    assert all("prompt" not in s and "reward_model" not in s for s in segments)
    assert [segment_row(loaded, s) for s in loaded["asr_segments"]] == before == rows
    assert len(wire) < len(json.dumps(rows)) / 100


def test_processed_segments_keep_distinct_prompts_references_and_metadata():
    parent, rows = sample(2)
    rows[1] = {
        **rows[1],
        "prompt": [{"role": "user", "content": "Different language hint."}],
        "asr_reference_text": "second segment reference",
        "extra_info": {**rows[1]["extra_info"], "seg_start": 2.0, "language": "French"},
    }
    segments = [segment_fields(parent, row) for row in rows]
    loaded = json.loads(json.dumps({**parent, "asr_segments": segments}))
    assert [segment_row(loaded, s) for s in loaded["asr_segments"]] == rows
    assert segments[1]["prompt"] == rows[1]["prompt"]
    assert segments[1]["extra_info"] == {"id": "last-segment", "seg_start": 2.0, "language": "French"}


def test_debug_detail_uses_row_and_segment_fields_without_repacking():
    parent, rows = sample(200)
    before = copy.deepcopy(rows)
    segments = [segment_fields(parent, row) for row in rows]
    texts = [f"raw hypothesis {i}" for i in range(len(rows))]
    contexts = [""] * len(rows)
    detail = json.loads(json.dumps(validation_detail({**parent, "asr_segments": segments}, texts, contexts)))
    assert detail["segment_outputs"] == texts
    assert detail["previous_contexts"] == contexts
    assert detail["extra_info"] == parent["extra_info"]
    assert [
        {**detail["extra_info"], **entry} for entry in [s.get("extra_info", {}) for s in detail["asr_segments"]]
    ] == [r["extra_info"] for r in rows]
    assert "parent_row" not in detail and "parent_info" not in detail
    assert [segment_row(detail, s) for s in detail["asr_segments"]] == before == rows


def test_ten_times_more_chunks_does_not_repeat_full_parent_per_chunk():
    sizes = []
    for count in (100, 1000):
        parent, rows = sample(count)
        segments = [segment_fields(parent, row) for row in rows]
        detail = validation_detail({**parent, "asr_segments": segments}, [""] * count, [""] * count)
        sizes.append(len(json.dumps(detail)))
    assert sizes[1] < sizes[0] * 10


def test_inline_audio_arrays_are_segment_data_not_compared_as_shared_metadata():
    first = np.arange(4, dtype=np.float32)
    second = np.arange(8, dtype=np.float32)
    row = {"audio": first, "prompt": "Transcribe.", "extra_info": {}}
    segment = segment_fields(row, {**row, "audio": second})
    assert segment["audio"] is second
    assert "prompt" not in segment
    assert segment_row(row, segment)["audio"] is second

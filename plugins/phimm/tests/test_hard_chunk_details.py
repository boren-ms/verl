"""Compact dataset rows retain selectors, references and segment-specific metadata."""

import copy
import json

import numpy as np
import pytest

from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.agentic.details import RUNTIME_FIELDS, validation_detail


def sample(count):
    paths = [f"audio.zip!100:160044#{i * 2}:{(i + 1) * 2}" for i in range(count)]
    source = {
        "prompt": [{"role": "user", "content": "Transcribe in English."}],
        "audio_path": paths,
        "reward_model": {"ground_truth": "Exact full-recording reference. " * 1000},
        "extra_info": {"id": "recording", "audio_path": paths, "n_segments": count},
    }
    dataset = ASRDataset.__new__(ASRDataset)
    dataset.ds, dataset.groups, dataset.prompt_key = [source], [[0]], "prompt"
    return dataset[0], [{**source, "audio_path": path} for path in paths]


def test_transport_keeps_shared_fields_on_row_and_survives_json():
    row, expected = sample(1000)
    wire = json.dumps(row)
    loaded = json.loads(wire)
    shared = {key: value for key, value in loaded.items() if key not in RUNTIME_FIELDS}
    assert isinstance(loaded["asr_segments"], list)
    assert loaded["asr_segments"] == [{"audio_path": r["audio_path"]} for r in expected]
    assert [{**shared, **s} for s in loaded["asr_segments"]] == expected
    assert len(wire) < len(json.dumps(expected)) / 100


def test_processed_segments_keep_distinct_prompts_references_and_metadata():
    _, rows = sample(2)
    rows[1] = {
        **rows[1],
        "prompt": [{"role": "user", "content": "Different language hint."}],
        "asr_reference_text": "second segment reference",
        "extra_info": {**rows[1]["extra_info"], "seg_start": 2.0, "language": "French"},
    }
    before = copy.deepcopy(rows)
    dataset = ASRDataset.__new__(ASRDataset)
    dataset.ds, dataset.groups, dataset.prompt_key = rows, [[0, 1]], "prompt"
    loaded = json.loads(json.dumps(dataset[0]))
    shared = {key: value for key, value in loaded.items() if key not in RUNTIME_FIELDS}
    assert (
        [
            {**shared, **s, "extra_info": {**shared["extra_info"], **s.get("extra_info", {})}}
            for s in loaded["asr_segments"]
        ]
        == before
        == rows
    )
    assert loaded["asr_segments"][1]["prompt"] == rows[1]["prompt"]
    assert loaded["asr_segments"][1]["extra_info"] == {"seg_start": 2.0, "language": "French"}


def test_debug_detail_keeps_raw_outputs_and_shared_reference():
    row, expected = sample(200)
    texts = [f"raw hypothesis {i}" for i in range(len(expected))]
    contexts = [""] * len(expected)
    detail = json.loads(json.dumps(validation_detail(row, texts, contexts)))
    shared = {key: value for key, value in detail.items() if key not in RUNTIME_FIELDS}
    assert detail["segment_outputs"] == texts
    assert detail["previous_contexts"] == contexts
    assert detail["extra_info"] == row["extra_info"]
    assert "parent_row" not in detail and "parent_info" not in detail
    assert [{**shared, **s} for s in detail["asr_segments"]] == expected


def test_ten_times_more_chunks_does_not_repeat_full_reference_per_chunk():
    sizes = []
    for count in (100, 1000):
        row, _ = sample(count)
        sizes.append(len(json.dumps(validation_detail(row, [""] * count, [""] * count))))
    assert sizes[1] < sizes[0] * 10


def test_inline_audio_is_preserved_for_generation_but_not_serialized_in_details():
    first, second = np.arange(4, dtype=np.float32), np.arange(8, dtype=np.float32)
    rows = [{"audio": audio, "prompt": "Transcribe.", "extra_info": {}} for audio in (first, second)]
    dataset = ASRDataset.__new__(ASRDataset)
    dataset.ds, dataset.groups, dataset.prompt_key = rows, [[0, 1]], "prompt"
    row = dataset[0]
    assert row["asr_segments"][1]["audio"] is second
    assert "prompt" not in row["asr_segments"][1]
    detail = validation_detail(row, ["raw hypothesis"] * 2, [""] * 2)
    assert "audio" not in detail and all("audio" not in s for s in detail["asr_segments"])
    assert json.loads(json.dumps(detail))["segment_outputs"] == ["raw hypothesis"] * 2


@pytest.mark.parametrize("compact", [False, True])
def test_saved_detail_readers_preserve_reference_order_and_prompt_checks(compact, monkeypatch):
    from recipe.phimm.tools.inhouse_entity_report import audit_rows
    from recipe.phimm.tools.inhouse_entity_rescore import measure, rescore_row

    row, _ = sample(2)
    row["data_source"] = "eer_test"
    row["extra_info"]["entity_transcription"] = "original entity reference"
    outputs = ["first hypothesis", "second hypothesis"]
    detail = validation_detail(row, outputs, ["", ""])
    if not compact:
        detail = {
            "parent_info": row["extra_info"],
            "segment_rows": [{"audio_path": s["audio_path"], "prompt": row["prompt"]} for s in row["asr_segments"]],
            "segment_outputs": outputs,
            "previous_contexts": ["", ""],
        }
    saved = {
        "asr_detail": json.dumps(detail),
        "asr_data_source": "eer_test",
        "gts": row["reward_model"]["ground_truth"],
        "eer_n_err": 0,
        "eer_n_ref": 1,
        "dter_n_err": 0,
        "dter_n_ref": 2,
    }
    source = {
        "id": "recording",
        "data_source": "eer_test",
        "text": saved["gts"],
        "entity_transcription": "original entity reference",
        "audio_path": row["audio_path"],
    }
    prompt = row["prompt"][0]["content"]
    counts = audit_rows([saved], [source], expected_prompt=prompt)
    assert counts["parents"] == 1 and counts["segments"] == 2
    monkeypatch.setattr(measure, "_compute_eer", lambda *args, **kwargs: (0, 1, 0.0))
    assert rescore_row(saved, source, "unused")["segments"] == 2
    detail["segment_rows" if not compact else "asr_segments"][1]["prompt"] = []
    saved["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError, match="prompt"):
        audit_rows([saved], [source], expected_prompt=prompt)
    detail["segment_rows" if not compact else "asr_segments"].reverse()
    saved["asr_detail"] = json.dumps(detail)
    with pytest.raises(ValueError, match="reordered"):
        rescore_row(saved, source, "unused")

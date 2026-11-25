"""OpenLong reference histories retain scoring text, audio order, and empty turns."""

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.phimm.agentic.context_prompt import ContextPrompt
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.data.dataset import jsonl_dataset
from recipe.phimm.tools.openlong_reference_context import parent_context_rows, validation_source


def example():
    parent = "az://account/data/openasr_long_jsonl/earnings21/test/audio/0.wav"
    original = {"id": "parent", "audio_path": parent, "text": "Alpha.  Beta.", "duration": 3}
    aligned = [
        {
            "id": f"segment-{i}",
            "parent_audio_path": parent,
            "seg_index": i,
            "n_segments": 3,
            "seg_start": i,
            "seg_end": i + 1,
            "audio_path": f"audio.zip!100:200#{i}:{i + 1}",
            "text": text,
            "reference_word_start": start,
            "reference_word_end": end,
            "alignment_method": "test_reference_baseline_qwen_consensus_v1",
            "alignment_status": "needs_review",
            "alignment_flags": ["reference_marker"],
        }
        for i, (text, start, end) in enumerate([("Alpha.", 0, 1), ("", None, None), ("Beta.", 1, 2)])
    ]
    return aligned, [original]


def test_reference_context_roundtrips_through_real_validation_loader(tmp_path):
    aligned, originals = example()
    before = deepcopy(aligned), deepcopy(originals)
    parents, audit = parent_context_rows(list(reversed(aligned)), originals)
    assert (aligned, originals) == before
    assert parents[0]["text"] == originals[0]["text"]
    assert parents[0]["audio_path"] == [row["audio_path"] for row in aligned]
    assert parents[0]["segment_reference_texts"] == ["Alpha.", "", "Beta."]
    assert audit[0]["alignment_flag_counts"] == {"reference_marker": 3}
    path = tmp_path / "parents.jsonl"
    path.write_text(json.dumps(parents[0]) + "\n")
    source = validation_source("earnings21", str(path))
    data = OmegaConf.create({"version": 2609, "val_files": "val", "val_data": [[source]]})
    dataset = ASRDataset("val", None, data)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["asr_validate"] is True
    assert sample["reward_model"]["ground_truth"] == originals[0]["text"]
    segments = sample["asr_segments"]
    assert [row["audio_path"] for row in segments] == parents[0]["audio_path"]
    assert [row["asr_reference_text"] for row in segments] == parents[0]["segment_reference_texts"]
    assert all("asr_reference_context" not in row for row in segments)
    assert all(set(row) == {"audio_path", "asr_reference_text"} for row in segments)
    prompts = []
    context = ContextPrompt("text", max_history=1)
    for row in segments:
        prompts.append(context.prompt)
        context.observe(row["asr_reference_text"])
    assert prompts == ["", "Previous context: Alpha.", ""]
    metadata = [json.loads(value) for value in parents[0]["source_metadata_json"]]
    assert [row["alignment_flags"] for row in metadata] == [["reference_marker"]] * 3


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_parent",
        "duplicate_parent",
        "missing_segment",
        "duplicate_segment",
        "count",
        "gap",
        "text",
        "time",
        "selector",
        "provenance",
    ],
)
def test_invalid_reference_alignment_is_rejected(corruption):
    aligned, originals = example()
    if corruption == "missing_parent":
        originals.clear()
    elif corruption == "duplicate_parent":
        originals.append(originals[0])
    elif corruption == "missing_segment":
        aligned.pop()
    elif corruption == "duplicate_segment":
        aligned[1]["seg_index"] = 0
    elif corruption == "count":
        aligned[1]["n_segments"] = 2
    elif corruption == "gap":
        aligned[2]["reference_word_start"] = 0
    elif corruption == "text":
        aligned[0]["text"] = "Future label."
    elif corruption == "time":
        aligned[0]["seg_start"] = float("nan")
    elif corruption == "selector":
        aligned[0]["audio_path"] = "audio.zip!100:200#9:10"
    else:
        aligned[0].pop("alignment_method")
    with pytest.raises(ValueError):
        parent_context_rows(aligned, originals)


def test_context_order_matches_validation_timestamp_order():
    aligned, originals = example()
    aligned[1]["seg_start"], aligned[1]["seg_end"] = 2, 3
    aligned[2]["seg_start"], aligned[2]["seg_end"] = 1, 2
    for row in aligned:
        row["audio_path"] = f"audio#{row['seg_start']}:{row['seg_end']}"
    parents, audit = parent_context_rows(aligned, originals)
    assert parents[0]["audio_path"] == [aligned[i]["audio_path"] for i in [0, 2, 1]]
    assert parents[0]["segment_reference_texts"] == ["Alpha.", "Beta.", ""]
    assert audit[0]["reordered_by_timestamp"] is True


def test_all_history_rows_larger_than_default_arrow_block_are_loadable(tmp_path):
    path = tmp_path / "large_parent.jsonl.gz"
    row = {"segment_reference_texts": ["", "reference " * 150000]}
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    source = validation_source("earnings21", str(path))
    dataset = jsonl_dataset(source["jsonl_paths"], jsonl_block_size=source["jsonl_block_size"])
    assert dataset[0]["segment_reference_texts"] == row["segment_reference_texts"]


def test_openlong_reference_preset_uses_complete_corpora():
    directory = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/eval"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="eval_openlong_reference_context_zip")
    assert config.asr.context_aware.enable
    assert config.asr.context_aware.validation_source == "reference"
    sources = config.data.val_data[0]
    assert [source.post_process.add_field.fields.data_source for source in sources] == [
        "earnings21",
        "earnings22",
        "coraal",
    ]
    assert all(source.val_batch_size == -1 for source in sources)
    assert all("segment_reference_texts" in source.post_process.verl_format.extra_keys for source in sources)

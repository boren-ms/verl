"""Consecutive windows preserve recording order, sample budgets and ASR content."""

import copy
import gzip
import json
import sys
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.data.dataset import jsonl_dataset
from recipe.phimm.tools.offline_contiguous_merge import contiguous_merge, main, prepare


def rows_for(parent, size):
    return [
        {
            "id": f"{parent}:{i}",
            "audio_path": f"{parent}_{i}.wav",
            "original_audio_url": parent,
            "start_seconds": 0,
            "part_start_seconds": i * 2,
            "audio_duration_seconds": 2.0,
            "text": f"text{i}",
            "language": "English",
            "keywords": ["shared", str(i)],
        }
        for i in range(size)
    ]


@pytest.mark.parametrize("size", [3, 4, 5, 6, 49, 224])
def test_same_size_unique_neighbor_windows_cover_every_input(size):
    rows = rows_for("a", size) + rows_for("b", size + 1)
    # Interleaved records and reversed input must not cross parents or confuse
    # producer segment starts with offsets within those segments.
    rows = list(reversed(rows[::2])) + list(reversed(rows[1::2]))
    before = copy.deepcopy(rows)
    output = contiguous_merge(rows)
    assert rows == before
    assert output == contiguous_merge(rows)
    assert len(output) == len(rows)
    assert len({tuple(row["source_indices"]) for row in output}) == len(rows)
    assert len({row["id"] for row in output}) == len(rows)
    used = []
    for row in output:
        members = [rows[i] for i in row["source_indices"]]
        used.extend(row["source_indices"])
        assert row["merge_count"] in (2, 3)
        assert len({part["original_audio_url"] for part in members}) == 1
        starts = [part["part_start_seconds"] for part in members]
        assert starts == list(range(starts[0], starts[0] + len(members) * 2, 2))
        assert row["audio_path"] == [part["audio_path"] for part in members]
        assert row["text"] == " ".join(part["text"] for part in members)
        assert row["keywords"] == ["shared", *[str(start // 2) for start in starts]]
        assert row["language"] == "English"
        assert row["source_ids"] == [part["id"] for part in members]
        assert row["audio_length"] == 2 * len(members)
    assert set(used) == set(range(len(rows)))
    assert len(used) > len(rows)


@pytest.mark.parametrize("merge_mode", ["contiguous23", "full_recording"])
def test_normalization_precedes_merge_and_training_preserves_segment_lists(tmp_path, merge_mode):
    rows = rows_for("a", 5)
    for index, row in enumerate(rows):
        row["display_ne_tagged"] = f"hello <NE>Name{index}</NE>"
    raw = tmp_path / "input.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(raw),
        "pre_process": {
            "rename_fields": {"mappings": {"text": "display_ne_tagged"}},
            "clean_tagged_text": {},
        },
        "add_task_info": {"style": "verb", "language": "English", "lang_hint": True},
        "post_process": {"verl_format": {"prompt_key": "prompt"}},
    }
    before = copy.deepcopy(source)
    root = tmp_path / "output"
    stats = prepare(source, root, merge_mode=merge_mode)
    assert source == before
    assert stats["input_rows"] == 5
    expected_count = 1 if merge_mode == "full_recording" else 5
    assert stats["output_rows"] == expected_count
    output = [json.loads(line) for line in (root / "data.jsonl").read_text().splitlines()]
    with gzip.open(root / "data.jsonl.gz", "rt") as stream:
        assert [json.loads(line) for line in stream] == output
    [block] = yaml.safe_load((root / "train_data.yaml").read_text())
    assert "pre_process" not in block
    config = OmegaConf.create({"train_files": "train", "version": 2609, "train_data": [block]})
    training = ASRDataset("train", None, config)
    assert len(training) == expected_count
    for merged, sample in zip(output, training, strict=True):
        members = [rows[i] for i in merged["source_indices"]]
        assert merged["text"] == " ".join(f"hello Name{i}" for i in merged["source_indices"])
        assert "prompt" not in merged and "gt_output" not in merged
        assert len(sample["asr_segments"]) == merged["merge_count"]
        assert sample["reward_model"]["ground_truth"] == merged["text"]
        assert all(f"Name{i}" in sample["extra_info"]["keywords"] for i in merged["source_indices"])
        assert [json.loads(value) for value in merged["source_metadata_json"]] == members


@pytest.mark.parametrize("order_by", ["timestamp", "segment_id"])
def test_full_recordings_use_every_utterance_once_in_order(order_by):
    rows = rows_for("a", 1) + rows_for("b", 2) + rows_for("c", 7)
    if order_by == "segment_id":
        for row in rows:
            parent, index = row["id"].split(":")
            row["id"] = f"{parent}_seg{index}"
    rows.reverse()
    before = copy.deepcopy(rows)
    output = contiguous_merge(rows, order_by=order_by, merge_mode="full_recording")
    assert rows == before
    assert output == contiguous_merge(rows, order_by=order_by, merge_mode="full_recording")
    assert len(output) == len({row["merge_parent_audio_path"] for row in output}) == 3
    assert sorted(index for row in output for index in row["source_indices"]) == list(range(len(rows)))
    for merged in output:
        members = [rows[index] for index in merged["source_indices"]]
        assert len({row["original_audio_url"] for row in members}) == 1
        assert [row["part_start_seconds"] for row in members] == list(range(0, 2 * len(members), 2))
        assert merged["audio_path"] == [row["audio_path"] for row in members]
        assert merged["text"] == " ".join(row["text"] for row in members)
        assert merged["keywords"] == ["shared", *map(str, range(len(members)))]
        assert merged["language"] == "English"
        assert [part["text"] for part in merged["components"]] == [row["text"] for row in members]
        assert merged["audio_length"] == 2 * len(members)
        assert merged["merge_count"] == len(members)
        assert merged["merge_method"] == "full_recording"
        assert merged["source_ids"] == [row["id"] for row in members]
        assert [json.loads(value) for value in merged["source_metadata_json"]] == members


@pytest.mark.parametrize("size", [0, 1, 2])
def test_impossible_budgets_fail_without_crossing_recordings(size):
    with pytest.raises(ValueError, match="nonempty|at least three"):
        contiguous_merge(rows_for("a", size))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("original_audio_url", None, "original_audio_url"),
        ("start_seconds", None, "start_seconds"),
        ("audio_path", ["a.wav", "b.wav"], "scalar audio path"),
        ("audio_duration_seconds", 0, "duration"),
    ],
)
def test_missing_order_or_unsupported_inputs_fail(field, value, message):
    rows = rows_for("a", 3)
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        contiguous_merge(rows)


def test_numeric_segment_order_handles_gaps_parts_and_separate_directories():
    rows = []
    for folder in ("batch1", "batch2"):
        for suffix in ("seg10", "seg2_part1", "seg0", "seg2_part0"):
            rows.append(
                {
                    "id": f"call_{suffix}",
                    "audio_path": f"{folder}/call_{suffix}.wav",
                    "audio_length": 1,
                    "text": suffix,
                }
            )
    output = contiguous_merge(rows, order_by="segment_id")
    assert len(output) == len(rows)
    assert len({row["merge_parent_audio_path"] for row in output}) == 2
    expected = ["seg0", "seg2_part0", "seg2_part1", "seg10"]
    for row in output:
        suffixes = [value.removeprefix("call_") for value in row["source_ids"]]
        start = expected.index(suffixes[0])
        assert suffixes == expected[start : start + len(suffixes)]
        assert "source_start_seconds" not in row
        assert row["source_order_by"] == "segment_id"
        assert len({path.split("/")[0] for path in row["audio_path"]}) == 1


def test_duration_cache_uses_normalized_paths_and_keeps_raw_provenance(tmp_path):
    raw = tmp_path / "input.jsonl"
    rows = [{"id": f"call_seg{i}", "audio_path": f"audio/call_seg{i}.wav", "text": str(i)} for i in range(3)]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = {
        "dataset_name": "jsonl",
        "jsonl_paths": str(raw),
        "pre_process": {"path_map": {"field": "audio_path", "src_part": "audio/", "dst_part": "az://source/"}},
    }
    cache = {f"az://source/call_seg{i}.wav": float(i + 1) for i in range(3)}
    originals = jsonl_dataset(str(raw)).to_list()
    stats = prepare(source, tmp_path / "prepared", order_by="segment_id", duration_cache=cache)
    assert stats["order_by"] == "segment_id"
    assert stats["recordings"] == 1
    output = [json.loads(line) for line in (tmp_path / "prepared/data.jsonl").read_text().splitlines()]
    for row in output:
        assert row["audio_length"] == sum(cache[path] for path in row["audio_path"])
        assert [json.loads(value) for value in row["source_metadata_json"]] == [
            originals[i] for i in row["source_indices"]
        ]


@pytest.mark.parametrize("order_by", ["timestamp", "segment_id"])
def test_duplicate_utterance_positions_are_rejected(order_by):
    rows = rows_for("a", 3)
    for row in rows:
        row["id"] = "call_seg0"
    rows[1]["part_start_seconds"] = rows[0]["part_start_seconds"]
    with pytest.raises(ValueError, match="duplicate utterance positions"):
        contiguous_merge(rows, order_by=order_by)


def test_invalid_segment_id_fails_instead_of_guessing_order():
    with pytest.raises(ValueError, match="IDs ending in"):
        contiguous_merge(rows_for("a", 3), order_by="segment_id")


@pytest.mark.parametrize("merge_mode", ["contiguous23", "full_recording"])
def test_multi_source_cli_preserves_each_source_budget_and_infers_order(tmp_path, monkeypatch, merge_mode):
    inputs, sources, durations = {}, [], {}
    for index, rows in enumerate(
        [rows_for("timestamp", 3), [{"id": f"call_seg{i}", "audio_path": f"{i}.wav", "text": str(i)} for i in range(4)]]
    ):
        path = tmp_path / f"input_{index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        uri = f"az://source/{index}.jsonl"
        inputs[uri] = str(path)
        sources.append({"dataset_name": "jsonl", "jsonl_paths": uri})
        durations.update({row["audio_path"]: 1 for row in rows})
    source_yaml = tmp_path / "sources.yaml"
    source_yaml.write_text(yaml.safe_dump(sources))
    input_map = tmp_path / "inputs.json"
    input_map.write_text(json.dumps(inputs))
    cache = tmp_path / "durations.json"
    cache.write_text(json.dumps(durations))
    root = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "offline_contiguous_merge",
            "--source-config",
            str(source_yaml),
            "--output-dir",
            str(root),
            "--input-map",
            str(input_map),
            "--duration-cache",
            str(cache),
            "--order-by",
            "auto",
            "--merge-mode",
            merge_mode,
            "--blob-root",
            "az://prepared/test",
        ],
    )
    main()
    stats = json.loads((root / "preparation.json").read_text())
    assert stats["input_rows"] == 7
    assert stats["output_rows"] == (2 if merge_mode == "full_recording" else 7)
    assert [record["output_rows"] for record in stats["sources"]] == (
        [1, 1] if merge_mode == "full_recording" else [3, 4]
    )
    assert [record["order_by"] for record in stats["sources"]] == ["timestamp", "segment_id"]
    blocks = yaml.safe_load((root / "train_data.yaml").read_text())
    assert [block["jsonl_paths"] for block in blocks] == [
        f"az://prepared/test/source_{index:03d}/data.jsonl.gz" for index in range(2)
    ]


@pytest.mark.parametrize("context_aware", [False, True])
def test_full_earnings_preset_preserves_mix_validation_and_shared_context(context_aware):
    directory = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/earnings"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        base = compose(config_name="base_earning")
        baseline = compose(config_name="remax_earnings_short_baseline")
        suffix = "context" if context_aware else "no_context"
        config = compose(config_name=f"remax_earnings_short_contiguous23_{suffix}")
    OmegaConf.resolve(config)
    assert len(config.data.train_data) == 5
    assert all("contiguous23" in source.jsonl_paths for source in config.data.train_data)
    assert all(not source.get("pre_process") for source in config.data.train_data)
    assert config.data.get("random_segment") is config.data.get("random_merge") is None
    assert "offline_data" not in config
    assert "train_data" not in base.data
    assert "mix" not in base.data
    assert not any(key.startswith("_earning_train") or key.startswith("_quartr_") for key in base)
    assert config.data.mix == baseline.data.mix
    assert config.data.val_data == base.data.val_data
    assert config.data.train_batch_size == base.data.train_batch_size == 128
    assert config.data.max_audio_dur is None
    assert config.trainer.total_training_steps == base.trainer.total_training_steps == 394
    assert config.asr.expected_training_rows == 25240
    assert config.asr.context_aware is context_aware
    assert config.algorithm.adv_estimator == "remax"
    assert config.actor_rollout_ref.rollout.n == 8
    base.trainer.experiment_name = config.trainer.experiment_name
    OmegaConf.resolve(base)
    assert config.actor_rollout_ref == base.actor_rollout_ref

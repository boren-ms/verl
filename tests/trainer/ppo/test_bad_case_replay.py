# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.bad_case_replay import BadCaseReplay, ReplayMetrics


def _objects(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _batch(ids, width=3):
    count = len(ids)
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(1, width + 1).expand(count, -1).clone(),
            "attention_mask": torch.ones((count, width), dtype=torch.long),
            "position_ids": torch.arange(width).expand(count, -1).clone(),
        },
        non_tensors={
            "id": np.asarray(ids, dtype=object),
            "data_source": np.full(count, "train", dtype=object),
            "extra_info": _objects([{"details": [str(item)]} for item in ids]),
            "raw_prompt_ids": _objects([list(range(width)) for _ in ids]),
            "multi_modal_inputs": _objects(
                [{"audio": np.arange(4, dtype=np.float32), "embedding": torch.ones(2, requires_grad=True)} for _ in ids]
            ),
        },
        meta_info={"settings": {"name": "original"}},
    )


def _controller(size=5, *, data_parallel_size=1, **overrides):
    config = {
        "enable": True,
        "capacity": 400,
        "replay_size": 0,
        "sampling": "worst",
        "selection": {"weights": {"word": 1}},
        "seed": 17,
    }
    config.update(overrides)
    return BadCaseReplay(config, size, pad_token_id=99, data_parallel_size=data_parallel_size)


def _observe(controller, ids, scores, *, width=3):
    combined, context = controller.prepare(_batch(ids, width))
    stats = controller.update(context, {"word": scores})
    return combined, context, stats


def _stored(controller):
    return {entry.inputs.non_tensor_batch["id"][0]: entry.score for entry in controller._entries.values()}


def test_disabled_is_identity_and_preserves_nominal_loader_size():
    controller = BadCaseReplay({"enable": False}, 100)
    raw = _batch(["a"])
    combined, context = controller.prepare(raw)
    assert combined is raw and context is None
    assert controller.fresh_batch_size == 100 and controller.target_replay_count == 0
    assert controller.update(context, {}) == {}
    assert len(controller) == 0


def test_nominal_100_starts_with_90_and_then_adds_10_replay_prompts():
    controller = _controller(100, replay_size=10)
    first, context, stats = _observe(controller, list(range(90)), np.arange(90))
    assert controller.fresh_batch_size == len(first) == 90
    assert controller.target_replay_count == 10
    assert stats["replay/candidates"] == len(controller) == 90
    combined, context = controller.prepare(_batch(list(range(100, 190))))
    assert len(combined) == 100
    assert len(controller) == 80
    assert combined.non_tensor_batch["id"].tolist()[-10:] == list(range(10))
    assert context.replay_mask.sum() == 10
    assert context.fresh_count == 90


def test_insufficient_buffer_does_not_refill_fresh_rows():
    controller = _controller(10, replay_size=4)
    _, _, stats = _observe(controller, list(range(6)), [0, 1, np.nan, np.nan, np.nan, np.nan])
    assert stats["replay/candidates"] == 2
    combined, context = controller.prepare(_batch(list(range(10, 16))))
    assert len(combined) == 8
    assert context.fresh_count == 6 and context.replay_mask.sum() == 2
    assert len(controller) == 0
    stats = controller.update(context, {"word": np.arange(8)})
    assert stats["replay/prompt_count"] == 8
    assert stats["replay/candidates"] == 8


@pytest.mark.parametrize("size,finite_count", [(16, 16), (112, 112), (128, 128), (128, 7), (128, 0), (512, 512)])
def test_all_finite_rows_participate_before_global_capacity_limit(size, finite_count):
    controller = _controller(size)
    scores = np.full(size, np.nan)
    scores[:finite_count] = np.arange(finite_count)
    _, _, stats = _observe(controller, list(range(size)), scores)
    expected = min(controller.capacity, finite_count)
    assert stats["replay/candidates"] == finite_count
    assert len(controller) == expected
    assert stats["replay/invalid_scores"] == size - finite_count
    assert _stored(controller) == {i: i for i in range(expected)}


@pytest.mark.parametrize("sampling", ["random", "worst"])
@pytest.mark.parametrize("available,expected", [(0, 0), (1, 0), (15, 0), (16, 16), (17, 16), (31, 16), (32, 32)])
def test_partial_replay_draw_keeps_whole_dp_groups_without_dropping_residents(sampling, available, expected):
    controller = _controller(128, replay_size=32, data_parallel_size=16, sampling=sampling)
    scores = np.full(96, np.nan)
    scores[:available] = np.arange(available)
    _observe(controller, list(range(96)), scores)
    assert len(controller) == available

    combined, context = controller.prepare(_batch(list(range(1000, 1096))))

    assert len(combined) == 96 + expected
    assert len(combined) % 16 == 0
    assert context.fresh_count == 96
    assert context.target_replay_count == 32
    assert context.replay_mask.sum() == expected
    assert len(controller) == available - expected
    replay_ids = combined.non_tensor_batch["id"][96:].tolist()
    assert len(set(replay_ids)) == expected
    assert not set(replay_ids) & set(_stored(controller))
    if sampling == "worst":
        assert replay_ids == list(range(expected))


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_data_parallel_size_is_rejected(size):
    with pytest.raises(ValueError, match="data_parallel_size"):
        _controller(128, replay_size=16, data_parallel_size=size)


def test_replay_and_nominal_sizes_must_be_divisible_by_data_parallel_size():
    with pytest.raises(ValueError, match="replay_size must be divisible"):
        _controller(128, replay_size=12, data_parallel_size=16)
    with pytest.raises(ValueError, match="nominal_batch_size must be divisible"):
        _controller(130, replay_size=16, data_parallel_size=16)


def test_small_batch_considers_all_valid_inputs_even_with_no_replay_draw():
    controller = _controller(4, replay_size=0)
    _, _, stats = _observe(controller, list("abcd"), [-4, -3, -2, -1])
    assert controller.target_replay_count == 0
    assert stats["replay/candidates"] == 4
    assert len(controller) == 4


def test_raw_weighted_metrics_keep_negative_values_and_skip_zero_weight_keys():
    controller = _controller(5, capacity=1, selection={"weights": {"word": 7, "char": 3, "absent": 0}})
    _, context = controller.prepare(_batch(list("abcde")))
    stats = controller.update(context, {"word": [-2, 0, 1, 2, 3], "char": [1, -8, 0, 0, 0]})
    assert _stored(controller) == {"b": pytest.approx(-2.4)}
    assert stats["replay/score_min"] == pytest.approx(-2.4)


def test_missing_metric_errors_without_silently_using_scalar_reward():
    controller = _controller(5)
    _, context = controller.prepare(_batch(list("abcde")))
    with pytest.raises(ValueError, match="'word'.*missing"):
        controller.update(context, {"score": [1] * 5})
    assert len(controller) == 0


def test_score_metric_requires_an_explicit_scalar_component():
    controller = _controller(5, capacity=1, selection={"weights": {"score": 1}})
    _, context = controller.prepare(_batch(list("abcde")))
    controller.update(context, {"score": [2, -8, 0, 1, 3]})
    assert _stored(controller) == {"b": -8}
    controller = _controller(5, selection={"weights": {"score": 1}})
    _, context = controller.prepare(_batch(list("abcde")))
    with pytest.raises(ValueError, match="'score'.*missing"):
        controller.update(context, {})
    assert len(controller) == 0


def test_nonfinite_samples_are_skipped_and_all_finite_samples_are_candidates():
    controller = _controller(10)
    _, _, stats = _observe(controller, list(range(10)), [np.nan, np.inf, -np.inf, -7, -6, -5, 0, 1, 2, 3])
    assert _stored(controller) == {3: -7, 4: -6, 5: -5, 6: 0, 7: 1, 8: 2, 9: 3}
    assert stats["replay/candidates"] == 7
    assert stats["replay/invalid_scores"] == 3
    assert stats["replay/score_count"] == 7
    controller = _controller(5)
    _, _, stats = _observe(controller, list("abcde"), [np.nan] * 5)
    assert len(controller) == 0
    assert stats["replay/invalid_scores"] == 5
    assert "replay/score_mean" not in stats


def test_capacity_retains_global_lowest_scores_instead_of_fifo():
    controller = _controller(3, capacity=3)
    _observe(controller, list("abc"), [-100, -4, -3])
    _, _, stats = _observe(controller, list("def"), [-2, -1, -50])
    assert _stored(controller) == {"a": -100, "f": -50, "b": -4}
    assert stats["replay/evicted"] == 3


def test_current_observation_refreshes_resident_before_global_capacity_selection():
    controller = _controller(2, capacity=2)
    _observe(controller, list("ab"), [-10, 0])
    _observe(controller, list("cd"), [-5, 0])
    _, _, stats = _observe(controller, list("ae"), [10, -6])
    assert _stored(controller) == {"e": -6, "c": -5}
    assert stats["replay/evicted"] == 1


def test_duplicate_ids_use_latest_current_score_not_historical_minimum():
    controller = _controller(5)
    _, _, stats = _observe(controller, ["a", "b", "a", "c", "d"], [-10, -4, 3, 0, 1])
    assert _stored(controller)["a"] == 3
    assert len(controller) == 4
    assert stats["replay/deduplicated"] == 1
    _, _, stats = _observe(controller, list("abcde"), [9, 8, 7, 6, 5])
    assert _stored(controller)["a"] == 9
    assert stats["replay/deduplicated"] == 4


def test_deduplication_precedes_global_capacity_cutoff():
    controller = _controller(5, capacity=3)
    _, _, stats = _observe(controller, ["a", "a", "a", "b", "c"], [-10, -9, -8, -7, -6])
    assert _stored(controller) == {"a": -8, "b": -7, "c": -6}
    assert stats["replay/candidates"] == 5
    assert stats["replay/deduplicated"] == 2
    assert stats["replay/evicted"] == 0


def test_worst_sampling_pops_entries_and_preserves_stable_ties():
    controller = _controller(10, replay_size=2)
    _observe(controller, list("abcdefgh"), [-2, -2, -2, -1, 0, 1, 2, 3])
    combined, _ = controller.prepare(_batch(list("ijklmnop")))
    assert combined.non_tensor_batch["id"].tolist()[-2:] == ["a", "b"]
    assert set(_stored(controller)) == set("cdefgh")


def test_random_sampling_is_reproducible_and_has_no_replacement():
    controllers = [_controller(10, replay_size=3, sampling="random") for _ in range(2)]
    draws = []
    for controller in controllers:
        _observe(controller, list("abcdefg"), list(range(7)))
        mixed, _ = controller.prepare(_batch(list("hijklmn")))
        draws.append(mixed.non_tensor_batch["id"].tolist()[-3:])
        assert len(controller) == 4
    assert draws[0] == draws[1]
    assert len(set(draws[0])) == 3


def test_popped_replay_reenters_only_if_it_survives_global_capacity_selection():
    controller = _controller(10, replay_size=2, capacity=2)
    _observe(controller, list("abcdefgh"), list(range(8)))
    _, context = controller.prepare(_batch(list("ijklmnop")))
    assert len(controller) == 0  # Both resident entries were popped.
    stats = controller.update(context, {"word": [-8, -7, -6, -5, -4, -3, -2, -1, 100, 101]})
    assert not {"a", "b"} & set(_stored(controller))
    assert stats["replay/reinsert_candidates"] == 2
    assert stats["replay/reinserted"] == 0
    _, context = controller.prepare(_batch(list("qrstuvwx")))
    scores = [0] * context.fresh_count + [-100] * int(context.replay_mask.sum())
    stats = controller.update(context, {"word": scores})
    assert stats["replay/reinsert_candidates"] == 2
    assert stats["replay/reinserted"] == 2
    assert stats["replay/reinsert_rate"] == 1


def test_reinsert_candidates_may_be_evicted_by_other_bad_samples():
    controller = _controller(5, capacity=2, replay_size=1)
    _observe(controller, list("abcd"), [-10, -9, 0, 1])
    _, context = controller.prepare(_batch(list("efgh")))
    stats = controller.update(context, {"word": [-20, -19, 0, 1, -5]})
    assert stats["replay/replayed"] == 1
    assert stats["replay/reinsert_candidates"] == 1
    assert stats["replay/reinserted"] == 0
    assert stats["replay/reinsert_rate"] == 0


def test_nonfinite_replay_is_not_automatically_reinserted():
    controller = _controller(5, capacity=2, replay_size=1)
    _observe(controller, list("abcd"), [-4, -3, -2, -1])
    _, context = controller.prepare(_batch(list("efgh")))
    assert "a" not in _stored(controller)
    stats = controller.update(context, {"word": [0, 1, 2, 3, np.nan]})
    assert _stored(controller) == {"b": -3, "e": 0}
    assert stats["replay/reinsert_candidates"] == stats["replay/reinserted"] == 0
    assert stats["replay/invalid_scores"] == 1


def test_snapshot_and_stored_audio_metadata_are_isolated_from_training_mutations():
    controller = _controller(5, replay_size=1)
    raw = _batch(list("abcd"))
    mixed, context = controller.prepare(raw)
    mixed.batch["input_ids"][0, 0] = 999
    mixed.non_tensor_batch["extra_info"][0]["details"].append("mutated")
    mixed.non_tensor_batch["multi_modal_inputs"][0]["audio"][0] = 999
    mixed.meta_info["settings"]["name"] = "changed"
    mixed.non_tensor_batch["uid"] = np.asarray(list("wxyz"), dtype=object)
    mixed.batch["responses"] = torch.ones(4, 3, dtype=torch.long)
    controller.update(context, {"word": [-4, -3, -2, -1]})
    replayed, _ = controller.prepare(_batch(list("efgh")))
    assert replayed.batch["input_ids"][-1, 0] == 1
    assert replayed.non_tensor_batch["extra_info"][-1] == {"details": ["a"]}
    assert replayed.non_tensor_batch["multi_modal_inputs"][-1]["audio"][0] == 0
    embedding = replayed.non_tensor_batch["multi_modal_inputs"][-1]["embedding"]
    assert not embedding.requires_grad and embedding.device.type == "cpu"
    assert "uid" not in replayed.non_tensor_batch and "responses" not in replayed.batch
    assert context.raw_inputs.meta_info["settings"]["name"] == "original"
    assert context.raw_inputs.non_tensor_batch["multi_modal_inputs"][0]["audio"][0] == 0


@pytest.mark.parametrize("generated", ["responses", "uid", "reward_baselines_word", "advantages"])
def test_prepare_rejects_post_generation_inputs(generated):
    controller = _controller(5)
    raw = _batch(list("abcde"))
    raw.non_tensor_batch[generated] = np.asarray([0] * 5)
    with pytest.raises(ValueError, match="pre-generation"):
        controller.prepare(raw)


def test_merge_left_pads_sequences_and_preserves_raw_multimodal_fields():
    controller = _controller(5, replay_size=1)
    _observe(controller, list("abcd"), [-4, -3, -2, -1], width=2)
    fresh = _batch(list("efgh"), width=4)
    fresh.non_tensor_batch["optional"] = _objects([{"x": 1}] * 4)
    mixed, context = controller.prepare(fresh)
    assert mixed.batch["input_ids"].shape == (5, 4)
    assert mixed.batch["input_ids"][-1].tolist() == [99, 99, 1, 2]
    assert mixed.batch["attention_mask"][-1].tolist() == [0, 0, 1, 1]
    assert mixed.batch["position_ids"][-1].tolist() == [0, 0, 0, 1]
    assert mixed.non_tensor_batch["raw_prompt_ids"][-1] == [0, 1]
    assert mixed.non_tensor_batch["optional"][-1] is None
    assert context.replay_mask.tolist() == [False] * 4 + [True]


def test_multiaxis_position_ids_keep_axis_order_during_left_padding():
    controller = _controller(5, replay_size=1)
    short = _batch(list("abcd"), width=2)
    short.batch["position_ids"] = short.batch["position_ids"].unsqueeze(1).expand(-1, 3, -1).clone()
    _, context = controller.prepare(short)
    controller.update(context, {"word": [-4, -3, -2, -1]})
    fresh = _batch(list("efgh"), width=4)
    fresh.batch["position_ids"] = fresh.batch["position_ids"].unsqueeze(1).expand(-1, 3, -1).clone()
    mixed, _ = controller.prepare(fresh)
    assert mixed.batch["position_ids"].shape == (5, 3, 4)
    assert mixed.batch["position_ids"][-1].tolist() == [[0, 0, 0, 1]] * 3


def test_source_namespaces_same_raw_id_and_fingerprint_handles_padding():
    controller = _controller(2)
    raw = _batch(["same", "same"])
    raw.non_tensor_batch["data_source"][1] = "other"
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [-2, -1]})
    assert len(controller) == 2
    controller = _controller(1)
    raw = _batch(["same"], width=2)
    raw.non_tensor_batch.pop("id")
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [-2]})
    raw.batch["input_ids"] = torch.tensor([[99, 99, 1, 2]])
    raw.batch["attention_mask"] = torch.tensor([[0, 0, 1, 1]])
    raw.batch["position_ids"] = torch.tensor([[0, 0, 0, 1]])
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [3]})
    assert len(controller) == 1
    assert next(iter(controller._entries.values())).score == 3


@pytest.mark.parametrize("audio_key", ["audio_path", "audio_chunk", "audio_file"])
def test_shared_parent_id_does_not_merge_distinct_audio_segments(audio_key):
    controller = _controller(2)
    raw = _batch(["parent", "parent"])
    raw.non_tensor_batch[audio_key] = np.asarray(["recording.wav#0:10", "recording.wav#10:20"], dtype=object)
    raw.non_tensor_batch["index"] = np.zeros(2, dtype=np.int64)
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [-8, -2]})
    assert len(controller) == 2
    assert context.identities[0] != context.identities[1]
    # Reobserving the same two children updates their own latest scores.
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [-1, -5]})
    assert len(controller) == 2
    assert sorted(entry.score for entry in controller._entries.values()) == [-5, -1]


def test_segment_coordinates_disambiguate_same_id_and_audio_reference():
    controller = _controller(2)
    raw = _batch(["parent", "parent"])
    raw.non_tensor_batch["audio_path"] = np.asarray(["recording.wav"] * 2, dtype=object)
    raw.non_tensor_batch["extra_info"] = _objects(
        [
            {"id": "parent", "seg_index": 0, "seg_start": 0, "seg_end": 10},
            {"id": "parent", "seg_index": 1, "seg_start": 10, "seg_end": 20},
        ]
    )
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [-2, -1]})
    assert len(controller) == 2


@pytest.mark.parametrize("placeholder_id", [None, "unknown_id"])
def test_default_zero_index_does_not_collapse_rows_without_original_identifiers(placeholder_id):
    controller = _controller(2)
    raw = _batch(["first", "second"])
    raw.non_tensor_batch.pop("id")
    raw.non_tensor_batch["index"] = np.zeros(2, dtype=np.int64)
    raw.non_tensor_batch["extra_info"] = _objects([{"id": placeholder_id}] * 2)
    raw.non_tensor_batch["multi_modal_inputs"][1]["audio"] += 10
    _, context = controller.prepare(raw)
    controller.update(context, {"word": [-2, -1]})
    assert len(controller) == 2


@pytest.mark.parametrize(
    "config,match",
    [
        ({"capacity": 0}, "capacity"),
        ({"capacity": 2.5}, "capacity"),
        ({"replay_size": 5}, "replay_size"),
        ({"replay_size": -1}, "replay_size"),
        ({"replay_size": 1.5}, "replay_size"),
        ({"replay_size": True}, "replay_size"),
        ({"replay_size": math.nan}, "replay_size"),
        ({"sampling": "fifo"}, "sampling"),
        ({"selection": {"weights": {}}}, "nonempty"),
        ({"selection": {"weights": {"word": -1}}}, "non-negative"),
        ({"selection": {"weights": {"word": math.inf}}}, "non-negative"),
        ({"selection": {"weights": {"word": 0}}}, "positive sum"),
    ],
)
def test_invalid_configuration_is_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        _controller(5, **config)


def test_score_shape_must_match_actual_prompt_count():
    controller = _controller(5)
    _, context = controller.prepare(_batch(list("abcde")))
    with pytest.raises(ValueError, match="shape"):
        controller.update(context, {"word": [1, 2]})
    with pytest.raises(ValueError, match="shape"):
        controller.update(context, {"word": [[1], [2], [3], [4], [5]]})
    with pytest.raises(ValueError, match="fresh prompts"):
        controller.prepare(_batch(list("abcd")))


def test_metrics_accumulate_counts_weight_score_means_and_keep_last_buffer_size():
    metrics = ReplayMetrics()
    assert metrics.as_dict() == {}
    metrics.update(
        {
            "replay/replayed": 1,
            "replay/reinserted": 1,
            "replay/reinsert_candidates": 1,
            "replay/reinsert_rate": 1,
            "replay/buffer_size": 8,
            "replay/score_count": 2,
            "replay/score_mean": -2,
            "replay/score_min": -3,
            "replay/score_max": -1,
        }
    )
    metrics.update(
        {
            "replay/replayed": 9,
            "replay/reinserted": 0,
            "replay/reinsert_candidates": 2,
            "replay/reinsert_rate": 0,
            "replay/buffer_size": 5,
            "replay/score_count": 1,
            "replay/score_mean": 2,
            "replay/score_min": 2,
            "replay/score_max": 2,
        }
    )
    result = metrics.as_dict()
    assert result["replay/reinsert_rate"] == 0.1
    assert result["replay/reinsert_candidates"] == 3
    assert result["replay/buffer_size"] == 5
    assert result["replay/score_mean"] == pytest.approx(-2 / 3)
    assert result["replay/score_min"] == -3 and result["replay/score_max"] == 2
    metrics.reset()
    assert metrics.as_dict() == {}

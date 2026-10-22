"""CPU checks for aligning teacher-forced GT scores with the actor temperature."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.rollout_log_probs import align_gt_rollout_log_probs


def _make_batch():
    row_ids = torch.arange(6)
    responses = torch.stack([row_ids + 10, row_ids + 20, row_ids + 30], dim=-1)
    return DataProto.from_dict(
        tensors={
            "row_id": row_ids,
            "responses": responses,
            "input_ids": torch.cat([row_ids[:, None], responses], dim=-1),
            "attention_mask": torch.ones((6, 4), dtype=torch.long),
            "position_ids": torch.arange(4).expand(6, -1).clone(),
            "gt_response_mask": torch.tensor([True, False, True, False, True, False]),
            "rollout_log_probs": -100.0 - responses.float(),
            "old_log_probs": -200.0 - responses.float(),
        },
        non_tensors={"uid": np.array([f"prompt-{i}" for i in row_ids.tolist()], dtype=object)},
        meta_info={"temperature": 1.2, "untouched": "original metadata"},
    )


class _Actor:
    def __init__(self, world_size=4, temperature=1.2):
        self.world_size = world_size
        self.temperature = temperature
        self.calls = []

    @staticmethod
    def scores(data):
        return -data.batch["responses"].float() / 10.0

    def compute_log_prob(self, data):
        assert len(data) % self.world_size == 0
        assert data.batch["gt_response_mask"].all()
        assert data.non_tensor_batch["uid"].tolist() == [f"prompt-{row_id}" for row_id in data.batch["row_id"].tolist()]
        self.calls.append(data)
        return DataProto.from_dict(
            tensors={
                "old_log_probs": self.scores(data),
                "entropys": torch.full_like(data.batch["responses"], 9.0, dtype=torch.float32),
            },
            meta_info={"temperature": self.temperature},
        )


@pytest.mark.parametrize("world_size", [1, 3, 4, 8])
def test_reordered_gt_subset_is_scored_padded_and_restored(world_size):
    data = _make_batch()
    data.reorder(torch.tensor([4, 1, 5, 0, 3, 2]))
    gt_mask = data.batch["gt_response_mask"]
    rollout_before = data.batch["rollout_log_probs"].clone()
    old_before = data.batch["old_log_probs"].clone()
    row_ids_before = data.batch["row_id"].clone()
    actor = _Actor(world_size=world_size)

    assert align_gt_rollout_log_probs(data, actor) is None

    assert len(actor.calls) == 1
    submitted = actor.calls[0]
    expected_padded_size = ((3 + world_size - 1) // world_size) * world_size
    assert len(submitted) == expected_padded_size
    assert submitted.batch["row_id"][:3].tolist() == [4, 0, 2]
    assert set(submitted.batch["row_id"].tolist()) <= {0, 2, 4}
    assert len(data) == 6
    torch.testing.assert_close(data.batch["row_id"], row_ids_before)
    torch.testing.assert_close(data.batch["rollout_log_probs"][gt_mask], actor.scores(data)[gt_mask])
    torch.testing.assert_close(data.batch["rollout_log_probs"][~gt_mask], rollout_before[~gt_mask])
    torch.testing.assert_close(data.batch["old_log_probs"], old_before)
    assert "entropys" not in data.batch
    assert data.meta_info["temperature"] == 1.2
    assert data.meta_info["untouched"] == "original metadata"


def test_one_gt_row_is_padded_across_all_actor_ranks():
    data = _make_batch()
    data.batch["gt_response_mask"][:] = False
    data.batch["gt_response_mask"][3] = True
    before = data.batch["rollout_log_probs"].clone()
    actor = _Actor(world_size=8)

    align_gt_rollout_log_probs(data, actor)

    assert len(actor.calls) == 1
    assert actor.calls[0].batch["row_id"].tolist() == [3] * 8
    expected = before.clone()
    expected[3] = actor.scores(data)[3]
    torch.testing.assert_close(data.batch["rollout_log_probs"], expected)


@pytest.mark.parametrize("condition", ["missing_gt_mask", "no_gt_rows", "missing_rollout_probs"])
def test_unneeded_gt_alignment_does_not_call_actor(condition):
    data = _make_batch()
    if condition == "missing_gt_mask":
        data.batch.pop("gt_response_mask")
    elif condition == "no_gt_rows":
        data.batch["gt_response_mask"][:] = False
    else:
        data.batch.pop("rollout_log_probs")
    before = {key: value.clone() for key, value in data.batch.items()}
    actor = _Actor()

    assert align_gt_rollout_log_probs(data, actor) is None

    assert not actor.calls
    assert set(data.batch.keys()) == set(before)
    for key, value in before.items():
        torch.testing.assert_close(data.batch[key], value)


def test_mismatched_actor_temperature_rejects_scores_without_partial_replacement():
    data = _make_batch()
    before = data.batch["rollout_log_probs"].clone()
    actor = _Actor(temperature=1.0)

    with pytest.raises(ValueError, match="[Tt]emperature"):
        align_gt_rollout_log_probs(data, actor)

    assert len(actor.calls) == 1
    torch.testing.assert_close(data.batch["rollout_log_probs"], before)


def test_actor_metadata_mutations_do_not_leak_into_original_batch():
    class MetadataMutatingActor(_Actor):
        def compute_log_prob(self, data):
            data.meta_info.pop("untouched")
            data.meta_info["micro_batch_size"] = 2
            data.meta_info["max_token_len"] = 4096
            data.meta_info["use_dynamic_bsz"] = True
            data.meta_info["temperature"] = self.temperature
            return super().compute_log_prob(data)

    data = _make_batch()
    metadata_before = deepcopy(data.meta_info)
    actor = MetadataMutatingActor()

    align_gt_rollout_log_probs(data, actor)

    assert len(actor.calls) == 1
    assert data.meta_info == metadata_before


@pytest.mark.parametrize("temperature", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_rejects_before_scoring(temperature):
    data = _make_batch()
    data.meta_info["temperature"] = temperature
    before = data.batch["rollout_log_probs"].clone()
    actor = _Actor()

    with pytest.raises(ValueError, match="[Tt]emperature"):
        align_gt_rollout_log_probs(data, actor)

    assert not actor.calls
    torch.testing.assert_close(data.batch["rollout_log_probs"], before)


@pytest.mark.parametrize("mask", [torch.ones(6, dtype=torch.int64), torch.ones((6, 1), dtype=torch.bool)])
def test_invalid_gt_mask_rejects_before_scoring(mask):
    data = _make_batch()
    data.batch["gt_response_mask"] = mask
    before = data.batch["rollout_log_probs"].clone()
    actor = _Actor()

    with pytest.raises(ValueError, match="gt_response_mask"):
        align_gt_rollout_log_probs(data, actor)

    assert not actor.calls
    torch.testing.assert_close(data.batch["rollout_log_probs"], before)


def test_mismatched_score_shape_rejects_without_partial_replacement():
    class WrongShapeActor(_Actor):
        def compute_log_prob(self, data):
            scored = super().compute_log_prob(data)
            scored.batch["old_log_probs"] = scored.batch["old_log_probs"][:, :-1]
            return scored

    data = _make_batch()
    before = data.batch["rollout_log_probs"].clone()
    actor = WrongShapeActor()

    with pytest.raises(ValueError, match="match the selected rollout rows"):
        align_gt_rollout_log_probs(data, actor)

    assert len(actor.calls) == 1
    torch.testing.assert_close(data.batch["rollout_log_probs"], before)

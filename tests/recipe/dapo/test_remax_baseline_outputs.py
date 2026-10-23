import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.remax_baseline import clear_baseline_outputs


@pytest.mark.parametrize('greedy_reason', ['stop', 'length'])
def test_sampled_finish_reason_replaces_greedy_after_reward(greedy_reason):
    prompts = DataProto.from_dict(
        tensors={'input_ids': torch.tensor([[1]])},
        non_tensors={'id': np.array(['utterance'], dtype=object)},
    )
    baseline = DataProto.from_dict(
        tensors={'responses': torch.tensor([[2, 3]])},
        non_tensors={'rollout_finish_reason': np.array([greedy_reason], dtype=object)},
    )
    batch = prompts.union(baseline)
    # Greedy reward computation must still see its own reason.
    assert batch.non_tensor_batch['rollout_finish_reason'].tolist() == [greedy_reason]
    batch.batch['reward_baselines'] = torch.tensor([0.8])
    clear_baseline_outputs(batch, baseline)
    sampled = DataProto.from_dict(
        tensors={'responses': torch.tensor([[3, 4], [5, 0]])},
        non_tensors={'rollout_finish_reason': np.array(['length', 'stop'], dtype=object)},
    )
    merged = batch.repeat(2, interleave=True).union(sampled)
    assert merged.non_tensor_batch['rollout_finish_reason'].tolist() == ['length', 'stop']
    assert merged.non_tensor_batch['id'].tolist() == ['utterance', 'utterance']
    assert merged.batch['reward_baselines'].tolist() == pytest.approx([0.8, 0.8])
    assert merged.batch['responses'].tolist() == [[3, 4], [5, 0]]
    # Cleanup does not mutate the baseline used for disagreement masks.
    assert baseline.non_tensor_batch['rollout_finish_reason'].tolist() == [greedy_reason]


def test_cleanup_supports_rollouts_without_native_finish_reasons():
    baseline = DataProto.from_dict(tensors={'responses': torch.tensor([[2]])})
    batch = DataProto.from_dict(tensors={'id': torch.tensor([1])}).union(baseline)
    clear_baseline_outputs(batch, baseline)
    assert set(batch.batch.keys()) == {'id'}

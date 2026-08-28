# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from unittest.mock import patch

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


def _trainer_with_val_reward(group_segment: bool, reward_fn) -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create({"val_reward": {"group_segment": group_segment}})
    trainer._val_reward_fn = reward_fn
    return trainer


def test_v1_val_reward_groups_and_orders_segments_by_parent():
    calls = []

    def reward_fn(**kwargs):
        calls.append(kwargs)
        return {"score": len(calls), "wer": 0.25 * len(calls)}

    trainer = _trainer_with_val_reward(group_segment=True, reward_fn=reward_fn)
    result = trainer._recompute_val_reward(
        outputs=["second", "other", "first"],
        reward_models=[
            {"ground_truth": "parent ref"},
            {"ground_truth": "other ref"},
            {"ground_truth": "parent ref"},
        ],
        data_sources=["inhouse", "openml", "inhouse"],
        extra_infos=[
            {"parent_audio_path": "parent.wav", "seg_start": 2.0},
            {"audio_path": "other.wav#0:1", "seg_start": 0.0},
            {"parent_audio_path": "parent.wav", "seg_start": 1.0},
        ],
    )

    scores, reward_extra_infos = result
    assert len(calls) == 2
    assert calls[0]["solution_str"] == "first second"
    assert calls[0]["ground_truth"] == "parent ref"
    assert calls[0]["data_source"] == "inhouse"
    assert calls[1]["solution_str"] == "other"
    assert scores == [1, 2, 1]
    assert reward_extra_infos == [
        {"score": 1, "wer": 0.25},
        {"score": 2, "wer": 0.5},
        {"score": 1, "wer": 0.25},
    ]


def test_v1_val_reward_scores_segments_individually_when_grouping_disabled():
    hypotheses = []

    def reward_fn(**kwargs):
        hypotheses.append(kwargs["solution_str"])
        return 3.0

    trainer = _trainer_with_val_reward(group_segment=False, reward_fn=reward_fn)
    scores, reward_extra_infos = trainer._recompute_val_reward(
        outputs=["first", "second"],
        reward_models=[{"ground_truth": "ref"}, {"ground_truth": "ref"}],
        data_sources=["inhouse", "inhouse"],
        extra_infos=[
            {"parent_audio_path": "parent.wav", "seg_start": 1.0},
            {"parent_audio_path": "parent.wav", "seg_start": 2.0},
        ],
    )

    assert hypotheses == ["first", "second"]
    assert scores == [3.0, 3.0]
    assert reward_extra_infos == [{"score": 3.0}, {"score": 3.0}]


def test_v1_val_reward_supports_async_reward_function():
    async def reward_fn(**kwargs):
        await asyncio.sleep(0)
        return {"score": len(kwargs["solution_str"])}

    trainer = _trainer_with_val_reward(group_segment=False, reward_fn=reward_fn)
    scores, reward_extra_infos = trainer._recompute_val_reward(
        outputs=["hello"],
        reward_models=[{"ground_truth": "ref"}],
        data_sources=["openml"],
        extra_infos=[{}],
    )

    assert scores == [5]
    assert reward_extra_infos == [{"score": 5}]

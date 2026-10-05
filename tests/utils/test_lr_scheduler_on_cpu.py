# Copyright 2026 Microsoft Corporation
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

import copy
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


def make_optimizer(lr=0.1, second_group=False):
    groups = [{"params": [torch.nn.Parameter(torch.ones(1))], "lr": lr}]
    if second_group:
        groups.append({"params": [torch.nn.Parameter(torch.ones(1))], "lr": lr / 10})
    return torch.optim.SGD(groups, lr=lr)


def make_scheduler(optimizer, style, **kwargs):
    if style == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_training_steps=100, **kwargs)
    return get_constant_schedule_with_warmup(optimizer, **kwargs)


def collect_lrs(optimizer, scheduler, steps):
    lrs = []
    for _ in range(steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return lrs


@pytest.mark.parametrize("style", ["constant", "cosine"])
def test_cycles_restart_with_multiple_peaks_and_reuse_last(style):
    optimizer = make_optimizer()
    scheduler = make_scheduler(optimizer, style, num_warmup_steps=0, cycle_steps=[4], cycle_lrs=[0.1, 0.05, 0.02])
    lrs = collect_lrs(optimizer, scheduler, 20)
    for step, actual in enumerate(lrs):
        cycle_index, local_step = divmod(step, 4)
        peak = [0.1, 0.05, 0.02][min(cycle_index, 2)]
        expected = peak if style == "constant" else peak * (1 + math.cos(math.pi * local_step / 4)) / 2
        assert actual == pytest.approx(expected)
    assert lrs[0::4] == pytest.approx([0.1, 0.05, 0.02, 0.02, 0.02])


def test_cosine_floor_is_relative_to_each_cycle_peak():
    optimizer = make_optimizer()
    scheduler = make_scheduler(
        optimizer, "cosine", num_warmup_steps=0, cycle_steps=[4], cycle_lrs=[0.1, 0.05], min_lr_ratio=0.2
    )
    lrs = collect_lrs(optimizer, scheduler, 12)
    for step, actual in enumerate(lrs):
        peak = 0.1 if step < 4 else 0.05
        expected = peak * (0.6 + 0.4 * math.cos(math.pi * (step % 4) / 4))
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("style", ["constant", "cosine"])
@pytest.mark.parametrize("peaks", [[0.1], [0.1, 0.05, 0.02, 0.01]])
def test_variable_cycle_lengths_reuse_final_duration_independently_of_lrs(style, peaks):
    optimizer = make_optimizer()
    scheduler = make_scheduler(optimizer, style, num_warmup_steps=0, cycle_steps=[3, 2], cycle_lrs=peaks)
    lrs = collect_lrs(optimizer, scheduler, 11)
    for step, actual in enumerate(lrs):
        if step < 3:
            cycle_index, local_step, length = 0, step, 3
        else:
            later_cycle, local_step = divmod(step - 3, 2)
            cycle_index, length = 1 + later_cycle, 2
        peak = peaks[min(cycle_index, len(peaks) - 1)]
        expected = peak if style == "constant" else peak * (1 + math.cos(math.pi * local_step / length)) / 2
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("style", ["constant", "cosine"])
def test_warmup_restarts_every_cycle(style):
    optimizer = make_optimizer()
    scheduler = make_scheduler(optimizer, style, num_warmup_steps=2, cycle_steps=[6], cycle_lrs=[0.1, 0.05])
    lrs = collect_lrs(optimizer, scheduler, 18)
    assert lrs[0::6] == [0, 0, 0]
    assert lrs[1::6] == pytest.approx([0.05, 0.025, 0.025])
    assert lrs[2::6] == pytest.approx([0.1, 0.05, 0.05])


@pytest.mark.parametrize("style", ["constant", "cosine"])
def test_cycles_without_peak_list_reuse_original_lr(style):
    optimizer = make_optimizer()
    scheduler = make_scheduler(optimizer, style, num_warmup_steps=0, cycle_steps=[4])
    lrs = collect_lrs(optimizer, scheduler, 12)
    assert lrs[:4] == pytest.approx(lrs[4:8])
    assert lrs[:4] == pytest.approx(lrs[8:])


def test_cycle_lrs_are_absolute_and_preserve_parameter_group_ratios():
    optimizer = make_optimizer(lr=0.2, second_group=True)
    scheduler = make_scheduler(optimizer, "constant", num_warmup_steps=0, cycle_steps=[2], cycle_lrs=[0.1, 0.04])
    assert scheduler.get_last_lr() == pytest.approx([0.1, 0.01])
    collect_lrs(optimizer, scheduler, 2)
    assert scheduler.get_last_lr() == pytest.approx([0.04, 0.004])


@pytest.mark.parametrize("style", ["constant", "cosine"])
def test_cycle_checkpoint_resume_preserves_position(style):
    optimizer = make_optimizer()
    kwargs = {"num_warmup_steps": 1, "cycle_steps": [5, 3], "cycle_lrs": [0.1, 0.04, 0.02]}
    scheduler = make_scheduler(optimizer, style, **kwargs)
    collect_lrs(optimizer, scheduler, 7)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    expected = collect_lrs(optimizer, scheduler, 15)

    restored_optimizer = make_optimizer()
    restored_scheduler = make_scheduler(restored_optimizer, style, **kwargs)
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    actual = collect_lrs(restored_optimizer, restored_scheduler, 15)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("style", ["constant", "cosine"])
def test_noncyclic_schedule_preserves_original_behavior(style):
    optimizer = make_optimizer()
    scheduler = make_scheduler(optimizer, style, num_warmup_steps=2)
    lrs = collect_lrs(optimizer, scheduler, 100)
    for step, actual in enumerate(lrs):
        if step < 2:
            multiplier = step / 2
        elif style == "constant":
            multiplier = 1
        else:
            multiplier = (1 + math.cos(math.pi * (step - 2) / 98)) / 2
        assert actual == pytest.approx(0.1 * multiplier)


@pytest.mark.parametrize("style", ["constant", "cosine"])
@pytest.mark.parametrize("warmup_steps", [-1, 4, 5])
def test_invalid_cycle_warmup_is_rejected(style, warmup_steps):
    with pytest.raises(ValueError, match="num_warmup_steps"):
        make_scheduler(make_optimizer(), style, num_warmup_steps=warmup_steps, cycle_steps=[6, 4])


@pytest.mark.parametrize("style", ["constant", "cosine"])
def test_raw_config_invalid_cycle_is_rejected(style):
    with pytest.raises(ValueError, match="cycle_steps"):
        make_scheduler(make_optimizer(), style, num_warmup_steps=0, cycle_steps=[0])
    with pytest.raises(ValueError, match="requires cycle_steps"):
        make_scheduler(make_optimizer(), style, num_warmup_steps=0, cycle_lrs=[0.1])
    with pytest.raises(ValueError, match="base LR"):
        make_scheduler(make_optimizer(lr=0), style, num_warmup_steps=0, cycle_steps=[4], cycle_lrs=[0.1])


def test_zero_cycle_peak_disables_lr_for_that_cycle():
    optimizer = make_optimizer()
    scheduler = make_scheduler(optimizer, "constant", num_warmup_steps=0, cycle_steps=[2], cycle_lrs=[0.1, 0])
    assert collect_lrs(optimizer, scheduler, 6) == pytest.approx([0.1, 0.1, 0, 0, 0, 0])


def test_hydra_actor_and_critic_accept_cycle_configuration():
    root = Path(__file__).parents[2]
    overrides = [
        "actor_rollout_ref.actor.optim.cycle_steps=[200,100]",
        "actor_rollout_ref.actor.optim.cycle_lrs=[5e-6,3e-6]",
        "critic.optim.cycle_steps=[100]",
        "critic.optim.cycle_lrs=[1e-5]",
    ]
    with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    actor_config = instantiate(config.actor_rollout_ref.actor.optim)
    critic_config = instantiate(config.critic.optim)
    assert actor_config.cycle_steps == [200, 100]
    assert actor_config.cycle_lrs == [5e-6, 3e-6]
    assert critic_config.cycle_steps == [100]
    assert critic_config.cycle_lrs == [1e-5]
    optimizer = make_optimizer(lr=5e-6)
    scheduler = make_scheduler(
        optimizer,
        "constant",
        num_warmup_steps=0,
        cycle_steps=config.actor_rollout_ref.actor.optim.cycle_steps,
        cycle_lrs=config.actor_rollout_ref.actor.optim.cycle_lrs,
    )
    assert scheduler.get_last_lr()[0] == pytest.approx(5e-6)
    assert OmegaConf.is_list(config.actor_rollout_ref.actor.optim.cycle_lrs)


def test_generated_actor_and_critic_cycle_defaults_match_source():
    root = Path(__file__).parents[2] / "verl/trainer/config"
    source = OmegaConf.load(root / "optim/fsdp.yaml")
    generated = OmegaConf.load(root / "_generated_ppo_trainer.yaml")
    for config in (generated.actor_rollout_ref.actor.optim, generated.critic.optim):
        assert config.cycle_steps == source.cycle_steps is None
        assert config.cycle_lrs == source.cycle_lrs is None

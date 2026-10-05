# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.workers.config.optimizer import FSDPOptimizerConfig


class TestFSDPOptimizerConfigCPU:
    def test_default_configuration(self):
        config = FSDPOptimizerConfig(lr=0.1)
        assert config.min_lr_ratio is None
        assert config.warmup_style == "constant"
        assert config.num_cycles == 0.5
        assert config.cycle_steps is None
        assert config.cycle_lrs is None

    @pytest.mark.parametrize("warmup_style", ["constant", "cosine"])
    def test_valid_warmup_styles(self, warmup_style):
        config = FSDPOptimizerConfig(warmup_style=warmup_style, lr=0.1)
        assert config.warmup_style == warmup_style

    def test_invalid_warmup_style(self):
        with pytest.raises((ValueError, AssertionError)):
            FSDPOptimizerConfig(warmup_style="invalid_style", lr=0.1)

    @pytest.mark.parametrize("num_cycles", [0.1, 1.0, 2.5])
    def test_num_cycles_configuration(self, num_cycles):
        config = FSDPOptimizerConfig(num_cycles=num_cycles, lr=0.1)
        assert config.num_cycles == num_cycles

    @pytest.mark.parametrize("warmup_style", ["constant", "cosine"])
    def test_cycle_configuration(self, warmup_style):
        config = FSDPOptimizerConfig(
            warmup_style=warmup_style, cycle_steps=[200, 100], cycle_lrs=[5e-6, 3e-6, 1e-6], lr=5e-6
        )
        assert config.cycle_steps == [200, 100]
        assert config.cycle_lrs == [5e-6, 3e-6, 1e-6]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cycle_steps": 0},
            {"cycle_steps": -1},
            {"cycle_steps": 1.5},
            {"cycle_steps": True},
            {"cycle_steps": []},
            {"cycle_steps": [0]},
            {"cycle_steps": [10, -1]},
            {"cycle_steps": [10, 1.5]},
            {"cycle_steps": [True]},
            {"cycle_lrs": [0.1]},
            {"cycle_steps": [10], "cycle_lrs": []},
            {"cycle_steps": [10], "cycle_lrs": [-0.1]},
            {"cycle_steps": [10], "cycle_lrs": [float("nan")]},
            {"cycle_steps": [10], "cycle_lrs": [float("inf")]},
            {"cycle_steps": [10], "cycle_lrs": [True]},
            {"cycle_steps": [10], "cycle_lrs": "0.1"},
        ],
    )
    def test_invalid_cycle_configuration(self, kwargs):
        with pytest.raises(ValueError):
            FSDPOptimizerConfig(**kwargs)

    def test_cycle_lrs_default_to_configured_lr(self):
        config = FSDPOptimizerConfig(lr=5e-6, cycle_steps=[200, 100])
        assert config.cycle_lrs == [5e-6]

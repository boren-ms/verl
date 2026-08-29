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
"""Long-audio ASR evaluation compatibility entry point.

The reference implementation uses the legacy
``ActorRolloutRefWorker.generate_sequences`` path. verl-mirror no longer has
that worker, so generation uses the supported server-based
:mod:`recipe.phimm.long_asr_rollout` pipeline while preserving the reference
command and config layout.
"""

import asyncio
from pprint import pprint

import hydra
from omegaconf import OmegaConf

from recipe.phimm.asr_rollout import init_ray
from recipe.phimm.long_asr_rollout import _run_long_asr_rollout


@hydra.main(
    config_path="config/eval",
    config_name="long_eval_mixlang_fy26q2",
    version_base=None,
)
def main(config):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(expression, {}, {}))

    init_ray(config)
    pprint(OmegaConf.to_container(config, resolve=True))
    asyncio.run(_run_long_asr_rollout(config))


if __name__ == "__main__":
    main()

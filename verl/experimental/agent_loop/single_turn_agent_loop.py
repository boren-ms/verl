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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, _convert_audio_messages_to_text, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer.chat_template import apply_chat_template
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs — prefer pre-loaded data from dataset
        pre_loaded_mm = kwargs.get("multi_modal_data")
        if pre_loaded_mm and isinstance(pre_loaded_mm, dict):
            # Dataset stores audio as {"audio": [(wav, sr), ...]}
            audios = pre_loaded_mm.get("audios") or pre_loaded_mm.get("audio")
            images = pre_loaded_mm.get("images") or pre_loaded_mm.get("image")
            videos = pre_loaded_mm.get("videos") or pre_loaded_mm.get("video")
            multi_modal_data = {"audios": audios, "images": images, "videos": videos}
        else:
            multi_modal_data = await self.process_multi_modal_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. Build the initial prompt with Continuous Token. Audio prompts retain
        # their rendered text so vLLM can tokenize model-specific placeholders.
        prompt_text = None
        self._assert_mm_supported(bool(images or videos))
        prompt_ids = await self.ct_build_initial_tokens(
            messages,
            images=images,
            videos=videos,
            audios=audios,
        )

        # For audio multimodal, pass the raw prompt text so vLLM can tokenize
        # with its own tokenizer that has audio placeholder tokens.
        if audios is not None:
            # Prefer pre-rendered prompt from dataset (has correct audio placeholders)
            prompt_text = kwargs.get("full_prompt_text") or kwargs.get("full_prompts")
            if prompt_text is None:
                prompt_text = getattr(self, "_last_raw_prompt", None)
            if prompt_text is None:
                # Reconstruct from messages if nothing else is available
                _chat_obj = self.processor if getattr(self.processor, "chat_template", None) else self.tokenizer
                text_messages = _convert_audio_messages_to_text(messages)
                prompt_text = apply_chat_template(
                    _chat_obj, text_messages, add_generation_prompt=True, tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                prompt_text = prompt_text.replace("<think>\n\n</think>\n\n", "")

        # 3. generate sequences (same pattern as main_asr_gen.py)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                prompt_text=prompt_text,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            output.token_ids,
            [],
            [] if output.log_probs else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output

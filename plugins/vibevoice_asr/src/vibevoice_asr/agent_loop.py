"""Native offline VibeVoice rollout on Main's existing ASR ReMax agent."""

import asyncio
from uuid import uuid4

from recipe.phimm.agentic.agent_loop import ASRAgentLoop
from recipe.phimm.agentic.context_prompt import context_options
from recipe.phimm.agentic.sampling import sampling_params_for_row

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.utils.profiler import simple_timer
from vibevoice_asr.decode import load_waveform
from vibevoice_asr.reward import transcript_text
from vibevoice_asr.transcription import collapse_audio_tokens


class VibeVoiceAgentLoop(ASRAgentLoop):
    transcript_text = staticmethod(transcript_text)

    async def generate_segments(self, rows, sampling_params, *, validate):
        if len(rows) != 1 or context_options(self.config.asr).get("enable", False):
            raise ValueError("VibeVoice RL uses independent clips and its native offline prompt")
        return await super().generate_segments(rows, sampling_params, validate=validate)

    async def generate_row(self, row, sampling_params, *, validate, previous_context=""):
        waveform = await asyncio.to_thread(load_waveform, row)
        maximum = (
            self.config.asr.get("validation_max_audio_seconds", self.config.data.max_audio_dur)
            if validate and row.get("data_source") == "davidai_test"
            else self.config.data.max_audio_dur
        )
        if maximum is not None and len(waveform) / 24000 > maximum:
            raise ValueError("VibeVoice RL audio exceeds the configured bound; references must never be truncated")
        inputs = await asyncio.to_thread(
            self.processor, audio=[waveform], sampling_rate=24000, return_tensors="pt", padding=True
        )
        prompt_ids = inputs.pop("input_ids")[0].tolist()
        inputs.pop("attention_mask", None)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("VibeVoice native audio prompt exceeds rollout.prompt_length")
        # Upstream vLLM stores the supplied waveform directly; it does not run
        # the HF audio normalizer. Reuse exactly the normalized HF waveform.
        rollout_waveform = inputs["speech_tensors"][0].detach().cpu().numpy()
        inference_ids = collapse_audio_tokens(
            prompt_ids,
            start_id=self.processor.speech_start_id,
            end_id=self.processor.speech_end_id,
            audio_id=self.tokenizer.convert_tokens_to_ids("<|AUDIO|>"),
        )
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=inference_ids,
                # Only full-recording evaluation needs the long response budget.
                # ReMax's short-clip greedy baseline keeps the policy's token cap.
                sampling_params=sampling_params_for_row(
                    self.rollout_config,
                    sampling_params,
                    validate=validate and row.get("data_source") == "davidai_test",
                ),
                audio_data=[(rollout_waveform, 24000)],
            )
        if not output.token_ids:
            raise ValueError("VibeVoice rollout returned an empty token sequence")
        if sampling_params.get("logprobs") and (
            output.log_probs is None or len(output.log_probs) != len(output.token_ids)
        ):
            raise ValueError("VibeVoice rollout must return one logprob per sampled token")
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids,
            response_mask=[1] * len(output.token_ids),
            response_logprobs=output.log_probs,
            multi_modal_data={"multi_modal_inputs": dict(inputs)},
            metrics=metrics,
            num_turns=2,
            extra_fields={**output.extra_fields, "asr_previous_context": ""},
        )

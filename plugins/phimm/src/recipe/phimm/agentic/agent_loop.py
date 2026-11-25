"""Qwen ASR rollout using main's agent/LLM-client and TransferQueue interfaces."""

import asyncio
import copy
import json
from uuid import uuid4

import ray
import torch

from recipe.phimm.agentic.context_prompt import (
    DEFAULT_CONTEXT_MODE,
    DEFAULT_MAX_BIAS_TERMS,
    ContextPrompt,
    context_options,
    context_source,
)
from recipe.phimm.agentic.details import RUNTIME_FIELDS, validation_detail
from recipe.phimm.agentic.sampling import sampling_params_for_row
from recipe.phimm.agentic.scoring import score_response
from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.audio import load_audio
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, DictConfigWrap, register
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQBase
from verl.utils.profiler import simple_timer


@register("asr")
class ASRAgentLoop(AgentLoopBase):
    async def generate_row(self, row, sampling_params, *, validate, previous_context=""):
        """Preserve the source recipe's chat template and audio processor."""
        audio = await asyncio.to_thread(load_audio, row, self.config.data.max_audio_dur)
        chat = self.processor if getattr(self.processor, "chat_template", None) else self.tokenizer
        messages = row[self.config.data.prompt_key]
        if previous_context:
            messages = copy.deepcopy(messages)
            for message in reversed(messages):
                if message["role"] == "user":
                    message["content"] += f"\n{previous_context}"
                    break
            else:
                raise ValueError("ASR previous context requires a user prompt")
        text = chat.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **dict(self.config.data.get("apply_chat_template_kwargs", {})),
        )
        inputs = await asyncio.to_thread(self.processor, text=[text], audios=[audio], return_tensors="pt")
        prompt_ids = inputs["input_ids"][0].tolist()
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError(f"Audio prompt has {len(prompt_ids)} tokens; limit={self.rollout_config.prompt_length}")
        # Retain the acoustic features for actor postprocessing instead of
        # extracting them again after generation. vLLM still receives waveform.
        inputs.pop("input_ids")
        inputs.pop("attention_mask", None)
        # vLLM V1 replaces the text's <audio> marker itself, just as in the
        # source rollout's raw_prompt_ids. Keep HF-expanded IDs for training.
        inference_prompt_ids = self.tokenizer.encode(text, add_special_tokens=False)
        params = sampling_params_for_row(self.rollout_config, sampling_params, validate=validate)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=inference_prompt_ids,
                sampling_params=params,
                audio_data=[audio],
            )
        if not output.token_ids:
            raise ValueError("ASR generation returned an empty response")
        if sampling_params.get("logprobs") and (
            output.log_probs is None or len(output.log_probs) != len(output.token_ids)
        ):
            raise ValueError("ASR generation must return one rollout logprob per sampled token")
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids,
            response_mask=[1] * len(output.token_ids),
            response_logprobs=output.log_probs,
            multi_modal_data={"multi_modal_inputs": dict(inputs)},
            metrics=metrics,
            num_turns=2,
            extra_fields={**output.extra_fields, "asr_previous_context": previous_context},
        )

    async def generate_segments(self, rows, sampling_params, *, validate, training_context=None, global_steps=None):
        """Decode with no context, fixed reference context, or generated history."""
        if not rows:
            raise ValueError("ASR requires at least one segment")
        # Greedy baselines use validation decoding limits but remain training trajectories.
        training = not validate if training_context is None else training_context
        config = context_options(getattr(self.config, "asr", {}))
        source = context_source(config, training=training, global_steps=global_steps)

        if source == "generated":
            outputs, texts = [], []
            context = self._context_prompt(config)
            for row in rows:
                output = await self.generate_row(
                    row, sampling_params, validate=validate, previous_context=context.prompt
                )
                text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
                outputs.append(output)
                texts.append(text)
                context.observe(text)
            return outputs, texts

        if source == "reference":
            prompts = self._reference_prompts(config, rows)
        else:
            prompts = [""] * len(rows)
        outputs = await asyncio.gather(
            *[
                self.generate_row(row, sampling_params, validate=validate, previous_context=prompt)
                for row, prompt in zip(rows, prompts, strict=True)
            ]
        )
        texts = [self.tokenizer.decode(output.response_ids, skip_special_tokens=True) for output in outputs]
        return outputs, texts

    @staticmethod
    def _context_prompt(config):
        return ContextPrompt(
            config.get("mode", DEFAULT_CONTEXT_MODE),
            prefix=config.get("prefix"),
            max_history=config.get("max_history", 1),
            max_terms=config.get("max_terms", DEFAULT_MAX_BIAS_TERMS),
            filter_common=config.get("filter_common", True),
            filter_numbers=config.get("filter_numbers", True),
            filter_fillers=config.get("filter_fillers", True),
            extra_common_words=config.get("common_words", ()),
        )

    def _reference_prompts(self, config, rows):
        if any(not isinstance(row.get("asr_reference_text"), str) for row in rows):
            raise ValueError(
                "Reference context requires one string segment_reference_texts entry per segment; "
                "prepare the dataset with recipe.phimm.tools.offline_reference_context"
            )
        context = self._context_prompt(config)
        prompts = []
        for row in rows:
            # Observe only after constructing this segment's prompt: no current/future label leakage.
            prompts.append(context.prompt)
            context.observe(row["asr_reference_text"])
        return prompts

    async def greedy_baseline(self, row):
        params = dict(temperature=0.0, top_p=1.0, top_k=-1, logprobs=False)
        shared = {key: value for key, value in row.items() if key not in RUNTIME_FIELDS}
        rows = [
            {**shared, **segment, "extra_info": {**(row.get("extra_info") or {}), **(segment.get("extra_info") or {})}}
            for segment in row.get("asr_segments", [{}])
        ]
        # validate=True selects greedy decoding limits. This is still a
        # training trajectory: it shares fixed reference inputs with sampled
        # training, or builds its own generated history under the default policy.
        _, texts = await self.generate_segments(
            rows, params, validate=True, training_context=True, global_steps=row.get("global_steps")
        )
        text = "\n".join(texts)
        info = copy.deepcopy(row.get("extra_info") or {})
        info.pop("greedy_reward", None)
        info["greedy_hyp"] = text
        result = await asyncio.to_thread(score_response, self.config, row, texts, extra_info=info)
        return text, result

    async def run(self, sampling_params, **kwargs):
        validate = kwargs["asr_validate"]
        segments = kwargs["asr_segments"]
        row = {key: value for key, value in kwargs.items() if key not in RUNTIME_FIELDS}
        rows = [
            {**row, **segment, "extra_info": {**(row.get("extra_info") or {}), **(segment.get("extra_info") or {})}}
            for segment in segments
        ]
        outputs, texts = await self.generate_segments(
            rows, sampling_params, validate=validate, global_steps=kwargs.get("global_steps")
        )
        output = outputs[0]
        if validate:
            info = copy.deepcopy(rows[0].get("extra_info") or {})
            info["n_segments"] = len(rows)
        else:
            info = kwargs["extra_info"]
        score_row = dict(rows[0])
        if "reward_model" in kwargs:
            score_row["reward_model"] = kwargs["reward_model"]
        # Train and validation both score raw newly generated responses against
        # the parent reference, preserving boundaries for language/format checks.
        result = await asyncio.to_thread(score_response, self.config, score_row, texts, extra_info=info)
        reward_info = {
            **result,
            "asr_data_source": rows[0]["data_source"],
            "asr_detail": json.dumps(
                validation_detail(
                    kwargs,
                    texts,
                    [o.extra_fields.get("asr_previous_context", "") for o in outputs],
                ),
                ensure_ascii=False,
            ),
        }
        if validate:
            # Aggregate text is only for validation display, after raw scoring.
            response = get_hyp_text(texts, version=self.config.data.version)
            output.response_ids = self.tokenizer.encode(response, add_special_tokens=False)
            if not output.response_ids:
                # TransferQueue attaches the precomputed reward to the last token.
                # Keep empty hypotheses in validation; EOS decodes to the same
                # empty display text without changing their raw scoring inputs.
                output.response_ids = [self.tokenizer.eos_token_id]
            output.response_mask = [1] * len(output.response_ids)
            output.response_logprobs = None
            output.reward_score = float(result["score"])
            output.extra_fields["reward_extra_info"] = reward_info
            output.metrics.generate_sequences = sum(o.metrics.generate_sequences for o in outputs)
            return output
        # Main's multi-output contract keeps independent prompts/features under
        # uid_session_segment, and applies the same outcome reward to every segment.
        for index, output in enumerate(outputs):
            output.reward_score = float(result["score"])
            output.extra_fields["reward_extra_info"] = reward_info
            output.extra_fields["asr_segment_index"] = index
            output.extra_fields["asr_num_segments"] = len(outputs)
        return outputs if len(outputs) > 1 else outputs[0]


@ray.remote
class ASRWorker(AgentLoopWorkerTQBase):
    def _compute_multi_modal_inputs(self, output, input_ids):
        return (output.multi_modal_data or {}).get("multi_modal_inputs")

    def make_agent(self):
        return ASRAgentLoop(
            trainer_config=DictConfigWrap(self.config),
            server_manager=self.llm_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            hf_model_type=self.hf_model_type,
        )

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        if not trajectory["validate"] and self.config.algorithm.adv_estimator == "remax":
            try:
                text, result = await self.make_agent().greedy_baseline(prompt)
                prompt = dict(prompt)
                prompt["extra_info"] = copy.deepcopy(prompt.get("extra_info") or {})
                prompt["extra_info"].update(greedy_hyp=text, greedy_reward=result)
                prompt["reward_baselines"] = torch.tensor(float(result["score"]), dtype=torch.float32)
            except Exception:
                # Settle the prompt even if baseline generation fails before sampled tasks start.
                import transfer_queue as tq

                await tq.async_kv_put(key=prompt["uid"], partition_id="train", tag={"status": "failure"})
                raise
        return await super()._run_prompt(prompt, sampling_params, trajectory, trace)


class ASRManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ASRWorker

"""Per-utterance DAPO manager preserving keyword envelopes and termination info."""

from collections import Counter

from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager


@register("asr_keywords")
class ASRKeywordsRewardManager(DAPORewardManager):
    def __call__(self, data, return_dict=False):
        # np.repeat can alias dictionaries between sampled candidates. Detach
        # only aliases; preserve unique validation dictionaries for generation dumps.
        extras = data.non_tensor_batch.get("extra_info", [])
        identities = Counter(id(value) for value in extras)
        for i, value in enumerate(extras):
            if identities[id(value)] > 1:
                extras[i] = dict(value)
        for i in range(len(data)):
            item = data[i]
            extra = item.non_tensor_batch.get("extra_info")
            if extra is None:
                raise ValueError("keyword reward requires extra_info")
            prompt_length = item.batch["prompts"].shape[-1]
            length = int(item.batch["attention_mask"][prompt_length:].sum())
            tokens = item.batch["responses"][:length]
            # vLLM can consume a stop string without returning EOS. A short
            # response is normal stop; a full budget without EOS is conservative length.
            eos = self.tokenizer.eos_token_id
            ended = bool(length and int(tokens[-1]) == eos)
            at_limit = length >= item.batch["responses"].shape[-1]
            native_reason = item.non_tensor_batch.get("rollout_finish_reason")
            if native_reason:
                extra["generation_finish_reason"] = native_reason if native_reason in {"stop", "length"} else "timeout"
                extra["generation_finish_reason_source"] = "vllm"
            else:
                extra["generation_finish_reason"] = "length" if at_limit and not ended else "stop"
                extra["generation_finish_reason_source"] = "inferred"
        return super().__call__(data, return_dict=return_dict)

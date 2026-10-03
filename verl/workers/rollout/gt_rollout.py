"""Teacher-forced GT response scoring with the active vLLM rollout engine."""

import math
from copy import deepcopy
from typing import Any


def generate_with_gt_responses(
    engine: Any,
    vllm_inputs: list[dict],
    sampling_params: Any,
    gt_responses: dict[int, list[int]],
    calculate_log_probs: bool,
    lora_requests: Any = None,
) -> tuple[list[list[int]], list[list[float]]]:
    """Generate sampled responses and score prompt+GT in the same engine call.

    prompt_logprobs=0 includes each observed token's probability. Use the
    returned prompt IDs to locate GT after multimodal placeholder expansion.
    GT scoring bypasses prefix-cache reads so every prompt token is scored.
    """
    if any(i < 0 or i >= len(vllm_inputs) for i in gt_responses):
        raise ValueError("GT response indices must refer to rollout inputs")
    request_inputs = list(vllm_inputs)
    request_params = sampling_params
    if gt_responses:
        gt_params = deepcopy(sampling_params)
        gt_params.max_tokens = 1
        if calculate_log_probs:
            gt_params.prompt_logprobs = 0
            gt_params.skip_reading_prefix_cache = True
        request_params = [sampling_params] * len(vllm_inputs)
        for i, response in gt_responses.items():
            if not response:
                raise ValueError("GT response must include at least EOS")
            request_inputs[i] = {
                **vllm_inputs[i],
                "prompt_token_ids": list(vllm_inputs[i]["prompt_token_ids"]) + response,
            }
            request_params[i] = gt_params
    outputs = engine.generate(
        prompts=request_inputs,
        sampling_params=request_params,
        lora_request=lora_requests,
        use_tqdm=False,
    )
    if len(outputs) != len(vllm_inputs):
        raise ValueError("vLLM must return one request output per input")

    responses = []
    log_probs = []
    for i, output in enumerate(outputs):
        if gt_responses and len(output.outputs) != 1:
            raise ValueError("GT rollout requires pre-repeated prompts and one vLLM output per input")
        gt_response = gt_responses.get(i)
        for sample in output.outputs:
            response = gt_response if gt_response is not None else list(sample.token_ids)
            responses.append(response)
            if not calculate_log_probs:
                continue
            if gt_response is None:
                token_probs = sample.logprobs
            else:
                if output.prompt_token_ids is None or list(output.prompt_token_ids[-len(response) :]) != response:
                    raise ValueError("vLLM scoring output does not end with the requested GT tokens")
                if output.prompt_logprobs is None or len(output.prompt_logprobs) != len(output.prompt_token_ids):
                    raise ValueError("vLLM did not return complete GT prompt log probabilities")
                token_probs = output.prompt_logprobs[-len(response) :]
            if any(probs is None or token not in probs for token, probs in zip(response, token_probs, strict=True)):
                raise ValueError("vLLM did not return a probability for every response token")
            row = [probs[token].logprob for token, probs in zip(response, token_probs, strict=True)]
            if gt_response is not None and not all(math.isfinite(prob) for prob in row):
                raise FloatingPointError("Non-finite vLLM GT response log probabilities")
            log_probs.append(row)
    return responses, log_probs

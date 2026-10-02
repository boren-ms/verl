from types import SimpleNamespace

import pytest
import torch

from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.gt_rollout import generate_with_gt_responses


def token_log_prob(prefix, token):
    logits = -((torch.arange(128).float() - sum(prefix) % 128) / 10).square()
    return torch.log_softmax(logits, dim=-1)[token].item()


class MockRolloutEngine:
    def __init__(self):
        self.calls = []
        self.outputs = []

    def generate(self, prompts, sampling_params, lora_request, use_tqdm):
        self.calls.append((prompts, sampling_params, lora_request))
        self.outputs = []
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        for i, (prompt, param) in enumerate(zip(prompts, params, strict=True)):
            prefix = list(prompt["prompt_token_ids"])
            if "multi_modal_data" in prompt:
                prefix = [111, 112] + prefix
            if param.max_tokens == 1:
                probs = None
                if param.prompt_logprobs == 0:
                    probs = [None] + [
                        {token: SimpleNamespace(logprob=token_log_prob(prefix[:j], token))}
                        for j, token in enumerate(prefix)
                        if j > 0
                    ]
                output = SimpleNamespace(
                    prompt_token_ids=prefix, prompt_logprobs=probs,
                    outputs=[SimpleNamespace(token_ids=[42])],
                )
            else:
                tokens = [i + 1, 99]
                output = SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=tokens,
                            logprobs=[{token: SimpleNamespace(logprob=-0.1 * (j + 1))}
                                      for j, token in enumerate(tokens)],
                        )
                    ]
                )
            self.outputs.append(output)
        return self.outputs


def run_rollout(engine, prompts, gt_responses, calculate_log_probs=True, max_tokens=5):
    sampled_params = SimpleNamespace(n=1, temperature=1.2, max_tokens=max_tokens, prompt_logprobs=None)
    responses, probs = generate_with_gt_responses(
        engine,
        prompts,
        sampled_params,
        gt_responses,
        calculate_log_probs=calculate_log_probs,
        lora_requests=list(range(len(prompts))),
    )
    assert sampled_params.max_tokens == max_tokens
    return responses, probs


@pytest.mark.parametrize("gt_tokens", [[7], [7, 8], [7, 8, 9], list(range(10))])
@pytest.mark.parametrize("multimodal", [False, True])
def test_one_generate_call_scores_gt_and_preserves_sampled_rows(gt_tokens, multimodal):
    engine = MockRolloutEngine()
    prompts = [{"prompt_token_ids": [10 + i, 11 + i]} for i in range(6)]
    if multimodal:
        for prompt in prompts:
            prompt["multi_modal_data"] = {"audio": "mock-audio"}
    gt_response = gt_tokens[:4] + [99]
    responses, probs = run_rollout(engine, prompts, {0: gt_response, 3: gt_response})

    assert len(engine.calls) == 1
    request_inputs, params, loras = engine.calls[0]
    assert len(request_inputs) == len(responses) == len(probs) == 6
    assert loras == list(range(6))
    for i in range(6):
        assert prompts[i]["prompt_token_ids"] == [10 + i, 11 + i]
        if i in (0, 3):
            assert request_inputs[i]["prompt_token_ids"] == prompts[i]["prompt_token_ids"] + gt_response
            assert params[i].max_tokens == 1
            assert params[i].prompt_logprobs == 0
            assert responses[i] == gt_response
            output = engine.outputs[i]
            offset = len(output.prompt_token_ids) - len(gt_response)
            assert probs[i] == [
                token_log_prob(output.prompt_token_ids[: offset + j], token)
                for j, token in enumerate(gt_response)
            ]
        else:
            assert request_inputs[i] is prompts[i]
            assert params[i].max_tokens == 5
            assert responses[i] == [i + 1, 99]
            assert probs[i] == [-0.1, -0.2]
        assert len(probs[i]) == len(responses[i])
        if multimodal:
            assert request_inputs[i]["multi_modal_data"] is prompts[i]["multi_modal_data"]
    assert pad_2d_list_to_length(responses, 0, max_length=5).shape == (6, 5)
    assert torch.isfinite(pad_2d_list_to_length(probs, -1, max_length=5)).all()


@pytest.mark.parametrize("max_tokens", [1, 8, 768])
def test_gt_decodes_only_one_token_without_mutating_caller_params(max_tokens):
    engine = MockRolloutEngine()
    run_rollout(engine, [{"prompt_token_ids": [10]}], {0: [7, 99]}, max_tokens=max_tokens)
    assert len(engine.calls) == 1
    assert engine.calls[0][1][0].max_tokens == 1


@pytest.mark.parametrize("calculate_log_probs", [False, True])
def test_gt_sampling_params_only_override_token_limit_and_prompt_logprobs(calculate_log_probs):
    engine = MockRolloutEngine()
    original_params = {
        "n": 1,
        "temperature": 1.2,
        "max_tokens": 5,
        "prompt_logprobs": None,
        "logprobs": 0,
        "top_p": 0.9,
        "stop_token_ids": [99],
        "extra_args": {"ngram_size": 3, "allowed_tokens": [7, 99]},
        "detokenize": False,
    }
    sampled_params = SimpleNamespace(**original_params)
    generate_with_gt_responses(
        engine,
        [{"prompt_token_ids": [10]}, {"prompt_token_ids": [20]}],
        sampled_params,
        {0: [7, 99]},
        calculate_log_probs=calculate_log_probs,
    )
    gt_params, non_gt_params = engine.calls[0][1]
    assert gt_params is not sampled_params
    assert non_gt_params is sampled_params
    assert vars(gt_params) == {
        **original_params,
        "max_tokens": 1,
        "prompt_logprobs": 0 if calculate_log_probs else None,
    }
    assert vars(sampled_params) == original_params
    assert gt_params.stop_token_ids is not sampled_params.stop_token_ids
    assert gt_params.extra_args is not sampled_params.extra_args
    assert gt_params.extra_args["allowed_tokens"] is not sampled_params.extra_args["allowed_tokens"]
    gt_params.stop_token_ids.append(100)
    gt_params.extra_args["ngram_size"] = 4
    gt_params.extra_args["allowed_tokens"].append(100)
    assert sampled_params.stop_token_ids == [99]
    assert sampled_params.extra_args == {"ngram_size": 3, "allowed_tokens": [7, 99]}


def test_disabled_logprobs_keeps_single_call_and_gt_tokens():
    engine = MockRolloutEngine()
    responses, probs = run_rollout(
        engine, [{"prompt_token_ids": [10]}, {"prompt_token_ids": [20]}], {0: [7, 99]},
        calculate_log_probs=False,
    )
    assert responses == [[7, 99], [2, 99]]
    assert probs == []
    assert len(engine.calls) == 1
    assert engine.calls[0][1][0].max_tokens == 1
    assert engine.calls[0][1][0].prompt_logprobs is None


def test_no_gt_uses_original_inputs_and_shared_sampling_params():
    engine = MockRolloutEngine()
    prompts = [{"prompt_token_ids": [10]}, {"prompt_token_ids": [20]}]
    responses, probs = run_rollout(engine, prompts, {})
    assert responses == [[1, 99], [2, 99]]
    assert probs == [[-0.1, -0.2]] * 2
    assert len(engine.calls) == 1
    assert not isinstance(engine.calls[0][1], list)
    assert all(request is prompt for request, prompt in zip(engine.calls[0][0], prompts, strict=True))


@pytest.mark.parametrize("bad_output", ["missing", "short", "missing_token", "wrong_suffix", "nan", "inf"])
def test_gt_scoring_rejects_incomplete_or_nonfinite_probabilities(bad_output):
    engine = MockRolloutEngine()
    generate = engine.generate

    def bad_generate(**kwargs):
        outputs = generate(**kwargs)
        for output in outputs:
            if bad_output == "missing":
                output.prompt_logprobs = None
            elif bad_output == "short":
                output.prompt_logprobs.pop()
            elif bad_output == "missing_token":
                output.prompt_logprobs[-1] = {}
            elif bad_output == "wrong_suffix":
                output.prompt_token_ids[-1] = 98
            else:
                output.prompt_logprobs[-1][99].logprob = float(bad_output)
        return outputs

    engine.generate = bad_generate
    with pytest.raises((ValueError, FloatingPointError), match="vLLM"):
        run_rollout(engine, [{"prompt_token_ids": [10]}], {0: [7, 99]})


def test_missing_request_output_fails_explicitly():
    engine = SimpleNamespace(generate=lambda **kwargs: [])
    with pytest.raises(ValueError, match="one request output"):
        run_rollout(engine, [{"prompt_token_ids": [10]}], {0: [7, 99]})


def test_multiple_samples_per_input_with_gt_fails_explicitly():
    engine = MockRolloutEngine()
    generate = engine.generate

    def multiple_samples(**kwargs):
        outputs = generate(**kwargs)
        for output in outputs:
            output.outputs *= 2
        return outputs

    engine.generate = multiple_samples
    with pytest.raises(ValueError, match="one vLLM output per input"):
        run_rollout(engine, [{"prompt_token_ids": [10]}], {0: [7, 99]})

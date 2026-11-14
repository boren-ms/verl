"""Check ASR feature reuse without starting Ray or importing the GPU runtime."""

import asyncio
import re
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def agent_loop(monkeypatch):
    import ray

    import recipe.phimm

    class Output(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.metrics = SimpleNamespace(**self.metrics)

    @contextmanager
    def timer(name, metrics):
        yield
        metrics[name] = 0.0

    # Isolate only the runtime boundaries; execute the production ASR methods.
    dependencies = {
        "verl.experimental.agent_loop.agent_loop": {
            "AgentLoopBase": object,
            "AgentLoopOutput": Output,
            "DictConfigWrap": lambda config: config,
            "register": lambda name: lambda cls: cls,
        },
        "verl.trainer.ppo.v1.agent_loop_tq": {
            "AgentLoopManagerTQ": object,
            "AgentLoopWorkerTQBase": object,
        },
        "verl.utils.profiler": {"simple_timer": timer},
    }
    with monkeypatch.context() as context:
        context.setattr(ray, "remote", lambda cls: cls)
        for name, attributes in dependencies.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            context.setitem(sys.modules, name, module)
        return runpy.run_path(str(Path(recipe.phimm.__file__).parent / "agentic/agent_loop.py"))


@pytest.mark.parametrize("validate", [False, True])
def test_generation_reuses_features_for_actor(agent_loop, validate):
    audio = (object(), 16000)
    features = {
        "input_audio_embeds": torch.arange(80 * 16).reshape(1, 16, 80).float(),
        "audio_attention_mask": torch.ones(1, 16, dtype=torch.long),
        "audio_embed_sizes": torch.tensor([2]),
        "audio_frames": torch.tensor([16]),
    }
    expanded_ids = [10, 248076, 248076, 11]
    processor = Mock(
        return_value={
            "input_ids": torch.tensor([expanded_ids]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            **features,
        }
    )
    processor.chat_template = "template"
    processor.apply_chat_template.return_value = "task<audio>"
    tokenizer = Mock()
    tokenizer.encode.return_value = [10, 99, 11]
    server = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(token_ids=[20, 21], log_probs=[-0.1, -0.2], extra_fields={}))
    )
    loop = agent_loop["ASRAgentLoop"]()
    loop.config = SimpleNamespace(
        data=SimpleNamespace(max_audio_dur=40, prompt_key="prompt", get=lambda key, default: default)
    )
    loop.rollout_config = SimpleNamespace(prompt_length=1024)
    loop.processor, loop.tokenizer, loop.server_manager = processor, tokenizer, server
    # runpy functions retain the module execution globals.
    namespace = loop.generate_row.__globals__
    namespace["load_audio"] = Mock(return_value=audio)
    namespace["sampling_params_for_row"] = Mock(return_value={"max_tokens": 512 if validate else 768})

    output = asyncio.run(loop.generate_row({"prompt": ["task"], "extra_info": {}}, {}, validate=validate))
    assert output.prompt_ids == expanded_ids
    assert output.response_ids == [20, 21]
    assert output.response_logprobs == [-0.1, -0.2]
    assert output.response_mask == [1, 1]
    request = server.generate.call_args.kwargs
    assert request["prompt_ids"] == [10, 99, 11]
    assert request["audio_data"][0] is audio

    worker = agent_loop["ASRWorker"]()
    # Repeated actor-input preparation must not tokenize or extract features.
    for _ in range(2):
        inputs = worker._compute_multi_modal_inputs(output, torch.tensor(expanded_ids + output.response_ids))
        assert inputs.keys() == features.keys()
        for key, value in features.items():
            assert inputs[key] is value
    processor.assert_called_once_with(text=["task<audio>"], audios=[audio], return_tensors="pt")
    tokenizer.decode.assert_not_called()
    assert "audios" not in output.multi_modal_data


def test_actor_inputs_without_multimodal_data(agent_loop):
    worker = agent_loop["ASRWorker"]()
    assert worker._compute_multi_modal_inputs(SimpleNamespace(multi_modal_data=None), torch.tensor([1])) is None


def test_grouped_validation_retains_first_segments_features(agent_loop):
    features = {"input_audio_embeds": torch.ones(1, 16, 80)}
    outputs = [
        SimpleNamespace(
            response_ids=[20 + index],
            response_logprobs=[-0.1],
            multi_modal_data={"multi_modal_inputs": features if index == 0 else {}},
            extra_fields={},
            metrics=SimpleNamespace(generate_sequences=1.0),
        )
        for index in range(2)
    ]
    loop = agent_loop["ASRAgentLoop"]()
    loop.config = SimpleNamespace(data=SimpleNamespace(version=2609))
    loop.generate_row = AsyncMock(side_effect=outputs)
    loop.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]), encode=lambda text, **kwargs: [30, 31])
    namespace = loop.run.__globals__
    score = Mock(return_value={"score": 1.0})
    namespace["score_response"] = score
    rows = [{"data_source": "test", "extra_info": {"segment": index}} for index in range(2)]

    output = asyncio.run(loop.run({}, asr_validate=True, asr_segments=rows))
    assert loop.generate_row.await_count == 2
    assert score.call_args.args[2] == ["20", "21"]
    assert output.response_logprobs is None
    assert output.response_ids == [30, 31]
    inputs = agent_loop["ASRWorker"]()._compute_multi_modal_inputs(output, torch.tensor([30, 31]))
    assert inputs is features


@pytest.mark.parametrize("validate", [False, True])
def test_both_modes_score_raw_headers_and_parent_reference(agent_loop, validate):
    from recipe.phimm.agentic.scoring import score_response

    texts = ["<src=English><tgt=English>\n<verbatim>hello</verbatim>", ""]
    outputs = [
        SimpleNamespace(
            response_ids=[index],
            extra_fields={},
            metrics=SimpleNamespace(generate_sequences=1.0),
        )
        for index in range(2)
    ]
    loop = agent_loop["ASRAgentLoop"]()
    loop.config = OmegaConf.create(
        {
            "data": {"version": 2609},
            "rewards": {
                "test": {
                    "path": "recipe.phimm.reward.asr_measure",
                    "name": "compute_score",
                    "reward_kwargs": {"measures": {"word": {}, "lang": {}, "fmt": {}}, "reduce": "mean"},
                }
            },
        }
    )
    loop.generate_segments = AsyncMock(return_value=(outputs, texts))
    encoded = []
    loop.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: encoded.append(text) or [30])
    loop.run.__globals__["score_response"] = score_response
    rows = [
        {"data_source": "test", "reward_model": {"ground_truth": "segment reference"}, "extra_info": {}} for _ in texts
    ]
    parent = {"ground_truth": "hello world"}
    output = asyncio.run(loop.run({}, asr_validate=validate, asr_segments=rows, reward_model=parent, extra_info={}))
    if not validate:
        assert len(output) == 2
        assert all(segment.reward_score == 0.5 for segment in output)
        output = output[0]
    result = output.extra_fields["reward_extra_info"]
    assert result["word"] == result["lang"] == result["fmt"] == 0.5
    assert output.reward_score == 0.5
    assert encoded == (["hello"] if validate else [])
    if validate:
        assert output.response_logprobs is None
        assert output.metrics.generate_sequences == 2


@pytest.fixture
def multi_turn_loop(agent_loop):
    class Tokenizer:
        specials = {"<audio>": 99, "<|im_start|>": 100, "<|im_end|>": 101}

        def encode(self, text, **kwargs):
            parts = re.split("(<audio>|<\\|im_start\\|>|<\\|im_end\\|>)", text)
            return [
                token
                for part in parts
                for token in ([self.specials[part]] if part in self.specials else [ord(char) + 256 for char in part])
            ]

        def decode(self, tokens, **kwargs):
            # Deliberately non-round-tripping: sampled IDs must never be replaced
            # by encoding this display/reward text.
            return f"transcript-{tokens[0]}"

    class Processor:
        chat_template = "chatml"

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

        def __call__(self, *, text, audios, **kwargs):
            self.calls.append((text, audios))
            frames = audios[0][0]
            ids = Tokenizer().encode(text[0])
            expanded = [token for value in ids for token in ([248076] * (frames // 8) if value == 99 else [value])]
            return {
                "input_ids": torch.tensor([expanded]),
                "attention_mask": torch.ones(1, len(expanded), dtype=torch.long),
                "input_audio_embeds": torch.full((1, frames, 80), float(frames)),
                "audio_attention_mask": torch.ones(1, frames, dtype=torch.long),
                "audio_embed_sizes": torch.tensor([frames // 8]),
                "audio_frames": torch.tensor([frames]),
            }

    loop = agent_loop["ASRAgentLoop"]()
    loop.config = OmegaConf.create(
        {"data": {"prompt_key": "prompt", "max_audio_dur": 40, "max_model_length": 2048, "version": 2609}}
    )
    loop.rollout_config = OmegaConf.create(
        {
            "prompt_length": 128,
            "response_length": 1024,
            "custom": {
                "no_repeat_ngram_size": 0,
                "validation_no_repeat_ngram_size": 0,
                "validation_response_length": 512,
            },
        }
    )
    loop.processor, loop.tokenizer = Processor(), Tokenizer()
    loop.generate_segments.__globals__["load_audio"] = lambda row, duration: (row["frames"], 16000)
    rows = [
        {
            "prompt": [{"role": "user", "content": "task<audio>"}],
            "extra_info": {"prefix": "prefix:"},
            "reward_model": {"ground_truth": f"reference-{i}"},
            "data_source": "test",
            "frames": frames,
        }
        for i, frames in enumerate((16, 8))
    ]
    generated = [[900, 901, 101], [902, 101]]

    async def generate(**kwargs):
        index = 0 if kwargs["audio_data"][0][0] == 16 else 1
        # Match the real server, which is allowed to consume sampling parameters.
        limit = kwargs["sampling_params"].pop("max_tokens")
        assert len(generated[index]) <= limit
        return SimpleNamespace(
            token_ids=generated[index],
            log_probs=[-float(token) / 1000 for token in generated[index]],
            extra_fields={"min_global_steps": index + 3, "max_global_steps": index + 4},
        )

    loop.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=generate))
    return loop, rows, generated


def test_segments_have_independent_contexts_and_aligned_actor_inputs(multi_turn_loop):
    loop, rows, generated = multi_turn_loop
    outputs, texts = asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=False))
    requests = sorted(
        (call.kwargs for call in loop.server_manager.generate.call_args_list),
        key=lambda request: request["audio_data"][0][0],
        reverse=True,
    )
    assert len(requests) == len(outputs) == 2
    for index, (request, output) in enumerate(zip(requests, outputs, strict=True)):
        assert request["audio_data"] == [(rows[index]["frames"], 16000)]
        expected_text = (
            loop.processor.apply_chat_template(rows[index]["prompt"], add_generation_prompt=True) + "prefix:"
        )
        assert request["prompt_ids"] == loop.tokenizer.encode(expected_text)
        assert 900 not in request["prompt_ids"] and 902 not in request["prompt_ids"]
        assert output.response_ids == generated[index]
        assert output.response_mask == [1] * len(generated[index])
        assert output.response_logprobs == [-float(token) / 1000 for token in generated[index]]
        features = output.multi_modal_data["multi_modal_inputs"]
        assert features["audio_frames"].tolist() == [rows[index]["frames"]]
        assert features["input_audio_embeds"].shape == (1, rows[index]["frames"], 80)
        assert output.prompt_ids.count(248076) == rows[index]["frames"] // 8
    assert texts == ["transcript-900", "transcript-902"]
    assert len(loop.processor.calls) == 2


def test_training_scores_combined_responses_once_and_shares_reward(multi_turn_loop):
    loop, rows, generated = multi_turn_loop
    score = Mock(return_value={"score": 0.5})
    loop.run.__globals__["score_response"] = score
    reference = {"ground_truth": "reference-0\nreference-1"}
    baseline = {"greedy_reward": {"score": 0}}
    outputs = asyncio.run(
        loop.run({"logprobs": True}, asr_validate=False, asr_segments=rows, reward_model=reference, extra_info=baseline)
    )
    score.assert_called_once()
    assert score.call_args.args[1]["reward_model"] == reference
    assert score.call_args.args[2] == ["transcript-900", "transcript-902"]
    assert score.call_args.kwargs["extra_info"] == baseline
    assert len(outputs) == 2
    for index, output in enumerate(outputs):
        assert output.reward_score == 0.5
        assert output.response_ids == generated[index]
        assert output.extra_fields["asr_segment_index"] == index
        assert output.extra_fields["asr_num_segments"] == 2
        assert output.extra_fields["reward_extra_info"]["score"] == 0.5


def test_greedy_baseline_uses_all_independent_segments(multi_turn_loop):
    loop, rows, _ = multi_turn_loop
    sample = {**rows[0], "asr_segments": rows, "reward_model": {"ground_truth": "combined reference"}}
    score = Mock(return_value={"score": 0, "word": 0.8})
    loop.run.__globals__["score_response"] = score
    text, result = asyncio.run(loop.greedy_baseline(sample))
    assert loop.server_manager.generate.await_count == 2
    for call in loop.server_manager.generate.call_args_list:
        assert call.kwargs["sampling_params"]["temperature"] == 0
        assert call.kwargs["sampling_params"]["logprobs"] is False
        assert len(call.kwargs["audio_data"]) == 1
    score.assert_called_once()
    assert score.call_args.args[1]["reward_model"]["ground_truth"] == "combined reference"
    assert score.call_args.args[2] == ["transcript-900", "transcript-902"]
    assert text == "transcript-900\ntranscript-902"
    assert result["word"] == 0.8


def test_segments_reject_missing_sampled_logprobs(multi_turn_loop):
    loop, rows, _ = multi_turn_loop
    loop.server_manager.generate = AsyncMock(return_value=SimpleNamespace(token_ids=[900], log_probs=None))
    with pytest.raises(ValueError, match="one rollout logprob"):
        asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=False))


def test_segments_reject_empty_samples(multi_turn_loop):
    loop, _, _ = multi_turn_loop
    with pytest.raises(ValueError, match="at least one segment"):
        asyncio.run(loop.generate_segments([], {}, validate=False))
    loop.server_manager.generate.assert_not_called()

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
    import recipe.phimm.utils.audio  # Import TransferQueue before replacing ray.remote below.

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


@pytest.mark.parametrize("reference", ["", "are you running a business"])
def test_validation_keeps_empty_hypotheses_and_their_scores(agent_loop, reference):
    loop = agent_loop["ASRAgentLoop"]()
    loop.config = SimpleNamespace(data=SimpleNamespace(version=2609))
    output = SimpleNamespace(
        response_ids=[9],
        extra_fields={},
        metrics=SimpleNamespace(generate_sequences=1.0),
    )
    texts = ["<src=English><tgt=English>\n<verbatim></verbatim>"]
    loop.generate_segments = AsyncMock(return_value=([output], texts))
    loop.tokenizer = SimpleNamespace(eos_token_id=9, encode=Mock(return_value=[]))
    score = {"score": -1.0, "n_err": 5 if reference else 0, "n_ref": 5 if reference else 1}
    scorer = Mock(return_value=score)
    loop.run.__globals__["score_response"] = scorer
    rows = [{"data_source": "test", "reward_model": {"ground_truth": reference}, "extra_info": {}}]

    result = asyncio.run(loop.run({}, asr_validate=True, asr_segments=rows))

    loop.tokenizer.encode.assert_called_once_with("", add_special_tokens=False)
    assert scorer.call_args.args[2] is texts
    assert result.response_ids == [9]
    assert result.response_mask == [1]
    assert result.reward_score == score["score"]
    assert result.extra_fields["reward_extra_info"]["n_err"] == score["n_err"]
    assert result.extra_fields["reward_extra_info"]["n_ref"] == score["n_ref"]


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
            "extra_info": {},
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
        expected_text = loop.processor.apply_chat_template(rows[index]["prompt"], add_generation_prompt=True)
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


@pytest.mark.parametrize("validate", [False, True])
def test_context_enters_user_prompt_and_scores_only_new_tokens(multi_turn_loop, validate):
    loop, rows, generated = multi_turn_loop
    loop.config.asr = {"context_aware": True}
    loop.rollout_config.prompt_length = 512
    outputs, texts = asyncio.run(loop.generate_segments(rows, {"logprobs": not validate}, validate=validate))
    requests = [call.kwargs for call in loop.server_manager.generate.call_args_list]
    for i, request in enumerate(requests):
        prefix = "" if i == 0 else "transcript-900"
        messages = [{**message} for message in rows[i]["prompt"]]
        if prefix:
            messages[-1]["content"] += f"\nPrevious context: {prefix}"
        text = loop.processor.apply_chat_template(messages, add_generation_prompt=True)
        assert text.endswith("<|im_start|>assistant\n")
        assert rows[i]["prompt"][-1]["content"] == "task<audio>"
        assert request["prompt_ids"] == loop.tokenizer.encode(text)
        assert loop.processor.calls[i][0] == [text]
        expected_actor_ids = [
            token
            for value in loop.tokenizer.encode(text)
            for token in ([248076] * (rows[i]["frames"] // 8) if value == 99 else [value])
        ]
        assert outputs[i].prompt_ids == expected_actor_ids
        assert outputs[i].multi_modal_data["multi_modal_inputs"]["audio_frames"].tolist() == [rows[i]["frames"]]
        assert outputs[i].response_ids == generated[i]
        assert outputs[i].response_mask == [1] * len(generated[i])
        assert outputs[i].response_logprobs == [-float(token) / 1000 for token in generated[i]]
        assert outputs[i].extra_fields["asr_previous_context"] == (f"Previous context: {prefix}" if prefix else "")
    assert texts == ["transcript-900", "transcript-902"]


def test_context_resets_between_parents_and_uses_only_previous_segment(multi_turn_loop):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": True}
    loop.rollout_config.prompt_length = 512

    async def evaluate():
        # Concurrent parents exercise the shared agent without sharing state.
        return await asyncio.gather(*(loop.generate_segments(rows + rows[:1], {}, validate=True) for _ in range(2)))

    results = asyncio.run(evaluate())
    for outputs, texts in results:
        assert [o.extra_fields["asr_previous_context"] for o in outputs] == [
            "",
            "Previous context: transcript-900",
            "Previous context: transcript-902",
        ]
        assert texts == ["transcript-900", "transcript-902", "transcript-900"]


@pytest.mark.parametrize(
    ("max_history", "filter_common", "expected"),
    [
        (1, True, ["", "*Qwen*, *Microsoft Research*, *OpenAI*", "*QWEN*"]),
        (1000, True, ["", "*Qwen*, *Microsoft Research*, *OpenAI*", "*QWEN*, *Microsoft Research*, *OpenAI*"]),
        (
            1,
            False,
            ["", "*Qwen*, *Microsoft Research*, *and*, *OpenAI*, *the*", "*know*, *you*, *QWEN*, *then*, *And*"],
        ),
    ],
)
def test_bias_context_reaches_rollout_and_actor(multi_turn_loop, max_history, filter_common, expected):
    loop, rows, generated = multi_turn_loop
    loop.config.asr = {
        "context_aware": {"enable": True, "mode": "bias", "max_history": max_history, "filter_common": filter_common}
    }
    loop.rollout_config.prompt_length = 512
    raw = {
        900: "<src=English><tgt=English>\n<VERBATIM>the OpenAI and Microsoft Research, uh, Q-wen, Qwen.</VERBATIM>",
        902: "<src=English><tgt=English>\n<VERBATIM>And then QWEN, you know.</VERBATIM>",
    }
    loop.tokenizer.decode = lambda ids, **kwargs: raw[ids[0]]
    for row in rows:
        row["prompt"][0]["content"] = "task<audio>\nThe language is English."

    outputs, texts = asyncio.run(loop.generate_segments(rows + rows[:1], {"logprobs": True}, validate=False))
    assert [out.extra_fields["asr_previous_context"] for out in outputs] == [
        f"Pay extra attention to the following phrases/words: {terms}." if terms else "" for terms in expected
    ]
    assert texts == [raw[900], raw[902], raw[900]]
    assert [out.response_ids for out in outputs] == generated + generated[:1]
    for out, row, call, terms in zip(
        outputs, rows + rows[:1], loop.server_manager.generate.call_args_list, expected, strict=True
    ):
        base = row["prompt"][0]["content"]
        assert base == "task<audio>\nThe language is English."
        content = base + (f"\nPay extra attention to the following phrases/words: {terms}." if terms else "")
        text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        request_ids = loop.tokenizer.encode(text)
        assert call.kwargs["prompt_ids"] == request_ids
        assert out.prompt_ids == [
            token for value in request_ids for token in ([248076] * (row["frames"] // 8) if value == 99 else [value])
        ]

    fresh, _ = asyncio.run(loop.generate_segments(rows[:1], {}, validate=True))
    assert fresh[0].extra_fields["asr_previous_context"] == ""


@pytest.mark.parametrize("max_history", [1, 1000])
@pytest.mark.parametrize(
    ("transcript", "filter_common"),
    [
        ("Well, uh, we have the.", True),
        ("...?!", False),
        ("2026, 3.14, 1,000, ½.", False),
        ("Erm, uh-huh, hmm.", False),
    ],
)
def test_bias_without_eligible_terms_keeps_base_prompt(multi_turn_loop, max_history, transcript, filter_common):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {
        "context_aware": {"enable": True, "mode": "bias", "max_history": max_history, "filter_common": filter_common}
    }
    loop.rollout_config.prompt_length = 512
    loop.tokenizer.decode = lambda ids, **kwargs: f"<src=English><tgt=English>\n<VERBATIM>{transcript}</VERBATIM>"
    row = rows[1]
    base_text = loop.processor.apply_chat_template(row["prompt"], add_generation_prompt=True)
    base_rollout_ids = loop.tokenizer.encode(base_text)
    base_actor_ids = loop.processor(text=[base_text], audios=[(row["frames"], 16000)])["input_ids"][0].tolist()
    loop.processor.calls.clear()

    outputs, _ = asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=False))

    assert outputs[1].extra_fields["asr_previous_context"] == ""
    assert loop.processor.calls[1][0] == [base_text]
    assert loop.server_manager.generate.call_args_list[1].kwargs["prompt_ids"] == base_rollout_ids
    assert outputs[1].prompt_ids == base_actor_ids


@pytest.mark.parametrize(
    ("filter_common", "filter_numbers", "expected"),
    [
        (True, True, "*GPT-4*, *3M*"),
        (True, False, "*GPT-4*, *3M*, *2026*, *314*, *1000*"),
        (False, True, "*GPT-4*, *3M*, *actually*, *usually*, *however*"),
        (False, False, "*GPT-4*, *3M*, *2026*, *314*, *1000*, *actually*, *usually*, *however*"),
    ],
)
def test_bias_filters_common_words_and_numbers_independently(multi_turn_loop, filter_common, filter_numbers, expected):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {
        "context_aware": {
            "enable": True,
            "mode": "bias",
            "filter_common": filter_common,
            "filter_numbers": filter_numbers,
        }
    }
    loop.rollout_config.prompt_length = 512
    transcript = "however, usually, actually, 1,000, 3.14, 2026, 3M, GPT4, GPT-4."
    loop.tokenizer.decode = lambda ids, **kwargs: f"<src=English><tgt=English>\n<VERBATIM>{transcript}</VERBATIM>"

    outputs, _ = asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=False))

    assert (
        outputs[1].extra_fields["asr_previous_context"]
        == f"Pay extra attention to the following phrases/words: {expected}."
    )


@pytest.mark.parametrize("max_history", [1, 1000])
def test_bias_preserves_vanity_phone_spelling(multi_turn_loop, max_history):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {
        "context_aware": {"enable": True, "mode": "bias", "max_history": max_history, "filter_numbers": True}
    }
    loop.rollout_config.prompt_length = 512
    loop.tokenizer.decode = lambda ids, **kwargs: "1800flowers, 1-800-FLOWERS, 425-555-0123."

    outputs, _ = asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=False))

    assert (
        outputs[1].extra_fields["asr_previous_context"]
        == "Pay extra attention to the following phrases/words: *1-800-FLOWERS*."
    )
    assert "Pay extra attention to the following phrases/words: *1-800-FLOWERS*." in loop.processor.calls[1][0][0]


@pytest.mark.parametrize(
    ("mode", "expected"), [("bias", "*Qwen*, *uh-huh*, *erm*, *UH*"), ("text", "UH, erm, uh-huh, Qwen.")]
)
def test_filler_filter_can_be_disabled_and_does_not_change_previous_text(multi_turn_loop, mode, expected):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": True, "mode": mode, "filter_fillers": mode == "text"}}
    loop.rollout_config.prompt_length = 512
    transcript = "UH, erm, uh-huh, Qwen."
    loop.tokenizer.decode = lambda ids, **kwargs: f"<src=English><tgt=English>\n<VERBATIM>{transcript}</VERBATIM>"
    outputs, _ = asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=False))
    assert outputs[1].extra_fields["asr_previous_context"] == (
        f"Previous context: {expected}"
        if mode == "text"
        else f"Pay extra attention to the following phrases/words: {expected}."
    )


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("mode", ["text", "bias"])
def test_response_window_reaches_prompts_and_resets_per_trajectory(multi_turn_loop, validate, mode):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": True, "mode": mode, "max_history": 2}}
    loop.rollout_config.prompt_length = 512
    transcripts = iter(["Alpha.", "Beta.", "Gamma.", "Delta."])
    loop.tokenizer.decode = lambda ids, **kwargs: next(transcripts)
    outputs, _ = asyncio.run(loop.generate_segments(rows * 2, {"logprobs": True}, validate=validate))
    expected = (
        ["", "Alpha.", "Alpha.\nBeta.", "Beta.\nGamma."]
        if mode == "text"
        else ["", "*Alpha*", "*Beta*, *Alpha*", "*Gamma*, *Beta*"]
    )
    assert [output.extra_fields["asr_previous_context"] for output in outputs] == [
        (
            f"Previous context: {value}"
            if mode == "text"
            else f"Pay extra attention to the following phrases/words: {value}."
        )
        if value
        else ""
        for value in expected
    ]
    for index, context in enumerate(expected[1:], start=1):
        assert context in loop.processor.calls[index][0][0]
    loop.tokenizer.decode = lambda ids, **kwargs: "Fresh."
    fresh, _ = asyncio.run(loop.generate_segments(rows[:1], {}, validate=validate))
    assert fresh[0].extra_fields["asr_previous_context"] == ""


@pytest.mark.parametrize(("max_history", "after_third"), [(1, "*NVIDIA*"), (1000, "*NVIDIA*, *Q-wen*")])
def test_bias_keeps_most_recent_distinct_terms(max_history, after_third):
    from recipe.phimm.agentic.context_prompt import ContextPrompt

    context = ContextPrompt("bias", max_history=max_history, max_terms=2, extra_common_words=["routine"])
    context.observe("THE and we, uh, don't. Routine.")
    assert context.context == ""
    context.observe("OpenAI, Qwen.")
    assert context.context == "*Qwen*, *OpenAI*"
    context.observe("OpenAI, Microsoft Research and QWEN, Q-wen.")
    assert context.context == "*Q-wen*, *Microsoft Research*"
    context.observe("NVIDIA.")
    assert context.context == after_third


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize(
    ("raw_response", "transcript"),
    [
        ("<src=English><tgt=English>\n<VERBATIM>\nHello there.</VERBATIM>", "Hello there."),
        (
            "<src=English><tgt=English>\n<TXT><READABLE>Hello.</READABLE></TXT>\n"
            "<src=French><tgt=French>\n<LEXICAL>Bonjour.</LEXICAL>",
            "Hello. Bonjour.",
        ),
        ("<src=English><tgt=English>\n<VERBATIM><foreign_language>bonjour</foreign_language>", "bonjour"),
        ("<nonspeech>", ""),
        ("<src=English><tgt=English>\n<VERBATIM></VERBATIM>", ""),
    ],
)
def test_context_contains_only_transcript_and_preserves_raw_scoring(
    multi_turn_loop, validate, raw_response, transcript
):
    loop, rows, generated = multi_turn_loop
    loop.config.asr = {"context_aware": True}
    loop.rollout_config.prompt_length = 512
    loop.tokenizer.decode = lambda ids, **kwargs: raw_response if ids[0] == 900 else "current transcript"
    outputs, texts = asyncio.run(loop.generate_segments(rows, {"logprobs": True}, validate=validate))

    assert texts == [raw_response, "current transcript"]
    assert outputs[0].response_ids == generated[0]
    assert outputs[1].extra_fields["asr_previous_context"] == (f"Previous context: {transcript}" if transcript else "")
    expected_user = "task<audio>" + (f"\nPrevious context: {transcript}" if transcript else "")
    expected = loop.processor.apply_chat_template(
        [{"role": "user", "content": expected_user}], add_generation_prompt=True
    )
    assert loop.processor.calls[1][0] == [expected]
    assert rows[1]["prompt"] == [{"role": "user", "content": "task<audio>"}]


@pytest.mark.parametrize("validate", [False, True])
def test_prefix_reaches_model(multi_turn_loop, validate):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {
        "context_aware": {
            "enable": True,
            "mode": "text",
            "prefix": "Previous segment transcript (for context):",
        }
    }
    loop.rollout_config.prompt_length = 512
    loop.tokenizer.decode = lambda ids, **kwargs: "Alpha." if ids[0] == 900 else "Beta."
    outputs, _ = asyncio.run(loop.generate_segments(rows, {}, validate=validate))
    assert "Previous segment transcript" not in loop.processor.calls[0][0][0]
    assert "Previous segment transcript (for context): Alpha." in loop.processor.calls[1][0][0]
    assert outputs[1].extra_fields["asr_previous_context"] == "Previous segment transcript (for context): Alpha."


def test_context_counts_towards_prompt_limit(multi_turn_loop):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": True}
    text = loop.processor.apply_chat_template(rows[0]["prompt"], add_generation_prompt=True)
    loop.rollout_config.prompt_length = len(loop.tokenizer.encode(text)) + rows[0]["frames"] // 8 - 1
    with pytest.raises(ValueError, match="Audio prompt has"):
        asyncio.run(loop.generate_segments(rows, {}, validate=True))
    assert loop.server_manager.generate.await_count == 1


@pytest.mark.parametrize("context_enabled", [False, True])
def test_training_scores_combined_responses_once_and_shares_reward(multi_turn_loop, context_enabled):
    loop, rows, generated = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": context_enabled}}
    loop.rollout_config.prompt_length = 512
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


@pytest.mark.parametrize("context_enabled", [False, True])
def test_greedy_baseline_uses_its_own_segment_context(multi_turn_loop, context_enabled):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": context_enabled}}
    loop.rollout_config.prompt_length = 512
    sample = {**rows[0], "asr_segments": rows, "reward_model": {"ground_truth": "combined reference"}}
    score = Mock(return_value={"score": 0, "word": 0.8})
    loop.run.__globals__["score_response"] = score
    text, result = asyncio.run(loop.greedy_baseline(sample))
    assert loop.server_manager.generate.await_count == 2
    for call in loop.server_manager.generate.call_args_list:
        assert call.kwargs["sampling_params"]["temperature"] == 0
        assert call.kwargs["sampling_params"]["logprobs"] is False
        assert len(call.kwargs["audio_data"]) == 1
    assert loop.processor.calls[0][0][0].endswith("<|im_start|>assistant\n")
    second_text = loop.processor.calls[1][0][0]
    assert second_text.endswith("<|im_start|>assistant\n")
    assert ("Previous context: transcript-900<|im_end|>" in second_text) is context_enabled
    score.assert_called_once()
    assert score.call_args.args[1]["reward_model"]["ground_truth"] == "combined reference"
    assert score.call_args.args[2] == ["transcript-900", "transcript-902"]
    assert text == "transcript-900\ntranscript-902"
    assert result["word"] == 0.8


@pytest.mark.parametrize("trajectory", ["sampled", "greedy", "validation"])
@pytest.mark.parametrize("context_enabled", [False, True])
def test_training_and_validation_share_history_flag(multi_turn_loop, trajectory, context_enabled):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": context_enabled}}
    loop.rollout_config.prompt_length = 512
    if trajectory == "greedy":
        loop.run.__globals__["score_response"] = Mock(return_value={"score": 0.5})
        asyncio.run(loop.greedy_baseline({**rows[0], "asr_segments": rows}))
    else:
        validate = trajectory == "validation"
        outputs, _ = asyncio.run(loop.generate_segments(rows, {"logprobs": not validate}, validate=validate))
        assert [o.extra_fields["asr_previous_context"] for o in outputs] == [
            "",
            "Previous context: transcript-900" if context_enabled else "",
        ]
    expected_prefix = "transcript-900" if context_enabled else ""
    second_text = next(text[0][0] for text in loop.processor.calls if text[1][0][0] == rows[1]["frames"])
    assert second_text.endswith("<|im_start|>assistant\n")
    assert (f"Previous context: {expected_prefix}<|im_end|>" in second_text) is context_enabled


def test_sampled_and_greedy_contexts_are_isolated(multi_turn_loop):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": True}
    loop.rollout_config.prompt_length = 512
    loop.run.__globals__["score_response"] = Mock(return_value={"score": 0.5})

    async def generate(**kwargs):
        sampled = kwargs["sampling_params"]["logprobs"]
        first = kwargs["audio_data"][0][0] == 16
        token = (910 if sampled else 920) + (0 if first else 1)
        return SimpleNamespace(token_ids=[token], log_probs=[-0.1] if sampled else None, extra_fields={})

    loop.server_manager.generate.side_effect = generate
    sample = {**rows[0], "asr_segments": rows}

    async def trajectories():
        return await asyncio.gather(
            loop.generate_segments(rows, {"logprobs": True}, validate=False), loop.greedy_baseline(sample)
        )

    (outputs, texts), (greedy, _) = asyncio.run(trajectories())
    assert texts == ["transcript-910", "transcript-911"]
    assert greedy == "transcript-920\ntranscript-921"
    assert [o.extra_fields["asr_previous_context"] for o in outputs] == ["", "Previous context: transcript-910"]
    for call in loop.server_manager.generate.call_args_list:
        request = call.kwargs
        if request["audio_data"][0][0] == 8:
            prefix = "transcript-910" if request["sampling_params"]["logprobs"] else "transcript-920"
            messages = [{**message} for message in rows[1]["prompt"]]
            messages[-1]["content"] += f"\nPrevious context: {prefix}"
            text = loop.processor.apply_chat_template(messages, add_generation_prompt=True)
            assert request["prompt_ids"] == loop.tokenizer.encode(text)
    assert all(o.response_mask == [1] and o.response_logprobs == [-0.1] for o in outputs)


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


@pytest.mark.parametrize("trajectory", ["sampled", "greedy"])
def test_reference_training_context_submits_all_segments_before_any_completes(multi_turn_loop, trajectory):
    loop, rows, generated = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": True, "training_source": "reference", "max_history": 1}}
    loop.rollout_config.prompt_length = 512
    rows[0]["asr_reference_context"] = []
    rows[1]["asr_reference_context"] = ["Annotated previous segment."]
    original_generate = loop.server_manager.generate.side_effect
    requests = []

    async def run():
        started = asyncio.Event()

        async def generate(**request):
            requests.append(request)
            if len(requests) == len(rows):
                started.set()
            # A sequential implementation times out here: no response can
            # finish until every segment has reached the inference server.
            await asyncio.wait_for(started.wait(), timeout=5)
            return await original_generate(**request)

        loop.server_manager.generate.side_effect = generate
        if trajectory == "greedy":
            loop.run.__globals__["score_response"] = Mock(return_value={"score": 0.5})
            text, result = await loop.greedy_baseline({**rows[0], "asr_segments": rows})
            assert text == "transcript-900\ntranscript-902" and result["score"] == 0.5
        else:
            outputs, texts = await loop.generate_segments(rows, {"logprobs": True}, validate=False)
            assert [o.response_ids for o in outputs] == generated
            assert texts == ["transcript-900", "transcript-902"]
            assert [o.extra_fields["asr_previous_context"] for o in outputs] == [
                "",
                "Previous context: Annotated previous segment.",
            ]

    asyncio.run(run())
    second_text = next(text[0][0] for text in loop.processor.calls if text[1][0][0] == 8)
    assert "Previous context: Annotated previous segment." in second_text
    assert "transcript-900" not in second_text and "reference-1" not in second_text
    assert rows[1]["prompt"][-1]["content"] == "task<audio>"


def test_reference_training_context_never_enters_validation(multi_turn_loop):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": True, "training_source": "reference"}}
    loop.rollout_config.prompt_length = 512
    for row in rows:
        row["asr_reference_context"] = ["SECRET VALIDATION LABEL"]
    outputs, _ = asyncio.run(loop.generate_segments(rows, {}, validate=True))
    assert [o.extra_fields["asr_previous_context"] for o in outputs] == ["", "Previous context: transcript-900"]
    assert all("SECRET" not in text[0][0] for text in loop.processor.calls)


@pytest.mark.parametrize("history", [None, "label", [None], [1]])
def test_reference_context_rejects_missing_or_malformed_data_before_generation(multi_turn_loop, history):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": True, "training_source": "reference"}}
    rows[0]["asr_reference_context"] = []
    rows[1]["asr_reference_context"] = history
    with pytest.raises(ValueError, match="offline_reference_context"):
        asyncio.run(loop.generate_segments(rows, {}, validate=False))
    loop.server_manager.generate.assert_not_called()


@pytest.mark.parametrize("history_size, expected", [(1, ""), (2, "Previous context: Annotated text.")])
def test_fixed_reference_window_counts_empty_segments(multi_turn_loop, history_size, expected):
    loop, rows, _ = multi_turn_loop
    loop.config.asr = {"context_aware": {"enable": True, "training_source": "reference", "max_history": history_size}}
    loop.rollout_config.prompt_length = 512
    rows[0]["asr_reference_context"] = ["Annotated text.", ""]
    rows[1]["asr_reference_context"] = []
    outputs, _ = asyncio.run(loop.generate_segments(rows, {}, validate=False))
    assert [o.extra_fields["asr_previous_context"] for o in outputs] == [expected, ""]


def test_u2_validation_decodes_with_history_and_scores_full_reference_once(agent_loop):
    import json

    from recipe.phimm.agentic.scoring import score_response

    loop = agent_loop["ASRAgentLoop"]()
    loop.config = OmegaConf.create(
        {
            "asr": {"context_aware": True},
            "data": {"version": 2609},
            "rewards": {
                "test": {
                    "path": "recipe.phimm.reward.asr_eval",
                    "name": "openasr_en_eval",
                    "reward_kwargs": {"version": 2609},
                }
            },
        }
    )
    outputs = [
        SimpleNamespace(response_ids=[i], extra_fields={}, metrics=SimpleNamespace(generate_sequences=1.0))
        for i in range(2)
    ]
    calls = []

    async def generate(row, params, *, validate, previous_context=""):
        calls.append(previous_context)
        output = outputs[len(calls) - 1]
        output.extra_fields["asr_previous_context"] = previous_context
        return output

    loop.generate_row = generate
    loop.tokenizer = SimpleNamespace(
        decode=lambda ids, **kwargs: ["hello", "world"][ids[0]], encode=lambda text, **kwargs: [30]
    )
    loop.run.__globals__["score_response"] = score_response
    rows = [
        {"data_source": "test", "reward_model": {"ground_truth": "hello world"}, "extra_info": {}} for _ in range(2)
    ]
    output = asyncio.run(loop.run({}, asr_validate=True, asr_segments=rows))
    assert calls == ["", "Previous context: hello"]
    info = output.extra_fields["reward_extra_info"]
    assert info["n_err"] == 0 and info["n_ref"] == 2
    detail = json.loads(info["asr_detail"])
    assert detail["previous_contexts"] == ["", "Previous context: hello"]
    assert detail["segment_outputs"] == ["hello", "world"]


@pytest.mark.parametrize("validate", [False, True])
def test_random_merge_audio_list_enters_context_decode_in_order(multi_turn_loop, validate):
    import datasets
    from recipe.phimm.agentic.dataset import ASRDataset
    from recipe.phimm.data.dataset import augment
    from recipe.phimm.tools.random_merge_list import random_merge

    loop, _, _ = multi_turn_loop
    raw = datasets.Dataset.from_list(
        [
            {"audio_path": path, "text": text, "prompt": "task<audio>", "audio_length": 1.0}
            for path, text in [("a.wav", "reference a"), ("b.wav", "reference b")]
        ]
    )
    prepared = augment(
        random_merge(raw, max_audio_length=3),
        post_process={"verl_format": {"prompt_key": "prompt"}},
    )
    dataset = ASRDataset.__new__(ASRDataset)
    dataset.ds, dataset.groups, dataset.prompt_key = prepared, None, "prompt"
    sample = dataset[0]
    loop.config.asr = {"context_aware": True}
    loop.rollout_config.prompt_length = 512
    frames = {"a.wav": 16, "b.wav": 8}
    loop.generate_segments.__globals__["load_audio"] = lambda row, duration: (frames[row["audio_path"]], 16000)
    outputs, texts = asyncio.run(loop.generate_segments(sample["asr_segments"], {}, validate=validate))
    assert len(outputs) == 2
    assert [out.extra_fields["asr_previous_context"] for out in outputs] == ["", f"Previous context: {texts[0]}"]
    requests = [call.kwargs for call in loop.server_manager.generate.call_args_list]
    assert [request["audio_data"][0][0] for request in requests] == [frames[path] for path in sample["audio_path"]]
    messages = [{**message} for message in sample["prompt"]]
    messages[-1]["content"] += f"\nPrevious context: {texts[0]}"
    assert requests[1]["prompt_ids"] == loop.tokenizer.encode(
        loop.processor.apply_chat_template(messages, add_generation_prompt=True)
    )
    assert sample["reward_model"]["ground_truth"] == " ".join(
        "reference a" if path == "a.wav" else "reference b" for path in sample["audio_path"]
    )

"""Check ASR feature reuse without starting Ray or importing the GPU runtime."""

import asyncio
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch


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
    namespace["get_hyp_text"] = lambda text, **kwargs: text
    score = Mock(return_value={"score": 1.0})
    namespace["score_response"] = score
    rows = [{"data_source": "test", "extra_info": {"segment": index}} for index in range(2)]

    output = asyncio.run(loop.run({}, asr_validate=True, asr_segments=rows))
    assert loop.generate_row.await_count == 2
    assert score.call_args.args[2] == "20\n21"
    assert output.response_logprobs is None
    assert output.response_ids == [30, 31]
    inputs = agent_loop["ASRWorker"]()._compute_multi_modal_inputs(output, torch.tensor([30, 31]))
    assert inputs is features

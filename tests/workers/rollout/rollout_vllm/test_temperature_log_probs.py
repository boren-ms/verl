"""CPU checks of the raw scoring boundary without importing vLLM/CUDA."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "temperature_log_probs",
    Path(__file__).parents[4] / "verl/utils/vllm/temperature_log_probs.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_sampler():
    class Sampler:
        logprobs_mode = "raw_logprobs"
        compute_logprobs = staticmethod(lambda logits: logits.log_softmax(-1, dtype=torch.float32))

        def forward(self, logits, sampling_metadata, **kwargs):
            scores = self.compute_logprobs(logits)
            # Emulate a processor masking the raw maximum, then sampling.
            logits[:, 0] = -torch.inf
            sampled = (logits / sampling_metadata.temperature.clamp_min(1).unsqueeze(-1)).argmax(-1)
            return scores, sampled

    return Sampler


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_temperature_scores_exclude_processors_and_preserve_sampling(dtype):
    sampler_cls = make_sampler()
    unpatched = sampler_cls()
    logits = torch.tensor([[4., 2., 1.], [3., 1., 2.]], dtype=dtype)
    metadata = SimpleNamespace(temperature=torch.tensor([1.2, 0.7]), all_greedy=False, max_num_logprobs=0)
    _, expected_sample = unpatched.forward(logits.clone(), metadata)
    module.patch_temperature_only_log_probs(sampler_cls)
    module.patch_temperature_only_log_probs(sampler_cls)
    scores, sampled = sampler_cls().forward(logits.clone(), metadata)
    expected = (logits.float() / metadata.temperature[:, None]).log_softmax(-1)
    torch.testing.assert_close(scores, expected)
    assert torch.isfinite(scores).all()  # masked token remains finite in raw scores
    torch.testing.assert_close(sampled, expected_sample)
    # Prompt scoring outside forward must remain raw, even after patching.
    torch.testing.assert_close(sampler_cls.compute_logprobs(logits), logits.float().log_softmax(-1))


@pytest.mark.parametrize("all_greedy,temperatures", [(True, [0., 0.]), (False, [0., 1.2])])
def test_greedy_and_mixed_requests(all_greedy, temperatures):
    sampler_cls = make_sampler()
    module.patch_temperature_only_log_probs(sampler_cls)
    logits = torch.tensor([[4., 2., 1.], [3., 1., 2.]])
    metadata = SimpleNamespace(temperature=torch.tensor(temperatures), all_greedy=all_greedy, max_num_logprobs=0)
    scores, _ = sampler_cls().forward(logits.clone(), metadata)
    temperature = torch.tensor([1., 1.]) if all_greedy else torch.tensor([1., 1.2])
    torch.testing.assert_close(scores, (logits / temperature[:, None]).log_softmax(-1))


def test_failure_restores_context_and_rejects_processed_mode():
    sampler_cls = make_sampler()
    module.patch_temperature_only_log_probs(sampler_cls)
    with pytest.raises(ValueError, match="raw_logprobs"):
        sampler_cls().forward(torch.ones(1, 3), None, logprobs_mode_override="processed_logprobs")
    with pytest.raises(AttributeError):
        sampler_cls().forward(torch.ones(1, 3), None)
    assert module._raw_score_context.get() is None


def test_default_worker_installation_preserves_engine_settings():
    kwargs = {"gpu_memory_utilization": 0.9}
    module.configure_temperature_only_log_probs(kwargs)
    module.configure_temperature_only_log_probs(kwargs)
    assert kwargs["gpu_memory_utilization"] == 0.9
    assert kwargs["worker_extension_cls"].endswith("TemperatureLogProbWorkerExtension")
    assert "logprobs_mode" not in kwargs


@pytest.mark.parametrize("kwargs", [
    {"logprobs_mode": "processed_logprobs"},
    {"worker_extension_cls": "custom.Extension"},
])
def test_worker_installation_rejects_conflicting_settings(kwargs):
    with pytest.raises(ValueError):
        module.configure_temperature_only_log_probs(kwargs)

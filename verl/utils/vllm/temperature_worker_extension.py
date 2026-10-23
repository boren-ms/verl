"""Loaded by vLLM in each engine worker, including spawned async workers."""

from vllm.v1.sample.sampler import Sampler

from verl.utils.vllm.temperature_log_probs import patch_temperature_only_log_probs

patch_temperature_only_log_probs(Sampler)


class TemperatureLogProbWorkerExtension:
    """Importing this extension installs temperature-only raw scoring."""

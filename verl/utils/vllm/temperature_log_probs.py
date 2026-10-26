"""Temperature-only scoring at vLLM V1's raw-logit scoring boundary."""

from contextvars import ContextVar
from functools import wraps

import torch


_raw_score_context = ContextVar("verl_raw_score_context", default=None)


def configure_temperature_only_log_probs(engine_kwargs, enabled=True):
    """Install the default scoring fix unless a control run explicitly disables it."""
    if not enabled:
        return
    if engine_kwargs.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        raise ValueError("vLLM rollout temperature-only scoring requires raw_logprobs")
    extension = "verl.utils.vllm.temperature_worker_extension.TemperatureLogProbWorkerExtension"
    existing = engine_kwargs.get("worker_extension_cls")
    if existing and existing != extension:
        raise ValueError("vLLM rollout temperature scoring cannot replace an existing worker extension")
    engine_kwargs["worker_extension_cls"] = extension


def patch_temperature_only_log_probs(sampler_cls):
    """Scale only the raw score computation; leave sampling logits untouched.

    V1 calls compute_logprobs on raw logits before applying any processors.
    Prompt log-prob scoring happens outside forward and remains untouched;
    teacher-forced GT rows are separately scored by the actor in the trainer.
    """
    if getattr(sampler_cls, "_verl_temperature_only_log_probs", False):
        return
    original_forward = sampler_cls.forward
    original_compute = sampler_cls.compute_logprobs

    @wraps(original_forward)
    def forward(self, logits, sampling_metadata, *args, **kwargs):
        override = kwargs.get("logprobs_mode_override", args[1] if len(args) > 1 else None)
        mode = override or self.logprobs_mode
        if mode != "raw_logprobs":
            raise ValueError("temperature-only rollout scoring requires raw_logprobs")
        context = [logits, sampling_metadata, False]
        token = _raw_score_context.set(context)
        try:
            output = original_forward(self, logits, sampling_metadata, *args, **kwargs)
            if sampling_metadata.max_num_logprobs is not None and not context[2]:
                raise RuntimeError("vLLM did not use the expected raw-logit scoring boundary")
            return output
        finally:
            _raw_score_context.reset(token)

    @wraps(original_compute)
    def compute_logprobs(logits):
        context = _raw_score_context.get()
        if context is not None and logits is context[0] and not context[2]:
            context[2] = True
            metadata = context[1]
            if not metadata.all_greedy and metadata.temperature is not None:
                temperature = metadata.temperature
                # vLLM uses zero temperature for greedy rows in a mixed batch.
                temperature = torch.where(temperature > 0, temperature, 1.0)
                logits = logits.float() / temperature.unsqueeze(-1)
        return original_compute(logits)

    sampler_cls.forward = forward
    sampler_cls.compute_logprobs = staticmethod(compute_logprobs)
    sampler_cls._verl_temperature_only_log_probs = True

"""Error-rate ASR reward reusing the original measurement functions.

Select ``path: recipe/phimm/reward/asr_measure_v0.py`` and
``name: compute_score`` in a reward-function configuration. Shared helpers
retain their original contracts; ``_parse_response`` returns error rates.
The original module and its configurations are unchanged.
"""

from __future__ import annotations

from recipe.phimm.reward.asr_measure import (
    _parse_response as _parse_accuracy,
    compute_deltas,
    lang_score as _lang_accuracy_score,
    reduce_scores,
    scale_score,
    signed_pow,
)


def _parse_response(solution_str, ground_truth=None, **kwargs) -> dict[str, float]:
    """Return all parsed reward components as unclipped error rates."""
    accuracies = _parse_accuracy(solution_str, ground_truth=ground_truth, **kwargs)
    return {name: 1.0 - accuracy for name, accuracy in accuracies.items()}


def compute_score(solution_str, ground_truth, **kwargs):
    """Return configured raw errors and ``0 - reduced_error_rate``.

    Each component contributes ``beta * signed_pow(clip(error, low, high),
    gamma)`` using the shared weighting helper. Defaults remain unchanged:
    beta/gamma are one, low is minus one, and high is one. Raw metrics are
    not clipped, so insertion errors may exceed one.

    ``reduce`` accepts a mode string or mapping; mapping options override
    reward kwargs. The reduced error is transformed by the global ``gamma``
    and multiplied by the raw ``char``/``word`` error if that ``scale`` is
    selected. Default ``scale: constant`` leaves it unchanged.

    This variant does not apply ``cut`` thresholds.

    Greedy baselines must use this variant. Shared ``d_<name>`` fields are
    raw value minus baseline; component deltas measure error increases and
    ``d_score`` measures reward gains.

    Configuration example (YAML)::

        reward_kwargs:
          reduce: {mode: average, total: 2}
          measures:
            word: {beta: 1.0, low: 0.0, high: 10.0}
            word_fmt: {beta: 0.5}
    """
    scale = kwargs.get("scale", "constant")
    if scale not in ("constant", "char", "word"):
        raise ValueError(f"Unsupported score scale: {scale!r}. Expected 'constant', 'char', or 'word'.")
    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    measures = kwargs.get("measures") or {}
    unknown = set(measures) - parsed.keys()
    if unknown:
        raise ValueError(f"Unsupported ASR error measures: {', '.join(sorted(unknown))}.")
    reduction = kwargs.get("reduce", "sum")
    if isinstance(reduction, str):
        reduction = {"mode": reduction.lower()}
    else:
        reduction = {**reduction, "mode": reduction.get("mode", "sum").lower()}
    gamma = float(kwargs.get("gamma", 1.0))
    errors = [scale_score(parsed[name], cfg) for name, cfg in measures.items()]
    reduced_error = reduce_scores(errors, **(kwargs | reduction))
    reduced_error = signed_pow(reduced_error, gamma) * (1.0 if scale == "constant" else parsed[scale])
    score = 0.0 - reduced_error

    result = {"score": score, **{name: parsed[name] for name in measures}}
    extra_info = kwargs.get("extra_info") or {}
    return {**result, **compute_deltas(result, extra_info.get("greedy_reward"))}


def lang_score(solution_str, ground_truth=None, **kwargs):
    """Return negative language error and the raw language error metric."""
    accuracy = _lang_accuracy_score(solution_str, ground_truth=ground_truth, **kwargs)["p_lang"]
    error = 1.0 - accuracy
    return {"score": 0.0 - error, "p_lang": error}

"""Delta-voting ASR reward using the shared accuracy measurements.

Select ``path: recipe/phimm/reward/asr_vote.py`` and ``name: compute_score``
in a reward-function configuration. Existing scorers are unchanged.
"""

from __future__ import annotations

import math

from recipe.phimm.reward.asr_measure import _parse_response, compute_delta, reduce_scores


def compute_vote(delta: float | None, low: float = -0.1, high: float = 0.1) -> float:
    """Return a measure's -1/0/+1 vote; a missing baseline delta votes zero.

    Threshold equality is neutral. Thresholds and non-None deltas must be
    finite, with ``low <= high``. Beta weighting is applied by the caller.
    """
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("ASR vote requires finite low <= high.")
    if delta is None:
        return 0.0
    if not math.isfinite(delta):
        raise ValueError("ASR vote requires a finite delta.")
    return 1.0 if delta > high else -1.0 if delta < low else 0.0


def _parse_reduction(reduction):
    if isinstance(reduction, str):
        reduction = {"mode": reduction.lower()}
    else:
        reduction = {**reduction, "mode": reduction.get("mode", "mean").lower()}
    if reduction["mode"] not in {"sum", "mean", "average", "multiply", "geometric", "harmonic"}:
        raise ValueError(f"Unsupported ASR vote reduction: {reduction['mode']!r}.")
    return reduction


def compute_measure_vote(name: str, value: float, baseline: float, cfg=None) -> tuple[float, float]:
    """Return a measure's beta-weighted vote and delta against a required baseline."""
    cfg = cfg or {}
    delta_mode = cfg.get("delta", "abs")
    if delta_mode not in ("abs", "absolute", "rel", "relative"):
        raise ValueError(
            f"Unsupported ASR vote delta mode for {name!r}: {delta_mode!r}. "
            "Expected 'abs'/'absolute' or 'rel'/'relative'."
        )
    low = float(cfg.get("low", -0.1))
    high = float(cfg.get("high", 0.1))
    beta = float(cfg.get("beta", 1.0))
    if not math.isfinite(beta):
        raise ValueError(f"ASR vote measure {name!r} requires finite beta.")

    delta = compute_delta(value, baseline, mode=delta_mode)
    vote = compute_vote(delta, low=low, high=high)
    return beta * vote, delta


def compute_score(solution_str, ground_truth, **kwargs):
    """Reduce beta-weighted votes on accuracy deltas against a greedy baseline.

    For each configured measure, ``delta: abs`` (the default) uses
    ``delta = accuracy - baseline_accuracy``. ``delta: rel`` uses
    ``delta = (accuracy - baseline_accuracy) / max(abs(baseline_accuracy), 1e-8)``.
    The denominator stays positive for negative accuracies and is floored at
    1e-8 for zero or near-zero baselines; identical zero values have delta zero.
    ``absolute`` and ``relative`` are also accepted for ``abs`` and ``rel``.
    Delta modes are configured per measure, not as a shared reward kwarg.
    Relative thresholds are fractions, so 0.1 means a 10% relative improvement.
    Vote +1 when ``delta > high``, -1 when ``delta < low``, and zero otherwise,
    including equality at either threshold. Per-measure ``low`` and ``high``
    default to -0.1 and 0.1, and ``beta`` defaults to one.

    ``score = reduce([beta * vote, ...])`` defaults to ``mean``. As in
    ``asr_measure``, ``reduce`` accepts a mode string or a mapping; mapping
    options override reward kwargs (e.g. ``total`` for ``average``).
    Accuracy clipping, gamma transforms, cut gates, and accuracy scaling are
    not applied: low/high define the voting interval, not clipping bounds.

    Raw configured accuracies and shared ``d_<name>`` fields are returned
    alongside ``score``. These deltas use each measure's ``delta`` mode and
    are the same values used for voting. With no baseline (None or empty),
    the score is zero and no delta fields are returned, allowing the same
    scorer to collect greedy baseline measurements. This returns early,
    skipping vote and reduction configuration and computation. A nonempty baseline must
    contain every configured measure; it may come from this scorer or
    ``asr_measure``. Only measure deltas are returned, never ``d_score``.

    Configuration example (YAML)::

        reward_kwargs:
          reduce: mean
          measures:
            word: {delta: rel, beta: 1.0, low: -0.1, high: 0.1}
            fmt: {delta: abs, beta: 0.5, low: -0.1, high: 0.1}
    """
    if "delta" in kwargs:
        raise ValueError("Configure ASR vote delta per measure, not as a shared reward kwarg.")

    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    measures = kwargs.get("measures") or {}
    unknown = set(measures) - parsed.keys()
    if unknown:
        raise ValueError(f"Unsupported ASR vote measures: {', '.join(sorted(unknown))}.")

    raw_scores = {name: parsed[name] for name in measures}
    extra_info = kwargs.get("extra_info") or {}
    baseline = extra_info.get("greedy_reward")
    if not baseline:
        return {"score": 0.0, **raw_scores}

    missing = set(measures) - baseline.keys()
    if missing:
        raise ValueError(f"Missing greedy baseline ASR vote measures: {', '.join(sorted(missing))}.")

    reduction = _parse_reduction(kwargs.get("reduce", "mean"))
    deltas = {}
    scores = []
    for name, cfg in measures.items():
        score, delta = compute_measure_vote(name, raw_scores[name], baseline[name], cfg)
        scores.append(score)
        deltas[f"d_{name}"] = delta

    score = reduce_scores(scores, **(kwargs | reduction))
    return {"score": score, **raw_scores, **deltas}

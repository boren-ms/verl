"""Delta-voting ASR reward using the shared accuracy measurements.

Select ``path: recipe/phimm/reward/asr_vote.py`` and ``name: compute_score``
in a reward-function configuration. Existing scorers are unchanged.
"""

from __future__ import annotations

import math
from numbers import Real

from omegaconf import ListConfig

from recipe.phimm.reward.asr_measure import _parse_response, clip, compute_delta, reduce_scores


def compute_vote(delta: float | None, low: float = -0.01, high: float = 0.01) -> float:
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


def _parse_bounds(name, option, bounds):
    if not isinstance(bounds, (list, tuple, ListConfig)) or len(bounds) != 2:
        raise ValueError(f"ASR vote measure {name!r} requires {option}: [low, high].")
    for value in bounds:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"ASR vote measure {name!r} requires finite numeric {option} bounds.")
    if bounds[0] > bounds[1]:
        raise ValueError(f"ASR vote measure {name!r} requires {option} low <= high.")
    return bounds


def compute_measure_vote(name: str, value: float, baseline: float, cfg=None) -> tuple[float, float, float]:
    """Return a beta-weighted clipped vote, raw delta, and clipped vote against a required baseline."""
    cfg = cfg or {}
    if "low" in cfg or "high" in cfg:
        raise ValueError(f"ASR vote measure {name!r} requires tol: [low, high] instead of low/high.")
    delta_mode = cfg.get("delta", "abs")
    if delta_mode not in ("abs", "absolute", "rel", "relative"):
        raise ValueError(
            f"Unsupported ASR vote delta mode for {name!r}: {delta_mode!r}. "
            "Expected 'abs'/'absolute' or 'rel'/'relative'."
        )
    vote_mode = cfg.get("vote", "thr")
    if vote_mode not in ("thr", "threshold", "delta"):
        raise ValueError(
            f"Unsupported ASR vote mode for {name!r}: {vote_mode!r}. "
            "Expected 'thr' (alias 'threshold') or 'delta'."
        )
    beta = float(cfg.get("beta", 1.0))
    if not math.isfinite(beta):
        raise ValueError(f"ASR vote measure {name!r} requires finite beta.")

    delta = compute_delta(value, baseline, mode=delta_mode)
    if vote_mode == "delta":
        if not math.isfinite(delta):
            raise ValueError("ASR vote requires a finite delta.")
        vote = delta
    else:
        low, high = _parse_bounds(name, "tol", cfg.get("tol", [-0.01, 0.01]))
        vote = compute_vote(delta, low=low, high=high)
    bounds = _parse_bounds(name, "clip", cfg.get("clip", [-10, 10]))
    vote = clip(vote, bounds[0], bounds[1])
    score = beta * vote
    if not math.isfinite(score):
        raise ValueError(f"ASR vote measure {name!r} produced a nonfinite weighted score.")
    return score, delta, vote


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
    including equality at either threshold. Per-measure ``tol: [low, high]``
    defaults to [-0.01, 0.01], and ``beta`` defaults to one.
    Separate per-measure ``low``/``high`` options are no longer accepted.

    Per-measure ``vote: thr`` is the default behavior above; ``threshold``
    is also accepted as an alias.
    ``vote: delta`` instead uses the computed finite delta directly as the
    vote, ignoring tol without thresholding or a neutral interval. It may
    exceed [-1, 1], especially for relative deltas near a zero baseline.
    Both modes can be mixed across measures; beta weighting and reduction
    are unchanged.

    Per-measure ``clip: [low, high]`` bounds the vote BEFORE beta
    multiplication and reduction, for either vote mode.
    Omitting clip uses [-10, 10]; both bounds must be finite
    numbers with low <= high. Raw deltas remain unchanged.
    ``tol`` defines voting thresholds, not clipping bounds.

    ``score = reduce([beta * clip(vote, *bounds), ...])``
    defaults to ``mean``. As in
    ``asr_measure``, ``reduce`` accepts a mode string or a mapping; mapping
    options override reward kwargs (e.g. ``total`` for ``average``).
    Accuracy clipping, gamma transforms, cut gates, and accuracy scaling are
    not applied: tol defines the voting interval, not clipping bounds.

    Raw configured accuracies, ``d_<name>`` deltas, and ``v_<name>`` votes
    are returned alongside ``score``. Returned votes are clipped but unweighted:
    -1/0/+1 in threshold mode or the delta itself in delta mode, followed by
    the configured clipping in either mode.
    These deltas use each measure's ``delta`` mode and
    are the same values used for voting. With no baseline (None or empty),
    the score is zero and no delta or vote fields are returned, allowing the same
    scorer to collect greedy baseline measurements. This returns early,
    skipping vote and reduction configuration and computation. A nonempty baseline must
    contain every configured measure; it may come from this scorer or
    ``asr_measure``. Only measure deltas are returned, never ``d_score``.

    Configuration example (YAML)::

        reward_kwargs:
          reduce: mean
          measures:
            word: {delta: rel, beta: 1.0, tol: [-0.01, 0.01]}
            fmt: {delta: abs, beta: 0.5, tol: [-0.01, 0.01]}

    Direct-delta example (YAML)::

        reward_kwargs:
          reduce: {mode: average, total: 1.5}
          measures:
            word: {delta: rel, vote: delta, beta: 1.0, clip: [-1000, 100]}
            lang: {delta: abs, vote: thr, beta: 0.5}
    """
    if "delta" in kwargs:
        raise ValueError("Configure ASR vote delta per measure, not as a shared reward kwarg.")
    if "vote" in kwargs:
        raise ValueError("Configure ASR vote mode per measure, not as a shared reward kwarg.")
    if "clip" in kwargs:
        raise ValueError("Configure ASR vote clipping per measure, not as shared reward kwargs.")
    if "tol" in kwargs:
        raise ValueError("Configure ASR vote tol per measure, not as a shared reward kwarg.")

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
    votes = {}
    scores = []
    for name, cfg in measures.items():
        score, delta, vote = compute_measure_vote(name, raw_scores[name], baseline[name], cfg)
        scores.append(score)
        deltas[f"d_{name}"] = delta
        votes[f"v_{name}"] = vote

    score = reduce_scores(scores, **(kwargs | reduction))
    return {"score": score, **raw_scores, **deltas, **votes}

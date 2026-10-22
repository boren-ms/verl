"""ASR reward mixing delta votes and absolute measure scores.

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
    """Return a weighted clipped vote, diagnostic delta, and clipped vote."""
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
    if vote_mode not in ("thr", "threshold", "delta", "abs", "absolute"):
        raise ValueError(
            f"Unsupported ASR vote mode for {name!r}: {vote_mode!r}. "
            "Expected 'thr' (alias 'threshold'), 'delta', or 'abs' (alias 'absolute')."
        )
    beta = float(cfg.get("beta", 1.0))
    if not math.isfinite(beta):
        raise ValueError(f"ASR vote measure {name!r} requires finite beta.")

    delta = compute_delta(value, baseline, mode=delta_mode)
    if vote_mode in ("abs", "absolute"):
        if not math.isfinite(value):
            raise ValueError("ASR absolute vote requires a finite measure value.")
        if not math.isfinite(delta):
            raise ValueError("ASR vote requires a finite delta.")
        vote = value
    elif vote_mode == "delta":
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
    """Reduce beta-weighted delta votes or absolute scores per measure.

    Per-measure ``score`` defaults to ``acc`` (accuracy). ``err``
    instead uses ``accuracy - 1`` (negative error) for that measure, before
    computing deltas and votes. Score types can be mixed across measures.
    Returned raw measures and greedy baseline measurements use each measure's
    selected representation. Absolute deltas are identical for both
    types; relative deltas use the magnitude of the selected baseline value.
    For each configured measure, ``delta: abs`` (the default) uses
    ``delta = value - baseline_value`` for both types, so improvements are positive.
    ``delta: rel`` divides this improvement by
    ``max(abs(baseline_value), 1e-8)``.
    The denominator stays positive for negative values and is floored at
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
    ``vote: abs`` (alias ``absolute``) uses the current measure value directly,
    without subtracting the greedy value or taking its mathematical absolute
    value. ``score: acc`` therefore votes with current accuracy; ``score: err``
    votes with current negative error. Language and format scores are parsed
    correctness scores, not model token probabilities. Delta remains available
    as a diagnostic and does not affect an absolute vote. All three modes can
    be mixed across measures; beta weighting and reduction are unchanged.

    Per-measure ``clip: [low, high]`` bounds the vote BEFORE beta
    multiplication and reduction, for every vote mode.
    Omitting clip uses [-10, 10]; both bounds must be finite
    numbers with low <= high. Raw deltas remain unchanged.
    ``tol`` defines voting thresholds, not clipping bounds.

    ``score = reduce([beta * clip(vote, *bounds), ...])``
    defaults to ``mean``. As in
    ``asr_measure``, ``reduce`` accepts a mode string or a mapping; mapping
    options override reward kwargs (e.g. ``total`` for ``average``).
    Accuracy clipping, gamma transforms, cut gates, and accuracy scaling are
    not applied: tol defines the voting interval, not clipping bounds.

    Raw configured measures, ``d_<name>`` deltas, and ``v_<name>`` votes
    are returned alongside ``score``. Returned votes are clipped but unweighted:
    -1/0/+1 in threshold mode, the delta in delta mode, or the current measure
    value in absolute mode, followed by the configured clipping.
    These deltas use each measure's ``delta`` mode and
    are used for voting only in threshold and delta modes. With no baseline
    (None or empty), the score is zero and no delta or vote fields are returned,
    including for absolute measures. This lets the same scorer collect greedy
    baseline measurements without ReMax subtracting absolute measures again.
    This returns early,
    skipping vote and reduction configuration and computation. A nonempty baseline must
    contain every configured measure in the same score type: use this scorer
    with the same per-measure ``score`` settings, ``asr_measure`` for ``acc``, or
    ``asr_measure_v0`` for ``err``.
    Baselines are already in the selected representation and are not converted
    again. Only measure deltas are returned, never ``d_score``.

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
            word: {score: err, delta: rel, vote: delta, beta: 1.0, clip: [-1000, 100]}
            lang: {score: acc, delta: abs, vote: thr, beta: 0.5}

    Mixed delta/absolute example (YAML)::

        reward_kwargs:
          reduce: {mode: average, total: 4}
          measures:
            word: {delta: abs, vote: delta, beta: 1.0, clip: [-2, 2]}
            lang: {score: acc, vote: abs, beta: 0.1, clip: [-2, 2]}
            fmt: {score: acc, vote: abs, beta: 0.1, clip: [-2, 2]}
    """
    if "delta" in kwargs:
        raise ValueError("Configure ASR vote delta per measure, not as a shared reward kwarg.")
    if "vote" in kwargs:
        raise ValueError("Configure ASR vote mode per measure, not as a shared reward kwarg.")
    if "clip" in kwargs:
        raise ValueError("Configure ASR vote clipping per measure, not as shared reward kwargs.")
    if "tol" in kwargs:
        raise ValueError("Configure ASR vote tol per measure, not as a shared reward kwarg.")
    if "score" in kwargs:
        raise ValueError("Configure ASR vote score per measure, not as a shared reward kwarg.")

    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    measures = kwargs.get("measures") or {}
    unknown = set(measures) - parsed.keys()
    if unknown:
        raise ValueError(f"Unsupported ASR vote measures: {', '.join(sorted(unknown))}.")

    raw_scores = {}
    for name, cfg in measures.items():
        score_type = (cfg or {}).get("score", "acc")
        if score_type not in ("acc", "err"):
            raise ValueError(
                f"Unsupported ASR vote score type for {name!r}: {score_type!r}. Expected 'acc' or 'err'."
            )
        raw_scores[name] = parsed[name] - 1.0 if score_type == "err" else parsed[name]
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

"""LLM-judge-based reward for ASR formatting quality (punc/cap/digital).

Sends (baseline_hyp, target_hyp) pairs to a vLLM OpenAI-compatible server
and returns scores for punctuation, capitalization, and digit formatting,
along with character error rate (CER) computed locally via jiwer.

For ReMax training, the baseline reference is the greedy-decoded hypothesis
(passed via extra_info["greedy_hyp"]).

Usage as a reward function:
    from recipe.phimm.reward.fmt_llm_judge_reward import compute_score

    result = compute_score(
        solution_str=model_output,
        ground_truth=reference_text,
        server="http://localhost:8000",
        model="Qwen3.5-35B-A3B",
        extra_info={"greedy_hyp": greedy_hypothesis_text},
    )
    # result: {"score": ..., "cer": ..., "punc": ..., "cap": ..., "digital": ..., "fmt": ..., "lang": ...}
"""

from __future__ import annotations

from recipe.phimm.reward.fmt_llm_judge_client import query_judge
from recipe.phimm.reward.punc_cap_measure import (
    check_fmt,
    reduce_scores,
    scale_score,
    signed_pow,
)


# ---------------------------------------------------------------------------
# CER computation
# ---------------------------------------------------------------------------


def compute_cer(ref: str, hyp: str) -> float:
    """Compute character error rate between ref and hyp.

    Returns a value in [0, inf) where 0 is perfect.
    """
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()
    if not ref:
        return 0.0 if not hyp else 1.0

    from jiwer import cer
    return cer(ref, hyp)


# ---------------------------------------------------------------------------
# Reward / scoring
# ---------------------------------------------------------------------------


def judge_fmt(hyp_text: str, baseline: str, server: str, model: str) -> dict:
    """Query the LLM judge for formatting scores (punc/cap/digital).

    Parameters
    ----------
    hyp_text : str
        The hypothesis text to evaluate.
    baseline : str
        The baseline (greedy hyp or ground truth) to compare against.
    server : str
        vLLM server URL.
    model : str
        Served model name.

    Returns
    -------
    dict with keys punc, cap, digital (float 0-1).
    Scoring: 3=better, 2=similar, 1=worse → normalized 1.0, 0.5, 0.0.
    """
    scores = {"punc": 2, "cap": 2, "digital": 2}
    if hyp_text and baseline:
        import sys
        print(f"[fmt_llm_judge] Querying judge: server={server} model={model}", file=sys.stderr)
        scores = query_judge(baseline, hyp_text, server, model)

    return {
        "punc": (scores["punc"] - 1) / 2.0,
        "cap": (scores["cap"] - 1) / 2.0,
        "digital": (scores["digital"] - 1) / 2.0,
    }


def _parse_response(solution_str, ground_truth=None, **kwargs):
    """Extract hyp text, CER, and LLM judge scores (punc/cap/digital)."""
    from recipe.phimm.utils.shared import parse_asr_response

    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    trans_dict = parse_asr_response(solution_str)
    hyp_text = trans_dict["text"]
    pred_lang = (trans_dict["lang"] or "").lower().strip()
    is_nonspeech = (hyp_text or "").strip().lower() == "<nonspeech>"

    # CER (always computed locally)
    char_err = compute_cer(ground_truth or "", hyp_text or "")
    char_acc = max(0.0, 1.0 - char_err)

    # LLM judge formatting scores
    server = kwargs.get("server", "http://verl-n1-i5-0:8000")
    model = kwargs.get("model", "Qwen3.5-35B-A3B")
    baseline = extra_info.get("greedy_hyp") or ground_truth

    if is_nonspeech:
        fmt_scores = {"punc": 0.5, "cap": 0.5, "digital": 0.5}
    else:
        fmt_scores = judge_fmt(hyp_text or "", baseline, server, model)

    return {
        "char": char_acc,
        **fmt_scores,
        "lang": 1.0 if is_nonspeech else float(pred_lang == tgt_lang),
        "fmt": float(check_fmt(solution_str)),
    }


def compute_score(solution_str, ground_truth, **kwargs):
    """Combined CER + LLM-judge formatting reward.

    Calls an LLM judge server to evaluate punc/cap/digital quality of the
    hypothesis against a baseline, and always computes CER locally.

    For ReMax training, the baseline should be the greedy-decoded hypothesis.
    Pass it via ``extra_info["greedy_hyp"]`` (populated by ray_trainer).
    Falls back to ground_truth.

    If no greedy_hyp baseline is available, the LLM judge uses ground_truth
    as the comparison baseline.

    Configuration example (YAML)::

        reward_kwargs:
          server: "http://localhost:8000"
          model: "Qwen3.5-35B-A3B"
          reduce: sum
          measures:
            char: {beta: 1.0, gamma: 1.0}
            punc: {beta: 0.5, gamma: 1.0}
            cap: {beta: 0.3, gamma: 1.0}
            digital: {beta: 0.2, gamma: 1.0}

    Returns
    -------
    dict with keys: score, cer, char, punc, cap, digital, lang, fmt.
    """
    extra_info = kwargs.get("extra_info") or {}
    greedy_hyp = extra_info.get("greedy_hyp")
    assert greedy_hyp, (
        f"greedy_hyp must be provided in extra_info['greedy_hyp']. "
        f"Got keys: {list(extra_info.keys())}, kwargs keys: {list(kwargs.keys())}"
    )

    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    is_good = parsed["fmt"] > 0.0 and parsed["lang"] > 0.0

    measures = kwargs.get("measures") or {"char": {}}
    reduce = kwargs.get("reduce", "sum").lower()
    gamma = float(kwargs.get("gamma", 1.0))

    scores = [scale_score(parsed.get(k, 1.0), cfg) for k, cfg in measures.items()]
    score = reduce_scores(scores, reduce)
    score = signed_pow(score, gamma)

    return {
        "score": score if is_good else -1.0,
        **parsed,
    }

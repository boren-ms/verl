"""Unified reward function for ReMax ASR training on verl-mirror async trainer.

Computes:
- punc_cap_measure.compute_score → `score` (RL training signal)
- asr_inhouse_measure.eval_score → `dter_p_err` (monitoring metric, val only)

The `score` field is the actual RL reward (lightweight, fast).
The `dter_*` fields are computed only for validation data (expensive DTER via SpeechInsight).

Branching logic: If data_source contains "conv" or "dtest" (typical val corpus names),
or if extra_info has `is_val=True`, compute DTER. Otherwise skip it for speed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DTER_AVAILABLE = None


def _check_dter():
    """Lazy check whether DTER computation is available."""
    global _DTER_AVAILABLE
    if _DTER_AVAILABLE is None:
        try:
            from recipe.phimm.reward.asr_inhouse_measure import ensure_pack_dir
            ensure_pack_dir(None)
            _DTER_AVAILABLE = True
        except Exception as e:
            logger.warning("DTER not available (SpeechInsight setup failed): %s", e)
            _DTER_AVAILABLE = False
    return _DTER_AVAILABLE


def _is_val_sample(data_source: str, extra_info: dict) -> bool:
    """Determine if this sample should have DTER computed."""
    if extra_info.get("is_val"):
        return True
    ds = (data_source or "").lower()
    val_patterns = ("conv", "dtest", "eval", "test", "val")
    return any(p in ds for p in val_patterns)


def compute_score(solution_str=None, ground_truth=None, data_source=None, extra_info=None, **kwargs):
    """Unified reward: punc_cap for score + DTER for val monitoring.

    Returns a dict with keys:
      - score: RL reward signal (punc+cap accuracy mean, -1 on format/lang error)
      - char, punc, cap, lang, fmt: component accuracies
      - dter_p_err, dter_n_err, dter_n_ref: DTER metrics (val only, None otherwise)
    """
    from recipe.phimm.reward.punc_cap_measure import compute_score as punc_cap_score

    extra_info = extra_info or {}

    # Always compute the lightweight punc_cap reward
    result = punc_cap_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)

    # Compute DTER only for validation samples
    if _is_val_sample(data_source or "", extra_info) and _check_dter():
        try:
            from recipe.phimm.reward.asr_inhouse_measure import eval_score
            dter_result = eval_score(solution_str, ground_truth, extra_info=extra_info, **kwargs)
            result["dter_p_err"] = dter_result.get("dter", None)
            result["dter_n_err"] = dter_result.get("dter_n_err", None)
            result["dter_n_ref"] = dter_result.get("dter_n_ref", None)
            result["dter_n_punc"] = dter_result.get("dter_n_punc", None)
            result["dter_n_cap"] = dter_result.get("dter_n_cap", None)
            result["dter_n_lex"] = dter_result.get("dter_n_lex", None)
        except Exception as e:
            logger.debug("DTER computation failed for sample: %s", e)

    return result

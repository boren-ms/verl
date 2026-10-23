"""Joint keyword-list reward and count-based WER/BWER/UWER evaluation.

WER uses unit-cost jiwer alignment on ls-common5k-v1 word normalization.
Substitutions/deletions follow reference keyword positions; insertions follow
hypothesis membership in the reference keyword set, as in the biasing scorer.
BWER and UWER are corpus ratios of summed counts, never sentence averages.
"""

from jiwer import process_words

from recipe.phimm.data.keyword_output import PROTOCOL, parse_keyword_output, words
from recipe.phimm.reward.asr_response import get_hyp_text


def parse_response(solution_str, extra_info=None, version=2609):
    extra = extra_info or {}
    # The actual assistant prefix is already part of model input, not completion.
    response = (extra.get("prefix") or "") + solution_str
    parsed = None
    if extra.get("output_protocol") == PROTOCOL:
        parsed = parse_keyword_output(response, extra.get("generation_finish_reason"))
        response = parsed.body_raw
    return get_hyp_text(response, version=version), parsed


def word_error_counts(reference, hypothesis, keywords):
    ref, hyp = words(reference), words(hypothesis)
    keyset = set(keywords)
    aligned = process_words(" ".join(ref), " ".join(hyp))
    b_errors = 0
    for chunk in aligned.alignments[0]:
        if chunk.type in {"substitute", "delete"}:
            b_errors += sum(w in keyset for w in ref[chunk.ref_start_idx : chunk.ref_end_idx])
        elif chunk.type == "insert":
            b_errors += sum(w in keyset for w in hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
    n_err = aligned.substitutions + aligned.deletions + aligned.insertions
    nb_ref = sum(w in keyset for w in ref)
    return {
        "n_err": n_err,
        "n_ref": len(ref),
        "nb_err": b_errors,
        "nb_ref": nb_ref,
        "nu_err": n_err - b_errors,
        "nu_ref": len(ref) - nb_ref,
    }


def list_scores(parsed, targets):
    predicted = set(parsed.predicted_keywords)
    target = set(targets)
    tp, fp, fn = len(predicted & target), len(predicted - target), len(target - predicted)
    valid = parsed.valid
    if not valid:
        # Invalid output never earns the empty-set special case.
        tp, fp, fn = 0, len(predicted), len(target)
    if valid and not predicted and not target:
        precision = recall = f1 = 1.0
    elif not valid or not predicted or not target:
        precision = recall = f1 = 0.0
    else:
        precision, recall = tp / len(predicted), tp / len(target)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "kw_list_precision": precision,
        "kw_list_recall": recall,
        "kw_list_f1": f1,
        "kw_list_exact": float(valid and predicted == target),
        "kw_block_fmt": float(valid),
        "kw_tp": tp,
        "kw_fp": fp,
        "kw_fn": fn,
        "kw_empty_target": int(not target),
        "kw_empty_false_positive": int(not target and bool(predicted)),
        "kw_empty_protocol_failure": int(not target and not valid),
        "kw_missing": float(parsed.keyword_block_status == "missing"),
        "kw_truncated": float(parsed.keyword_block_status == "truncated"),
    }


def eval_score(solution_str, ground_truth, **kwargs):
    extra = kwargs.get("extra_info") or {}
    targets = extra.get("keywords")
    if not isinstance(targets, list) or any(not isinstance(w, str) or not w for w in targets):
        raise ValueError("keyword scoring requires validated keywords (explicit [] for empty targets)")
    hyp, parsed = parse_response(solution_str, extra, kwargs.get("version", 2609))
    counts = word_error_counts(ground_truth, hyp, targets)
    result = {"score": 1 - counts["n_err"] / max(counts["n_ref"], 1), **counts}
    metadata = {"parsed_transcription": hyp}
    if parsed is not None:
        if not extra.get("keyword_annotation_valid"):
            raise ValueError("joint scoring requires keyword_annotation_valid=true")
        result.update(list_scores(parsed, targets))
        metadata.update(
            predicted_keywords=parsed.predicted_keywords,
            keyword_block_status=parsed.keyword_block_status,
            protocol_errors=parsed.protocol_errors,
        )
        predicted = set(parsed.predicted_keywords)
        result["kw_in_transcript"] = float(predicted <= set(words(hyp)))
    result["extra_info"] = metadata
    return result


def compute_score(solution_str, ground_truth, list_weight=0.5, format_weight=0.1, **kwargs):
    result = eval_score(solution_str, ground_truth, **kwargs)
    # Keep the existing ASR reward on parsed transcription and the same scale.
    from recipe.phimm.reward.asr_measure import compute_score as asr_score

    extra = dict(kwargs.get("extra_info") or {})
    hyp, parsed = parse_response(solution_str, extra, kwargs.get("version", 2609))
    if parsed is None:
        raise ValueError("joint training requires asr-keywords-v1 protocol")
    # Preserve header/mode scoring; never pass the legal list into legacy scoring.
    asr_body = parsed.body_raw
    old_kwargs = {**kwargs, "extra_info": {**extra, "prefix": ""}}
    original = asr_score(asr_body, ground_truth, **old_kwargs)
    result["asr_score"] = original["score"]
    result["score"] = original["score"] + list_weight * result["kw_list_f1"] + format_weight * result["kw_block_fmt"]
    result.update({k: v for k, v in original.items() if k != "score"})
    return result

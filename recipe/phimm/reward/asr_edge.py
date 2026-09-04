# %%
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from recipe.phimm.reward.asr_eval import openasr_eval as openasr_eval
from recipe.phimm.reward.asr_response import get_hyp_text, parse_task_output
from recipe.phimm.utils.shared import has_brackets, has_missing_keyword, has_repeat_error, has_tail_hallucination
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_compound_pairs


@dataclass
class Error:
    n_sub: int = 0
    n_del: int = 0
    n_ins: int = 0
    n_hit: int = 0
    n_edge: int = 0

    @property
    def n_ref(self):
        return self.n_sub + self.n_del + self.n_hit

    @property
    def n_err(self):
        return self.n_sub + self.n_del + self.n_ins

    def accuracy(self, **betas):
        return 1 - self.wer(**betas)

    def wer(self, **betas):
        n_edit = self.edit_distance(**betas)
        return n_edit / max(self.n_ref, 1)

    def edit_distance(self, **betas):
        w_ins = betas.get("ins", 1.0)
        w_sub = betas.get("sub", 1.0)
        w_del = betas.get("del", 1.0)
        w_edge = betas.get("edge", 0.0)
        return w_sub * self.n_sub + w_del * self.n_del + w_ins * self.n_ins + w_edge * self.n_edge

    def edge_wer(self):
        return self.n_edge / max(self.n_ref, 1)

    def __add__(self, other):
        if not isinstance(other, Error):
            return NotImplemented
        return Error(
            n_sub=self.n_sub + other.n_sub,
            n_del=self.n_del + other.n_del,
            n_ins=self.n_ins + other.n_ins,
            n_hit=self.n_hit + other.n_hit,
            n_edge=self.n_edge + other.n_edge,
        )


def _norm_text(text, name=None, lang="english"):

    from recipe.phimm.utils.tn import default_tn_name, text_norm as apply_text_norm

    name = name or default_tn_name(lang)
    return apply_text_norm(text.strip(), name=name)


def _count_ops(ref_words, hyp_words):
    err = Error()
    n_ref = len(ref_words)
    sm = SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        r_len, h_len = i2 - i1, j2 - j1
        is_edge = i1 == 0 or i2 == n_ref
        if tag == "equal":
            err.n_hit += r_len
            continue
        if tag == "delete":
            err.n_del += r_len
            if is_edge:
                err.n_edge += r_len
            continue
        if tag == "insert":
            err.n_ins += h_len
            if is_edge:
                err.n_edge += h_len
            continue
        if tag == "replace":
            err.n_sub += min(r_len, h_len)
            if r_len > h_len:
                err.n_del += r_len - h_len
            elif h_len > r_len:
                err.n_ins += h_len - r_len
            if is_edge:
                err.n_edge += max(r_len, h_len)

    return err


def _split_units(text, unit="word"):
    if unit.lower() in {"word", "words"}:
        return text.split()
    if unit.lower() in {"char", "chars"}:
        text = re.sub(r"\s+", "", text)
        return list(text)
    raise ValueError(f"Unsupported error measure unit: {unit!r}. Expected 'word' or 'char'.")


def acc_to_bucket(acc, n_buckets=10, lo=0.8, hi=1.0):
    """Map accuracy in [lo, hi] to a normalized score in [0, 1] using log-spaced buckets.

    Buckets are finer near hi (1.0) and coarser near lo (0.8).
    Returns bucket_index / n_buckets so the score is in [0, 1].
    Accuracy <= lo maps to 0, accuracy >= hi maps to 1.
    """
    if acc >= hi:
        return 1.0
    if acc <= lo:
        return 0.0
    span = hi - lo
    # distances from hi, log-spaced from span down to span/100
    distances = [span * 10 ** (-2 * i / n_buckets) for i in range(n_buckets + 1)]
    boundaries = sorted(hi - d for d in distances)
    # clamp endpoints
    boundaries[0] = lo
    boundaries[-1] = hi
    for i in range(n_buckets):
        if acc < boundaries[i + 1]:
            return i / n_buckets
    return (n_buckets - 1) / n_buckets


def measure(hyp, ref, tgt_lang="english", **kwargs):
    norm_name = kwargs.get("text_norm")
    unit = kwargs.get("unit", "word")
    compound_norm = kwargs.get("compound_norm", False)
    ref_text = _norm_text(ref, name=norm_name, lang=tgt_lang)
    hyp_text = _norm_text(hyp, name=norm_name, lang=tgt_lang)
    if compound_norm:
        [ref_text], [hyp_text] = normalize_compound_pairs([ref_text], [hyp_text])
    ref_units = _split_units(ref_text, unit=unit)
    hyp_units = _split_units(hyp_text, unit=unit)
    return _count_ops(ref_units, hyp_units)


def _parse_response(solution_str, ground_truth=None, **kwargs):
    """Extract ASR text and validation signals from a model response."""
    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower().strip()
    version = kwargs.get("version")
    task_output = parse_task_output(solution_str, version=version)
    hyp_text = get_hyp_text(solution_str, version=version)
    lang_index = 1 if str(version) == "2607" else 0
    pred_langs = task_output[lang_index] if task_output is not None else []
    pred_lang = (pred_langs[0] if pred_langs else "").lower().strip()
    is_formatted = task_output is not None and (str(version) == "2607" or bool(task_output[0]))
    repeat_opts = kwargs.get("repeat") or {}
    p_repeat = has_repeat_error(
        hyp_text,
        ground_truth,
        min_reps=repeat_opts.get("min_reps", 4),
        max_ngram=repeat_opts.get("max_ngram", 5),
        tn_name=kwargs.get("text_norm"),
        lang=tgt_lang,
    )
    keyword_opts = kwargs.get("keyword_missing") or {}
    keywords = extra_info.get("keywords") or []
    keyword_norm = keyword_opts.get("text_norm", None)
    p_kw_missing = has_missing_keyword(keywords, hyp_text, norm_name=keyword_norm, lang=tgt_lang)
    tail_opts = kwargs.get("tail_hallucination") or {}
    p_tail_hallu = has_tail_hallucination(
        hyp_text,
        ground_truth,
        min_words=tail_opts.get("min_words", 3),
        tn_name=tail_opts.get("text_norm"),
        lang=tgt_lang,
    )

    is_nonspeech = (hyp_text or "").strip().lower() == "<nonspeech>"

    return {
        "hyp_text": hyp_text,
        "tgt_lang": tgt_lang,
        "p_lang": 1.0 if is_nonspeech else float(pred_lang == tgt_lang),
        "p_fmt": float(is_formatted),
        "p_bracket": float(has_brackets(hyp_text)),
        "p_repeat": float(p_repeat),
        "p_kw_missing": float(p_kw_missing),
        "p_tail_hallu": float(p_tail_hallu),
    }


# Mapping from check name -> (parsed key, expected_truthy)
# expected_truthy=True means the check passes when parsed[key] is truthy (e.g. lang, fmt match)
# expected_truthy=False means the check passes when parsed[key] is falsy (e.g. no bracket/repeat)
_CHECK_SPEC = {
    "lang": ("p_lang", True),
    "fmt": ("p_fmt", True),
    "bracket": ("p_bracket", False),
    "repeat": ("p_repeat", False),
    "keyword": ("p_kw_missing", False),
    "tail_hallu": ("p_tail_hallu", False),
}

DEFAULT_CHECKS = ("fmt", "lang", "bracket", "repeat", "tail_hallu")


def _is_good_response(parsed, checks=DEFAULT_CHECKS):
    for name in checks:
        if name not in _CHECK_SPEC:
            raise ValueError(f"Unknown check: {name!r}. Expected one of {list(_CHECK_SPEC)}.")
        key, expected_truthy = _CHECK_SPEC[name]
        ok = bool(parsed[key]) if expected_truthy else not bool(parsed[key])
        if not ok:
            return False
    return True


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def signed_pow(x, gamma):
    sign = 1 if x >= 0 else -1
    return (abs(x) ** gamma) * sign


def compute_score(solution_str, ground_truth, **kwargs):
    """ASR reward with regular WER and insertion-sensitive penalties."""
    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    err = measure(parsed["hyp_text"], ground_truth, tgt_lang=parsed["tgt_lang"], **kwargs)
    betas = kwargs.get("betas", {})
    gamma = kwargs.get("gamma", 1)
    checks = kwargs.get("checks", DEFAULT_CHECKS)
    is_good = _is_good_response(parsed, checks=checks)

    score = signed_pow(err.accuracy(**betas), gamma)
    # good (-1, 1), bad: -2
    score = clip(score, -1.0, 1.0) if is_good else -2.0

    return {
        "score": score,
        "wer": err.wer(**betas),
        "n_ref": err.n_ref,
        "n_err": err.n_err,
        "n_edge": err.n_edge,
        "p_fmt": parsed["p_fmt"],
        "p_lang": parsed["p_lang"],
        "p_bracket": parsed["p_bracket"],
        "p_repeat": parsed["p_repeat"],
        "p_kw_missing": parsed["p_kw_missing"],
        "p_tail_hallu": parsed["p_tail_hallu"],
    }


def eval_score(solution_str, ground_truth, **kwargs):
    """Validation scoring: returns raw error counts for aggregation."""
    parsed = _parse_response(solution_str, ground_truth=ground_truth, **kwargs)
    err = measure(parsed["hyp_text"], ground_truth, tgt_lang=parsed["tgt_lang"], **kwargs)

    return {
        "score": err.accuracy(),
        "wer": err.wer(),
        "n_err": err.n_err,
        "n_ref": err.n_ref,
        "n_edge": err.n_edge,
        "p_fmt": parsed["p_fmt"],
        "p_lang": parsed["p_lang"],
        "p_bracket": parsed["p_bracket"],
        "p_repeat": parsed["p_repeat"],
        "p_kw_missing": parsed["p_kw_missing"],
        "p_tail_hallu": parsed["p_tail_hallu"],
    }


# %%
if __name__ == "__main__":
    test_cases = [
        # {
        #     "name": "perfect_match",
        #     "ref": "turn on the living room lights",
        #     "hyp": "turn on the living room lights",
        # },
        # {
        #     "name": "mid_insertion",
        #     "ref": "play jazz music now",
        #     "hyp": "play some jazz music now",
        # },
        # {
        #     "name": "head_insertion",
        #     "ref": "set timer for ten minutes",
        #     "hyp": "please set timer for ten minutes",
        # },
        # {
        #     "name": "tail_insertion",
        #     "ref": "open calendar",
        #     "hyp": "open calendar now please",
        # },
        # {
        #     "name": "tail_insertion",
        #     "ref": "open calendar",
        #     "hyp": "open calendr now please",
        # },
        {
            "name": "tail_insertion",
            "ref": "open calendar",
            "hyp": "open some calendar now please",
        },
        {
            "name": "mixed_errors",
            "ref": "book a table for two tonight",
            "hyp": "please book table for three tonight now",
        },
    ]

    print("== asr_edge self-test ==")
    print("default weights: beta=1.0, ins_beta=0.25, edge_ins_beta=0.75")
    for tc in test_cases:
        out = compute_score(tc["hyp"], tc["ref"], text_norm="english")
        print(f"\n[{tc['name']}]")
        print(f"ref: {tc['ref']}")
        print(f"hyp: {tc['hyp']}")
        print(
            "metrics:",
            {
                "score": out["score"],
                "wer": out["wer"],
                "n_err": out["n_err"],
                "n_edge": out["n_edge"],
            },
        )

# %%

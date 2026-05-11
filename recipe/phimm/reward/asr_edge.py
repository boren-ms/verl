# %%
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging
import re
from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.shared import has_brackets, parse_asr_response
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_compound_pairs


logger = logging.getLogger(__name__)


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
        return max(1 - self.wer(**betas), 0.0)

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


def _norm_text(text, norm_name="english"):

    from recipe.phimm.utils.tn import text_norm as apply_text_norm

    return apply_text_norm(text.strip(), name=norm_name)


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
    default_norm = "openasr_en" if tgt_lang.lower() == "english" else "openasr_ml"
    norm_name = kwargs.get("text_norm", default_norm)
    unit = kwargs.get("unit", "word")
    compound_norm = kwargs.get("compound_norm", False)
    ref_text = _norm_text(ref, norm_name=norm_name)
    hyp_text = _norm_text(hyp, norm_name=norm_name)
    if compound_norm:
        [ref_text], [hyp_text] = normalize_compound_pairs([ref_text], [hyp_text])
    ref_units = _split_units(ref_text, unit=unit)
    hyp_units = _split_units(hyp_text, unit=unit)
    return _count_ops(ref_units, hyp_units)


def _parse_response(solution_str, **kwargs):
    """Shared parsing: extract target language, parse ASR response, check lang/format."""
    extra_info = kwargs.get("extra_info") or {}
    tgt_lang = extra_info.get("language", kwargs.get("language", "English")).lower()
    trans_dict = parse_asr_response(solution_str)
    hyp_text = trans_dict["text"]
    pred_lang = (trans_dict["lang"] or "").lower()
    is_lang = pred_lang == tgt_lang
    is_fmt = bool(trans_dict["formatted"])

    return hyp_text, tgt_lang, is_lang, is_fmt, has_brackets(hyp_text)


def compute_score(solution_str, ground_truth, **kwargs):
    """ASR reward with regular WER and insertion-sensitive penalties."""
    hyp_text, tgt_lang, is_lang, is_fmt, p_bracket = _parse_response(solution_str, **kwargs)

    err = measure(hyp_text, ground_truth, tgt_lang=tgt_lang, **kwargs)
    betas = kwargs.get("betas", {})
    metric = kwargs.get("metric", "acc")  # wer, acc, ed
    gamma = kwargs.get("gamma", 1)

    is_good = is_fmt and not p_bracket # skip lang check, due to inaccurate lang tag#

    if metric == "wer":
        wer = err.wer(**betas)
        score = -(wer**gamma) if is_good else -1
    elif metric == "ed":
        n_edit = err.edit_distance(**betas)
        score = -(n_edit**gamma)
    elif metric == "bucket":
        acc = err.accuracy(**betas)
        n_buckets = kwargs.get("n_buckets", 10)
        bucket_lo = kwargs.get("bucket_lo", 0.8)
        score = acc_to_bucket(acc, n_buckets=n_buckets, lo=bucket_lo)
        score = score if is_good else -1
    else:
        score = err.accuracy(**betas) ** gamma
        score = score if is_good else -1

    return {
        "score": score,
        "n_ref": err.n_ref,
        "n_err": err.n_err,
        "n_edge": err.n_edge,
        "p_fmt": float(is_fmt),
        "p_lang": float(is_lang),
        "p_bracket": float(p_bracket),
    }


def eval_score(solution_str, ground_truth, **kwargs):
    """Validation scoring: returns raw error counts for aggregation."""
    hyp_text, tgt_lang, is_lang, is_fmt, p_bracket = _parse_response(solution_str, **kwargs)
    err = measure(hyp_text, ground_truth, tgt_lang=tgt_lang, **kwargs)

    return {
        "score": err.accuracy(),
        "wer": err.wer(),
        "edge_wer": err.edge_wer(),
        "n_err": err.n_err,
        "n_ref": err.n_ref,
        "n_edge": err.n_edge,
        "p_fmt": float(is_fmt),
        "p_lang": float(is_lang),
        "p_bracket": float(p_bracket),
    }


def openasr_eval(solution_str, ground_truth, **kwargs):
    """Evaluation using OpenASR normalizers + editdistance WER.

    Matches HFWerScorer from phyagi/eval/utils/score_utils.py.
    """
    from recipe.phimm.utils.open_asr_normalizer.eval_utils import measure_wer

    hyp_text, tgt_lang, is_lang, is_fmt, p_bracket = _parse_response(solution_str, **kwargs)
    logger.info("openasr_eval language check: tgt_lang=%s is_lang=%s", tgt_lang, is_lang)

    result = measure_wer(hyp_text, ground_truth, lang=get_language_code(tgt_lang))
    return {
        "score": 1.0 - result["wer"],
        "wer": result["wer"],
        "n_err": result["n_err"],
        "n_ref": result["n_ref"],
        "p_fmt": float(is_fmt),
        "p_lang": float(is_lang),
        "p_bracket": float(p_bracket),
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

# %%
from dataclasses import dataclass
from difflib import SequenceMatcher

from recipe.phimm.utils.shared import parse_asr_response


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

    @property
    def accuracy(self):
        return (self.n_ref - self.n_err) / max(self.n_ref, 1)

    @property
    def wer(self):
        return self.n_err / max(self.n_ref, 1)

    @property
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


def measure(hyp, ref, **kwargs):
    norm_name = kwargs.get("text_norm", "english")
    ref = _norm_text(ref, norm_name=norm_name)
    hyp = _norm_text(hyp, norm_name=norm_name)
    return _count_ops(ref.split(), hyp.split())


def compute_score(solution_str, ground_truth, **kwargs):
    """ASR reward with regular WER and insertion-sensitive penalties."""
    trans_dict = parse_asr_response(solution_str)

    lang = (trans_dict["lang"] or "").lower() == "english"
    format = bool(trans_dict["formatted"])

    err = measure(trans_dict["text"], ground_truth, **kwargs)

    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 1.0)
    cutoff = kwargs.get("cutoff", 0.0)
    gamma = kwargs.get("gamma", 1)

    nw_err = alpha * err.n_err + beta * err.n_edge
    acc = (err.n_ref - nw_err) / max(err.n_ref, 1)  # in case zero n_ref

    if lang and format and acc > cutoff:
        score = ((acc - cutoff) / (1 - cutoff)) ** gamma
    else:
        score = 0.0

    return {
        "score": score,
        "n_ref": err.n_ref,
        "n_err": err.n_err,
        "n_edge": err.n_edge,
        "p_fmt": float(format),
        "p_lang": float(lang),
    }


def eval_score(solution_str, ground_truth, **kwargs):
    """Validation scoring: returns raw error counts for aggregation."""
    trans_dict = parse_asr_response(solution_str)
    lang = (trans_dict["lang"] or "").lower() == "english"
    format = bool(trans_dict["formatted"])

    err = measure(trans_dict["text"], ground_truth, **kwargs)
    return {
        "score": err.accuracy,
        "n_err": err.n_err,
        "n_ref": err.n_ref,
        "n_edge": err.n_edge,
        "p_fmt": float(format),
        "p_lang": float(lang),
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

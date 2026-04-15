# %%
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class Error:
    n_sub: int = 0
    n_del: int = 0
    n_ins: int = 0
    n_hit: int = 0
    n_ins_edge: int = 0

    @property
    def n_ref(self):
        return self.n_sub + self.n_del + self.n_hit

    @property
    def n_err(self):
        return self.n_sub + self.n_del + self.n_ins

    @property
    def accuracy(self):
        return self.n_hit / self.n_ref if self.n_ref > 0 else 0.0

    @property
    def wer(self):
        return self.n_err / self.n_ref if self.n_ref > 0 else 0.0



def _norm_text(text, norm_name="english"):
    
    from recipe.phimm.utils.tn import text_norm as apply_text_norm
    norm = apply_text_norm(name=norm_name)
    # from whisper_normalizer.english import EnglishTextNormalizer
    # norm = EnglishTextNormalizer()
    return norm(text.strip())


def _count_ops(ref_words, hyp_words):
    err = Error()
    n_ref = len(ref_words)
    sm = SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        r_len, h_len = i2 - i1, j2 - j1
        if tag == "equal":
            err.n_hit += r_len
            continue
        if tag == "delete":
            err.n_del += r_len
            continue
        if tag == "insert":
            err.n_ins += h_len
            if i1 == 0 or i1 == n_ref:
                err.n_ins_edge += h_len
            continue
        if tag == "replace":
            shared = min(r_len, h_len)
            err.n_sub += shared
            if r_len > h_len:
                err.n_del += r_len - h_len
            elif h_len > r_len:
                extra = h_len - r_len
                err.n_ins += extra
                if i1 == 0 or i2 == n_ref:
                    err.n_ins_edge += extra

    return err


def measure(hyp, ref, **kwargs):
    norm_name = kwargs.get("text_norm", "english")
    ref = _norm_text(ref, norm_name=norm_name)
    hyp = _norm_text(hyp, norm_name=norm_name)
    return _count_ops(ref.split(), hyp.split())


def compute_score(solution_str, ground_truth, **kwargs):
    """ASR reward with regular WER and insertion-sensitive penalties."""
    err = measure(solution_str, ground_truth, **kwargs)

    beta = kwargs.get("beta", 1.0)
    edge_beta = kwargs.get("edge_beta", 1.0)
    cutoff = kwargs.get("cutoff", 0.3)

    weighted_err = beta * err.n_err + edge_beta * err.n_ins_edge
    acc = (err.n_ref - weighted_err) / err.n_ref

    score = (acc - cutoff) / (1 - cutoff) if acc > cutoff else 0.0

    return {
        "score": score,
        "wer": err.wer,
        "n_ref": err.n_ref,
        "n_err": err.n_err,
        "n_ins_edge": err.n_ins_edge,
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

    print("== asr_insertion self-test ==")
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
                "n_ins_edge": out["n_ins_edge"],
            },
        )

# %%

from jiwer import process_words
from dataclasses import dataclass
from recipe.phimm.utils.tn import text_norm as apply_text_norm

@dataclass
class Error:
    S: int  # substitution
    D: int  # deletion
    I: int  # insertion
    H: int  # hit

    @property
    def total(self):
        return self.S + self.D + self.H

    @property
    def count(self):
        return self.S + self.D + self.I

    @property
    def accuracy(self):
        if self.total == 0:
            return 0.0
        return self.H / self.total

    @property
    def wer(self):
        if self.total == 0:
            return 0.0
        return self.count / self.total


def measure(hyp, ref, **kwargs):
    norm = kwargs.get("text_norm", "english")
    ref = apply_text_norm(ref, name=norm)
    hyp = apply_text_norm(hyp, name=norm)
    
    output = process_words(ref, hyp)
    return Error(S=output.substitutions, D=output.deletions, I=output.insertions, H=output.hits)


def compute_score(solution_str, ground_truth, **kwargs):
    """The scoring function for Speech Recognition."""
    error = measure(solution_str, ground_truth, **kwargs)
    beta = kwargs.get("beta", 1.0)
    score = -error.count * beta
    return {
        "score": score,
        "wer": error.wer,
        "acc": error.accuracy,
        "n_ref": error.total,
        "n_hit": error.H,
        "n_err": error.count,
        "n_del": error.D,
        "n_ins": error.I,
        "n_sub": error.S,
    }


def wer(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth, **kwargs)
    return -error.wer


def accuracy(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth, **kwargs)
    return error.accuracy


def error_count(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth, **kwargs)
    return -error.count


def hit_count(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth, **kwargs)
    return error.H

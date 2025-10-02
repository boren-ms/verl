from jiwer import transforms as tr
from jiwer import process_words
from dataclasses import dataclass


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
    tn = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            tr.RemovePunctuation(),
            tr.RemoveKaldiNonWords(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    output = process_words(
        truth=ref,
        hypothesis=hyp,
        truth_transform=tn,
        hypothesis_transform=tn,
    )
    return Error(S=output.substitutions, D=output.deletions, I=output.insertions, H=output.hits)


def compute_score(solution_str, ground_truth, **kwargs):
    """The scoring function for Speech Recognition."""
    # breakpoint()
    error = measure(solution_str, ground_truth)
    return 0 - error.wer


def wer(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth)
    return 0 - error.wer


def accuracy(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth)
    return error.accuracy


def error_count(solution_str, ground_truth, **kwargs):
    error = measure(solution_str, ground_truth)
    return 0 - error.count

# %%
import re
from enum import Enum
from functools import partial
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parents[3]))

from recipe.phimm.utils.tn import text_norm as text_normalize
from recipe.phimm.utils.shared import parse_asr_response
from collections import deque


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class AlignmentResult(object):
    def __init__(self, refs, hyps, codes, score):
        self.refs = refs  # deque<int>
        self.hyps = hyps  # deque<int>
        self.codes = codes  # deque<Code>
        self.score = score  # float


class WordError(object):
    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.n_ref = 0

    @property
    def n_err(self):
        return sum(self.errors.values())

    @property
    def wer(self):
        return self.n_err / self.n_ref if self.n_ref > 0 else 0.0

    @property
    def acc(self):
        return 1.0 - self.wer

    @property
    def n_sub(self):
        return self.errors[Code.substitution]

    @property
    def n_ins(self):
        return self.errors[Code.insertion]

    @property
    def n_del(self):
        return self.errors[Code.deletion]

    def get_result_string(self):
        return f"WER={self.wer * 100}, refs={self.n_ref}, subs={self.n_sub}, ins={self.n_ins}, dels={self.n_del}"
    
    def __add__(self, other):
        if not isinstance(other, WordError):
            return NotImplemented
        if other.n_ref == 0:
            return self
        result = WordError()
        result.n_ref = self.n_ref + other.n_ref
        for code in self.errors:
            result.errors[code] = self.errors[code] + other.errors[code]
        return result


def coordinate_to_offset(row, col, ncols):
    return int(row * ncols + col)


def offset_to_row(offset, ncols):
    return int(offset / ncols)


def offset_to_col(offset, ncols):
    return int(offset % ncols)


class EditDistance(object):
    def __init__(self):
        self.scores_ = None
        self.backtraces_ = None
        self.confusion_pairs_ = {}
        self.inserted_words_ = {}
        self.deleted_words_ = {}

    def cost(self, ref, hyp, code):
        if code == Code.match:
            return 0
        elif code == Code.insertion or code == Code.deletion:
            return 3
        else:  # substitution
            return 4

    def get_result(self, refs, hyps):
        res = AlignmentResult(refs=deque(), hyps=deque(), codes=deque(), score=None)

        num_rows, num_cols = len(self.scores_), len(self.scores_[0])
        res.score = self.scores_[num_rows - 1][num_cols - 1]

        curr_offset = coordinate_to_offset(num_rows - 1, num_cols - 1, num_cols)

        while curr_offset != 0:
            curr_row = offset_to_row(curr_offset, num_cols)
            curr_col = offset_to_col(curr_offset, num_cols)

            prev_offset = self.backtraces_[curr_row][curr_col]

            prev_row = offset_to_row(prev_offset, num_cols)
            prev_col = offset_to_col(prev_offset, num_cols)

            res.refs.appendleft(curr_row - 1)
            res.hyps.appendleft(curr_col - 1)
            if curr_row - 1 == prev_row and curr_col == prev_col:
                ref_str = refs[res.refs[0]]
                deleted_word = ref_str
                if deleted_word not in self.deleted_words_:
                    self.deleted_words_[deleted_word] = 1
                else:
                    self.deleted_words_[deleted_word] += 1

                res.codes.appendleft(Code.deletion)

            elif curr_row == prev_row and curr_col - 1 == prev_col:
                hyp_str = hyps[res.hyps[0]]
                inserted_word = hyp_str
                if inserted_word not in self.inserted_words_:
                    self.inserted_words_[inserted_word] = 1
                else:
                    self.inserted_words_[inserted_word] += 1

                res.codes.appendleft(Code.insertion)

            else:
                # assert(curr_row - 1 == prev_row and curr_col - 1 == prev_col)
                ref_str = refs[res.refs[0]]
                hyp_str = hyps[res.hyps[0]]

                if ref_str == hyp_str:
                    res.codes.appendleft(Code.match)
                else:
                    res.codes.appendleft(Code.substitution)

                    confusion_pair = "%s -> %s" % (ref_str, hyp_str)
                    if confusion_pair not in self.confusion_pairs_:
                        self.confusion_pairs_[confusion_pair] = 1
                    else:
                        self.confusion_pairs_[confusion_pair] += 1

            curr_offset = prev_offset

        return res

    def align(self, refs, hyps):
        if len(refs) == 0 and len(hyps) == 0:
            raise ValueError("Doesn't support empty ref AND hyp!")

        # NOTE: we're not resetting the values in these matrices because every value
        # will be overridden in the loop below. If this assumption doesn't hold,
        # be sure to set all entries in self.scores_ and self.backtraces_ to 0.
        self.scores_ = [[0.0] * (len(hyps) + 1) for _ in range(len(refs) + 1)]
        self.backtraces_ = [[0] * (len(hyps) + 1) for _ in range(len(refs) + 1)]

        num_rows, num_cols = len(self.scores_), len(self.scores_[0])

        for i in range(num_rows):
            for j in range(num_cols):
                if i == 0 and j == 0:
                    self.scores_[i][j] = 0.0
                    self.backtraces_[i][j] = 0
                    continue

                if i == 0:
                    self.scores_[i][j] = self.scores_[i][j - 1] + self.cost(None, hyps[j - 1], Code.insertion)
                    self.backtraces_[i][j] = coordinate_to_offset(i, j - 1, num_cols)
                    continue

                if j == 0:
                    self.scores_[i][j] = self.scores_[i - 1][j] + self.cost(refs[i - 1], None, Code.deletion)
                    self.backtraces_[i][j] = coordinate_to_offset(i - 1, j, num_cols)
                    continue

                # Below here both i and j are greater than 0
                ref = refs[i - 1]
                hyp = hyps[j - 1]
                best_score = self.scores_[i - 1][j - 1] + (
                    self.cost(ref, hyp, Code.match) if ref == hyp else self.cost(ref, hyp, Code.substitution)
                )

                prev_row = i - 1
                prev_col = j - 1
                ins = self.scores_[i][j - 1] + self.cost(None, hyp, Code.insertion)
                if ins < best_score:
                    best_score = ins
                    prev_row = i
                    prev_col = j - 1

                delt = self.scores_[i - 1][j] + self.cost(ref, None, Code.deletion)
                if delt < best_score:
                    best_score = delt
                    prev_row = i - 1
                    prev_col = j

                self.scores_[i][j] = best_score
                self.backtraces_[i][j] = coordinate_to_offset(prev_row, prev_col, num_cols)

        return self.get_result(refs, hyps)


def find_word_indices(text, pieces):
    indexs = []
    for piece in pieces:
        for m in re.finditer(re.escape(piece), text):
            start_idx = len(text[: m.start()].split())  # previous words
            piece_len = len(piece.split())
            indexs.extend(range(start_idx, start_idx + piece_len))
    return indexs


def find_char_indices(text, pieces):
    indexs = []
    for piece in pieces:
        for m in re.finditer(re.escape(piece), text):
            indexs.extend(range(m.start(), m.end()))
    return indexs


def calc_errors(refs, hyps, tn=None, unit=None):
    """Calculate WER, U-WER, and B-WER."""
    wer = WordError()
    u_wer = WordError()
    b_wer = WordError()
    for uttid, ref in refs.items():
        if uttid not in hyps:
            continue
        norm = partial(text_normalize, name=tn)
        # Normalize reference and hypothesis text
        tn_ref = norm(ref.get("text", ""))
        tn_hyp = norm(hyps[uttid])
        keywords = norm(ref.get("keywords", []))

        if unit == "char":
            kwd_ref_idxs = find_char_indices(tn_ref, keywords)
            kwd_hyp_idxs = find_char_indices(tn_hyp, keywords)

            ref_words = list(tn_ref)
            hyp_words = list(tn_hyp)
        else:  # default to words
            kwd_ref_idxs = find_word_indices(tn_ref, keywords)
            kwd_hyp_idxs = find_word_indices(tn_hyp, keywords)

            ref_words = tn_ref.split()
            hyp_words = tn_hyp.split()

        if len(ref_words) == 0 and len(hyp_words) == 0:
            continue

        ed = EditDistance()
        result = ed.align(ref_words, hyp_words)
        for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
            if code == Code.match:
                wer.n_ref += 1
                if ref_idx in kwd_ref_idxs:
                    b_wer.n_ref += 1
                else:
                    u_wer.n_ref += 1
            elif code == Code.substitution:
                wer.n_ref += 1
                wer.errors[Code.substitution] += 1
                if ref_idx in kwd_ref_idxs:
                    b_wer.n_ref += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.n_ref += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.n_ref += 1
                wer.errors[Code.deletion] += 1
                if ref_idx in kwd_ref_idxs:
                    b_wer.n_ref += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.n_ref += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if hyp_idx in kwd_hyp_idxs:
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1
    return wer, u_wer, b_wer


def format_ref_with_keywords(text, keywords=None):
    """Extract keywords from the text based on biasing words."""
    if keywords is None:  # tagged words if found
        keywords = re.findall(r"\*.*?\*", text)
    return {
        "keywords": keywords,
        "text": text,
    }


def measure_errors(hyp, ref, keywords=None, **kwargs):
    """Measure WER, U-WER, and B-WER between hypothesis and reference."""
    refs = {0: format_ref_with_keywords(ref, keywords)}
    hyps = {0: hyp}
    text_norm = kwargs.get("text_norm", "simple")  # recipe/phimm/utils/tn.py
    unit = kwargs.get("unit", "word")
    wer, u_wer, b_wer = calc_errors(refs, hyps, tn=text_norm, unit=unit)
    return wer, u_wer, b_wer

def sum_wers(wers):
    total_wer = WordError()
    for wer in wers:
        total_wer += wer
    return total_wer

def compute_wers(refs, hyps, **kwargs):
    """Compute WERs for a batch of references and hypotheses."""
    wers  = []
    for ref, hyp in zip(refs, hyps):
        wer, _, _ = measure_errors(hyp, ref, **kwargs)
        wers.append(wer)
    return wers     

def get_score(wer, choice="wer"):
    choice = choice.lower()
    if choice in ["word_error_rate", "wer"]:
        return 0 - wer.wer
    elif choice in ["accuracy", "acc"]:
        return wer.acc
    elif choice in ["word_error_count", "err"]:
        return 0 - wer.n_err
    else:
        raise ValueError(f"Unsupported score choice: {choice}")


def is_valid(wer, **kwargs):
    """Check if the WER and error count are within the specified limits."""
    max_wer = kwargs.get("max_wer", 100)
    max_err = kwargs.get("max_err", 10000)
    return wer.wer <= max_wer and wer.n_err <= max_err


def compute_score(solution_str, ground_truth, **kwargs):
    """The scoring function for ASR with keywords."""
    extra_info = kwargs.pop("extra_info", {})
    keywords = extra_info.get("keywords", None)
    hyp_text = parse_asr_response(solution_str).get("text") or ""

    wer, u_wer, b_wer = measure_errors(
        hyp_text,
        ground_truth,
        keywords=keywords,
        **kwargs,
    )
    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 0.0)
    choice = kwargs.get("choice", "err")
    score = get_score(wer, choice) * alpha + get_score(b_wer, choice) * beta

    if not is_valid(wer, **kwargs):
        score = -100.0  # penalty for too high WER

    result = {
        "score": score,
        "n_err": wer.n_err,
        "nu_err": u_wer.n_err,
        "nb_err": b_wer.n_err,
        "wer": wer.wer,
        "u_wer": u_wer.wer,
        "b_wer": b_wer.wer,
        "n_ref": wer.n_ref,
        "nu_ref": u_wer.n_ref,
        "nb_ref": b_wer.n_ref,
    }
    return result


def eval_score(solution_str, ground_truth, **kwargs):
    """The scoring function for ASR with keywords."""
    extra_info = kwargs.pop("extra_info", {})
    text_norm = kwargs.pop("text_norm", "english")
    hyp_text = parse_asr_response(solution_str).get("text") or ""
    wer, u_wer, b_wer = measure_errors(
        hyp_text,
        ground_truth,
        keywords=extra_info.get("keywords", None),
        text_norm=text_norm,
        unit="word",
    )
    return {
        "score": wer.n_err,
        "n_err": wer.n_err,
        "nu_err": u_wer.n_err,
        "nb_err": b_wer.n_err,
        "n_ref": wer.n_ref,
        "nu_ref": u_wer.n_ref,
        "nb_ref": b_wer.n_ref,
    }


if __name__ == "__main__":
    pairs = [
        {
            "ref": "It's a barn-burner, the best and worst life insurance .",
            "hyp": "it's a barn burner the best and worst *life insurance*",
        },
        {
            "hyp": "or of the habits of or *people* it is quite impossible.",
            "ref": "or of the habits of *our* people it is quite impossible",
        },
        {
            "hyp": "Who was it she was in love with? The story will t",
            "ref": "who was it she was in love with the story will te upon myself to reply oh i can't wait fre's* the pity then",
        },
        {
            "hyp": "The air and the earth are curiously *mated* and *intermingled* as if the one were the breath of the other,",
            "ref": "the air and the earth are curiously *mated* and *intermingled* as if the one were the breath of the other",
        },
        {
            "hyp": "These thoughts agitated me all day, and my imagination scarcely *calmed* down after several hours' sleep.",
            "ref": "these thoughts agitated me all day and my imagination scarcely *calmed* down after several hours sleep",
        },
        {
            "hyp": "The task will not be difficult, returned David, hesitating, though I greatly fear your presence would rather increase than,*mitigate* his unhappy fortunes.",
            "ref": "the task will not be difficult returned david *hesitating* though i greatly fear your presence would rather increase than *mitigate* ,his unhappy fortunes",
        },
        {
            "hyp": "it was silent and gloomy, *beeing* *tenanted* *solely* by the *captive* and lighted by the dying *embers* of a fire which had been,used for the purpose of *cookery*",
            "ref": "it was silent and gloomy being *tenanted* *solely* by the *captive* and lighted by the dying *embers* of a fire which had been used ,for the *purposed* of *cookery*",
        },
        {
            "hyp": "or of the habits of our people it is quite impossible.",
            "ref": "or of the habits of our people it is quite impossible",
        },
        {
            "hyp": "To be or not to be, that is the question Whether 'tis *nobler* in the mind to suffer the *slings* and arrows what? No, *hamlet*,speaking",
            "ref": "to be or not to be that is the question whether tis *nobler* in the mind to suffer the *slings* and arrows what no *hamlet* ,speaking",
        },
        {
            "hyp": "By quick *marches* through these *inaccessible* mountains, that general *freed* himself from the superior forces of the,*covenanters*",
            "ref": "by quick *marches* through these *inaccessible* mountains that general *freed* himself from the superior forces of the ,*covenanters*",
        },
        {
            "hyp": "This *nobleman's* character, though celebrated for political courage and conduct, was very low for military *prowess* and after some,*skirmishes* in which he was *worsted* he here allowed *montrose* to escape him.",
            "ref": "this *nobleman's* character though celebrated for political courage and conduct was very low for military *prowess* and after some ,*skirmishes* in which he was *worsted* he here allowed *montrose* to escape him",
        },
    ]

    for pair in pairs:
        # result = eval_score(pair["hyp"], pair["ref"])

        result = compute_score(pair["hyp"], pair["ref"], text_norm="english", choice="wer", beta=1.0)
        print(result)

# %%

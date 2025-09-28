# %%
import re
from collections import deque, Counter
from enum import Enum
from functools import partial
from itertools import zip_longest
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
import jiwer.transforms as tr


class RemovePunctuationExclude(tr.RemovePunctuation):
    """RemovePunctuation excluding certain characters."""

    def __init__(self, exclude=None):
        super().__init__()
        self.exclude = exclude or []
        self.tokens_to_remove = [x for x in self.tokens_to_remove if x not in self.exclude]
        # print(f"tokens_to_remove: {self.tokens_to_remove}")


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
        self.ref_words = 0

    def get_wer(self):
        if self.ref_words == 0:
            return 0.0
        # assert self.ref_words != 0
        # errors = self.errors[Code.substitution] + self.errors[Code.insertion] + self.errors[Code.deletion]
        return 100.0 * self.error_count / self.ref_words

    @property
    def accuracy(self):
        return 100.0 - self.get_wer()

    @property
    def error_count(self):
        return self.errors[Code.substitution] + self.errors[Code.insertion] + self.errors[Code.deletion]

    @property
    def error_rate(self):
        return self.get_wer() / 100.0

    def get_result_string(self):
        return f"error_rate={self.get_wer()}, ref_words={self.ref_words}, subs={self.errors[Code.substitution]}, ins={self.errors[Code.insertion]}, dels={self.errors[Code.deletion]}"


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
                best_score = self.scores_[i - 1][j - 1] + (self.cost(ref, hyp, Code.match) if ref == hyp else self.cost(ref, hyp, Code.substitution))

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


def identity(text):
    """Identity normalization function."""
    return text


def lower(text):
    """Lowercase normalization function."""
    return text.lower()


def simple_with_tag(text):
    """Simple normalization function."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            # tr.ExpandCommonEnglishContractions(),
            # tr.RemovePunctuation(),
            RemovePunctuationExclude(exclude=["*"]),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
            # tr.ReduceToListOfListOfWords(),
        ]
    )
    return norm(text)


def simple(text):
    """Simple normalization function."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.RemovePunctuation(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
        ]
    )
    return norm(text)


TN_DICT = {
    "english": EnglishTextNormalizer(),
    "identity": identity,
    "lower": lower,
    "basic": BasicTextNormalizer(),
    "simple": simple,
    "simple_with_tag": simple_with_tag,
}


def text_norm(txt, name=None):
    """Normalize tokens by removing leading and trailing whitespace."""
    name = name or "english"  # Default to EnglishTextNormalizer
    norm = TN_DICT[name]
    if isinstance(txt, str):
        return norm(txt.strip())
    elif isinstance(txt, list):
        return [norm(x) for x in txt]
    else:
        raise ValueError(f"Unsupported type for text normalization: {type(txt)}. Expected str or list of str.")


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


def calc_wers(refs, hyps, tn=None):
    return calc_errors(refs, hyps, tn=tn, unit="word")


def calc_cers(refs, hyps, tn=None):
    return calc_errors(refs, hyps, tn=tn, unit="char")


def calc_errors(refs, hyps, tn=None, unit=None):
    """Calculate WER, U-WER, and B-WER."""
    wer = WordError()
    u_wer = WordError()
    b_wer = WordError()
    for uttid, ref in refs.items():
        if uttid not in hyps:
            continue
        norm = partial(text_norm, name=tn)
        # Normalize reference and hypothesis text
        tn_ref = norm(ref["text"])
        tn_hyp = norm(hyps[uttid])
        bias_words = norm(ref["biasing_words"])

        if unit == "char":
            bias_ref_indexs = find_char_indices(tn_ref, bias_words)
            bias_hyp_indexs = find_char_indices(tn_hyp, bias_words)

            ref_words = list(tn_ref)
            hyp_words = list(tn_hyp)
        else:  # default to words
            bias_ref_indexs = find_word_indices(tn_ref, bias_words)
            bias_hyp_indexs = find_word_indices(tn_hyp, bias_words)

            ref_words = tn_ref.split()
            hyp_words = tn_hyp.split()

        if len(ref_words) == 0 and len(hyp_words) == 0:
            continue

        ed = EditDistance()
        result = ed.align(ref_words, hyp_words)
        for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
            if code == Code.match:
                wer.ref_words += 1
                if ref_idx in bias_ref_indexs:
                    b_wer.ref_words += 1
                else:
                    u_wer.ref_words += 1
            elif code == Code.substitution:
                wer.ref_words += 1
                wer.errors[Code.substitution] += 1
                if ref_idx in bias_ref_indexs:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                if ref_idx in bias_ref_indexs:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if hyp_idx in bias_hyp_indexs:
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1
    return wer, u_wer, b_wer


def format_ref_with_keywords(text, keywords=None):
    """Extract keywords from the text based on biasing words."""
    if keywords is None:  # tagged words if found
        keywords = re.findall(r"\*.*?\*", text)
    return {
        "biasing_words": keywords,
        "text": text,
    }


def compute_biasing_metrics(results):
    """compute biasing metrics"""
    wer, u_wer, b_wer = compute_wers(results)
    return {
        "WER": wer.get_wer(),
        "UWER": u_wer.get_wer(),
        "BWER": b_wer.get_wer(),
    }


def remove_pfx_tag(text, tags=None):
    """Remove leading <|hit|> or <|miss|> and trailing <|/hit|> or <|/miss|> tags from the text."""
    tag = None
    if m := re.search(r"^<\|[^<\|>]+\|>", text.strip()):
        tag = m.group(0)[2:-2]
    if tag is None or (tags is not None and tag not in tags):
        return text.strip()

    s_tag = f"<|{tag}|>"
    e_tag = f"<|/{tag}|>"
    if not text.startswith(s_tag):
        return text
    text = text.removeprefix(s_tag)
    if (idx := text.find(e_tag)) >= 0:
        text = text[idx + len(e_tag) :]
    return text.strip()


def remove_pfx_tags(text, tags=None):
    while len(text):
        n_len = len(text)
        text = remove_pfx_tag(text, tags)
        if len(text) == n_len:
            break
    return text


def test_remove_pfx_tags():
    test_cases = [
        ("<|hit|> this is a test <|/hit|> Hello world", "Hello world"),
        ("<|miss|> <|/miss|> Goodbye", "Goodbye"),
        ("<|hit|>Test", "Test"),
        ("<|miss|> Something", "Something"),
        ("<|hit|><|miss|> Something", "Something"),
        ("No marker here", "No marker here"),
    ]
    num_errors = 0
    for inp, expected in test_cases:
        output = remove_pfx_tags(inp)
        if output != expected:
            print(f"Input: {inp!r} | Output: {output!r} | Expected: {expected!r}")
            num_errors += 1
    return num_errors > 0


def format_hyp(text, remove_pfx=False):
    if remove_pfx:
        text = remove_pfx_tags(text)
    return text


def compute_wers(results, tn=None, unit=None, remove_hyp_pfx=False):
    """compute WER, U-WER, and B-WER"""
    # Extract reference and hypothesis pairs from groups
    refs = {result.get("id", i): format_ref_with_keywords(result["ref"], result.get("keywords", None)) for i, result in enumerate(results)}
    hyps = {result.get("id", i): format_hyp(result["hyp"], remove_hyp_pfx) for i, result in enumerate(results)}
    # Calculate WER, U-WER, and B-WER
    wer, u_wer, b_wer = calc_errors(refs, hyps, tn=tn, unit=unit)
    return wer, u_wer, b_wer


def eval_biasing_metrics(groups):
    """compute eval metrics"""
    # Extract reference and hypothesis from top group (i.e., the first group)
    results = [
        {
            "keywords": group[0]["keywords"],
            "ref": group[0]["text"],
            "hyp": group[0]["completions"],
            "id": group[0].get("id", i),  # if "id" is not present, use index
        }
        for i, group in enumerate(groups)
    ]
    # not check hyp pfx for evaluation
    wer, u_wer, b_wer = compute_wers(results, remove_hyp_pfx=True)
    return {
        "WER": wer.get_wer(),
        "UWER": u_wer.get_wer(),
        "BWER": b_wer.get_wer(),
    }


def compute_reward_wers(completions, tn=None, unit=None, **kwargs):
    """Compute rewards for a list of completions."""
    references = kwargs["text"]
    keyword_lists = kwargs.get("keywords", [])
    rewards = []
    for i, (hyp, ref, keywords) in enumerate(zip_longest(completions, references, keyword_lists, fillvalue=None)):
        rewards.append(compute_wers([{"id": i, "ref": ref, "keywords": keywords, "hyp": hyp}], tn=tn, unit=unit))
    return rewards


def reward_full_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [wer[0].accuracy for wer in wers]  # WER


def reward_unbias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [wer[1].accuracy for wer in wers]  # U-WER


def reward_bias_accuracy(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [wer[2].accuracy for wer in wers]  # B-WER


def reward_full_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[0].error_count for wer in wers]  # WER


def reward_unbias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[1].error_count for wer in wers]  # U-WER


def reward_bias_error(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[2].error_count for wer in wers]  # B-WER


def reward_full_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[0].error_rate for wer in wers]  # WER


def reward_unbias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[1].error_rate for wer in wers]  # U-WER


def reward_bias_error_rate(completions, **kwargs):
    """Compute the reward for a list of completions."""
    wers = compute_reward_wers(completions, **kwargs)
    return [-wer[2].error_rate for wer in wers]  # B-WER


def reward_major_vote(completions, **kwargs):
    """Compute the reward for a list of completions."""
    num_generations = kwargs.get("num_generations", 1)
    tn = kwargs.get("tn", None)
    hyps = [text_norm(hyp, tn) for hyp in completions]
    groups = [hyps[i : i + num_generations] for i in range(0, len(hyps), num_generations)]
    rewards = []
    for group in groups:
        counter = Counter(group)
        candidate, votes = counter.most_common(1)[0]
        vote_rate = votes / len(group)
        rewards += [int(hyp == candidate) for hyp in group]
    return rewards


# %%


if __name__ == "__main__":
    pairs = [
        {
            "hyp": "Who was it *she was* in love with? The story will tell, I took upon myself to reply. Oh, I can't wait for the story. The story won't tell, said *douglas* Not in any literal, vulgar way. Nor is the pity then.",
            "ref": "who was it *she was* in love with the story will tell i took upon myself to reply oh i can't wait for the story the story won't tell said *douglas* not in any *literal* vulgar way *more's* the pity then",
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
        wer, u_wer, b_wer = compute_wers([pair], tn="simple")
        print(f"WER: {wer.get_result_string()}")
        print(f"U-WER: {u_wer.get_result_string()}")
        print(f"B-WER: {b_wer.get_result_string()}")

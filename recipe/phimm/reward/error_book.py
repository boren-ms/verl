# %%
from difflib import SequenceMatcher
from whisper_normalizer.english import EnglishTextNormalizer
from collections import Counter


def align(refs, hyps):
    matcher = SequenceMatcher(None, refs, hyps)
    hits = []
    edits = []
    for op, i1, i2, _, _ in matcher.get_opcodes():
        if op == "equal":
            hits += refs[i1:i2]
        else:
            edits += refs[i1:i2]
    return hits, edits


class ErrorBook:
    def __init__(self, tn=None):
        self.cnt = Counter()
        self.tn = tn or EnglishTextNormalizer()
        self.hit_cnt = Counter()
        self.edit_cnt = Counter()

    def error_words(self, ref, hyp=None):
        hyp = hyp or ref
        refs = self.tn(ref).split()
        hyps = self.tn(hyp).split()
        hits, edits = align(refs, hyps)
        self.hit_cnt += Counter(hits)
        self.edit_cnt += Counter(edits)
        # return Counter({w: self.cnt[w] for w in refs if w in self.cnt})
        return list(set(w for w in refs if w in self.cnt))

    def update(self):
        self.cnt += self.edit_cnt
        self.cnt -= Counter(self.hit_cnt)
        self.hit_cnt = Counter()
        self.edit_cnt = Counter()


# Add singleton storage and accessor
_EB = None


def get_eb():
    """
    Return a singleton ErrorBook instance.

    Creates the ErrorBook on first call and reuses it thereafter.
    """
    global _EB
    if _EB is None:
        _EB = ErrorBook()
    return _EB


# %%

if __name__ == "__main__":
    pairs = [
        {
            "hyp": "Who was it she was in love with? The story will t",
            "ref": "who was it she was in love with the story will te upon myself to reply oh i can't wait for the story the story won't tell said *douglas* not in any *literal* vulgar way *more's* the pity then",
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

    eb = ErrorBook()
    for i, pair in enumerate(pairs):
        print(f"==== Example {i} ====")
        print("Hyp: ", pair["hyp"])
        print("Ref: ", pair["ref"])
        kws = eb.error_words(pair["ref"], pair["hyp"])
        print("Keywords: ", kws)
        print("Error book: ", eb.cnt)
        eb.update()
        print()

# %%

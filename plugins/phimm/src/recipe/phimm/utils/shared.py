from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence


def is_list(obj):
    return not isinstance(obj, str | bytes) and isinstance(obj, Sequence)


def get_value(d, key, default=None):
    if isinstance(d, Mapping) and key in d:
        return d[key]
    keys = key.split(".")
    for k in keys:
        if isinstance(d, Mapping) and k in d:
            d = d[k]
        elif isinstance(d, list | tuple) and k.isdigit() and int(k) < len(d):
            d = d[int(k)]
        else:
            return default
    return d


BRACKET_PATTERN = re.compile(
    r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|"
    r"（[^）]*）|［[^］]*］|｛[^｝]*｝|＜[^＞]*＞|"
    r"【[^】]*】|〔[^〕]*〕|〈[^〉]*〉|《[^》]*》|「[^」]*」|『[^』]*』|〖[^〗]*〗|｟[^｠]*｠"
)
BRACKET_EXCLUDE = re.compile(r"^<nonspeech>$", re.IGNORECASE)


def has_brackets(text):
    text = "" if text is None else str(text)
    matches = BRACKET_PATTERN.findall(text)
    return any(not BRACKET_EXCLUDE.match(m) for m in matches)


def rank_print(*args, main=True, **kwargs):
    if main and not dist_state().is_main_process:
        return
    rank = dist_state().process_index
    print(f"[{rank}]", *args, **kwargs)


def dist_state():
    from accelerate import PartialState

    return PartialState()


def all_rank_print(*args, **kwargs):
    rank_print(*args, main=False, **kwargs)


def has_repeat(text, min_reps=4, max_ngram=5):
    """Return True if the text contains any n-gram (size 1..max_ngram) repeated >= min_reps times consecutively.

    Scans the full text, not just the tail.

    Examples (min_reps=4):
        "the the the marketing and and and and and and and and"  -> True (and x8)
        "as we have access to of of of of of of of of of of"     -> True (of x10)
        "we want to grow is it is it is it is it is it is it"    -> True ("is it" x6)
        "foo foo foo foo then more normal words after that here" -> True (foo x4)
        "this is a normal sentence."                              -> False
    """
    words = text.split() if isinstance(text, str) else list(text or [])
    n = len(words)
    if n < min_reps:
        return False
    for ng in range(1, max_ngram + 1):
        if n < ng * min_reps:
            continue
        for start in range(0, n - ng * min_reps + 1):
            ngram = tuple(words[start : start + ng])
            reps = 1
            pos = start + ng
            while pos + ng <= n and tuple(words[pos : pos + ng]) == ngram:
                reps += 1
                pos += ng
            if reps >= min_reps:
                return True
    return False


def has_repeat_error(hyp, ref=None, min_reps=4, max_ngram=5, tn_name=None, lang="english"):
    """Return True if the error portion of ``hyp`` (vs ``ref``) contains a repeated n-gram.

    Aligns ``ref`` vs ``hyp`` at the word level and runs ``has_repeat`` only on
    the hyp segments inside non-equal opcodes (insertions/substitutions). When
    ``ref`` is empty or None, falls back to scanning the full hyp. If ``tn_name``
    is given, both inputs are passed through ``text_norm(..., tn_name)`` first;
    if ``tn_name`` is None but ``lang`` is given, ``default_tn_name(lang)`` is
    used. Otherwise the caller is expected to pre-normalize.
    """
    if not hyp:
        return False
    from recipe.phimm.utils.tn import default_tn_name, text_norm

    tn_name = tn_name or default_tn_name(lang)
    hyp = text_norm(hyp, tn_name)
    ref = text_norm(ref or "", tn_name)
    if not ref:
        return has_repeat(hyp, min_reps=min_reps, max_ngram=max_ngram)
    from difflib import SequenceMatcher

    ref_words = ref.split()
    hyp_words = hyp.split()
    sm = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        if has_repeat(hyp_words[j1:j2], min_reps=min_reps, max_ngram=max_ngram):
            return True
    return False


def has_tail_hallucination(hyp, ref=None, min_words=3, tn_name=None, lang="english"):
    """Return True if ``hyp`` ends with hallucinated insertions vs ``ref``.

    Aligns ``ref`` vs ``hyp`` at the word level. Returns True when the final
    alignment opcode is ``insert`` (or ``replace`` whose hyp side extends to
    the end of hyp) and the number of trailing hyp-only words is
    ``>= min_words``. Returns False when either side is empty. If ``tn_name``
    is given, both inputs are passed through ``text_norm(..., tn_name)`` first;
    if ``tn_name`` is None but ``lang`` is given, ``default_tn_name(lang)`` is
    used.
    """
    if not hyp or not ref:
        return False
    from recipe.phimm.utils.tn import default_tn_name, text_norm

    tn_name = tn_name or default_tn_name(lang)
    hyp = text_norm(hyp, tn_name)
    ref = text_norm(ref, tn_name)
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not hyp_words or not ref_words:
        return False
    from difflib import SequenceMatcher

    sm = SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    opcodes = sm.get_opcodes()
    if not opcodes:
        return False
    tag, i1, i2, j1, j2 = opcodes[-1]
    if tag not in ("insert", "replace"):
        return False
    if j2 != len(hyp_words):
        return False
    n_extra = (j2 - j1) - (i2 - i1)
    return n_extra >= min_words


def to_keyword_list(raw_keywords):
    """Normalize a keyword field into a list of strings.

    Supports list/tuple/set, JSON/python-literal serialized strings, and scalars.
    """
    if raw_keywords is None:
        return []
    if is_list(raw_keywords) or isinstance(raw_keywords, tuple | set):
        return [str(x) for x in raw_keywords]
    if isinstance(raw_keywords, str):
        s = raw_keywords.strip()
        if not s:
            return []
        for parse_fn in (json.loads, ast.literal_eval):
            try:
                parsed = parse_fn(s)
            except Exception:
                continue
            if is_list(parsed) or isinstance(parsed, tuple | set):
                return [str(x) for x in parsed]
        return [s]
    return [str(raw_keywords)]


def has_missing_keyword(keywords, response, norm_name=None, lang="english"):
    """Return True when any normalized keyword is absent in normalized response.

    If ``norm_name`` is not provided, falls back to ``default_tn_name(lang)``.
    """
    from recipe.phimm.utils.tn import default_tn_name, text_norm

    norm_name = norm_name or default_tn_name(lang)

    keywords = to_keyword_list(keywords)
    if not keywords:
        return False

    norm_response = text_norm(str(response or ""), norm_name)
    for kw in keywords:
        norm_kw = text_norm(str(kw), norm_name).strip()
        if not norm_kw:
            continue
        if re.search(re.escape(norm_kw), norm_response) is None:
            return True
    return False

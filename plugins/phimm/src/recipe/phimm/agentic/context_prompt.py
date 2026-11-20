"""Build ASR context prompts from the current trajectory's generated text."""

import copy
import re
import unicodedata
from collections import deque
from itertools import chain

from recipe.phimm.reward.asr_response import get_hyp_text

DEFAULT_CONTEXT_MODE = "text"
DEFAULT_MAX_BIAS_TERMS = -1
CONTEXT_MODES = {DEFAULT_CONTEXT_MODE, "bias"}


def context_options(asr_config):
    """Read nested options or a legacy boolean context flag."""
    options = asr_config.get("context_aware", False)
    if isinstance(options, bool):
        return {"enable": options}
    return options


# English function words and frequent conversational words. Recipes
# can extend this list for their language or domain without changing code.
COMMON_WORDS = frozenset(
    """
a about actually after again all also am an and any are as at back be because been before being
between both but by can could did do does doing done down each even every first for from get gets
getting give go goes going good got had has have having he her here him his how i if in into is it
its just know last let like look lot made make many may me mean might more most much must my need
never new next no not now of off okay on once one only or other our out over really right said same
say see she should so some still such take tell than that the their them then there these they thing
things think this those through time to too two up us use very want was way we well went were
what when where which who why will with would yeah yes yet you your
ain't aren't can't couldn't didn't doesn't don't hadn't hasn't haven't he's here's i'd i'll i'm i've
isn't it's let's that's there's they're they've wasn't we're we've weren't what's won't wouldn't
you'd you'll you're you've thank thanks gonna wanna today tomorrow yesterday
able across almost along already always another anything anywhere around aside away became become
becomes becoming began begin begins behind below beside beyond bit bring brings brought came comes
coming consider considering continue continued continues currently definitely despite different
differently during either else enough especially etc ever everybody everyone everything exactly
example except expect expected expects fairly far feel feels felt few finally find finds fine
following found further generally given gives glad gotten great happens happened hardly help helps
however indeed instead keep keeping keeps kind kinda knew known knows later least left less
likely literally little looking looks mainly maybe means meant meanwhile merely mine moreover
mostly move moved moves moving myself namely nearly neither nevertheless nobody none normally
nothing obviously often ourselves outside overall perhaps please plenty possible probably quite
rather ready real reason recently regardless remember remembered roughly says saying saw seems
seemed seem seen seeing several short simply since slight slightly sorta something sometimes
somewhere soon sorry speak speaking spoken started starting starts stay stayed stays stuff suppose
supposed sure taken takes taking talked talking talks tends terms themselves therefore though
thought thoughts thus together told toward try tried trying twice under unless until upon usually
various versus via wants wanted whatever whenever wherever whether while whoever whole whose
within without wonder wondered working works worked yourself yourselves itself
""".split()
)
WORD = re.compile(r"[^\W_]+(?:(?:['’\-‐‑]|(?<=\d)[.,](?=\d))[^\W_]+)*")
VERSIONED_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-‐‑][0-9][A-Za-z0-9]*(?:\.[0-9][A-Za-z0-9]*)*)+")


def _without_punctuation(term, *, keep_hyphens=False):
    return " ".join(
        "".join(
            character
            for character in word
            if not unicodedata.category(character).startswith("P") or (keep_hyphens and character in "-‐‑")
        )
        for word in term.split()
    )


def _key(word):
    return _without_punctuation(unicodedata.normalize("NFKC", word)).casefold()


COMMON_WORD_KEYS = frozenset(_key(word) for word in COMMON_WORDS)
FILLER_WORD_KEYS = frozenset(
    """
ah ahh ahhh eh ehh er erm ermm hm hmm hmmm huh
mhmm mhm mm mmm mmmm mmhm mmhmm
uh uhh uhhh uhhuh uhuh uhm um umm ummm umhum
""".split()
)


def _bias_terms(transcript, excluded_words, filter_numbers):
    """Keep content words and short adjacent capitalized phrases."""
    words = list(WORD.finditer(transcript))
    i = 0
    while i < len(words):
        first = words[i]
        i += 1
        key = _key(first[0])
        if key in excluded_words or (filter_numbers and (key.isnumeric() or first[0].isnumeric())):
            continue
        last = first
        if first[0][0].isupper():
            end = min(i + 2, len(words))
            while (
                i < end
                and words[i][0][0].isupper()
                and _key(words[i][0]) not in excluded_words
                and transcript[last.end() : words[i].start()].isspace()
            ):
                last = words[i]
                i += 1
        term = transcript[first.start() : last.end()]
        # Preserve internal hyphens and version dots in the displayed term.
        # _key still removes punctuation for comparison and deduplication.
        yield " ".join(
            word if VERSIONED_NAME.fullmatch(word) else _without_punctuation(word, keep_hyphens=True)
            for word in term.split()
        )


class ContextPrompt:
    """One builder per trajectory; max_history counts previous response turns, including empty ones."""

    def __init__(
        self,
        mode,
        *,
        version=None,
        max_history=1,
        max_terms=DEFAULT_MAX_BIAS_TERMS,
        filter_common=True,
        filter_numbers=True,
        filter_fillers=True,
        extra_common_words=(),
    ):
        if mode not in CONTEXT_MODES:
            raise ValueError(f"Unknown ASR context mode {mode!r}; expected one of {sorted(CONTEXT_MODES)}")
        if not isinstance(max_history, int) or isinstance(max_history, bool) or max_history < 1:
            raise ValueError("asr.context_aware.max_history must be a positive integer")
        if not isinstance(max_terms, int) or isinstance(max_terms, bool) or (max_terms != -1 and max_terms < 1):
            raise ValueError("asr.context_aware.max_terms must be -1 or a positive integer")
        self.mode = mode
        self.version = version
        self.max_history = max_history
        self.responses = deque(maxlen=max_history)
        self.max_terms = max_terms
        self.filter_numbers = filter_numbers
        self.excluded_words = (
            COMMON_WORD_KEYS.union(_key(word) for word in extra_common_words) if filter_common else frozenset()
        )
        if filter_fillers:
            self.excluded_words |= FILLER_WORD_KEYS
        self.history = ""
        self.terms = ()

    @property
    def context(self):
        if self.mode == DEFAULT_CONTEXT_MODE:
            return self.history
        return ", ".join(f"*{term}*" for term in self.terms)

    def observe(self, response):
        transcript = get_hyp_text(response, version=self.version)
        if self.mode == DEFAULT_CONTEXT_MODE:
            self.responses.append(transcript)
            self.history = "\n".join(text for text in self.responses if text)
            # Cap the combined window by whitespace-delimited words while
            # preserving punctuation and spacing within the retained suffix.
            if self.max_terms != -1:
                words = list(re.finditer(r"\S+", self.history))
                if len(words) > self.max_terms:
                    self.history = self.history[words[-self.max_terms].start() :]
            return

        # Rank by last occurrence: later terms in this segment precede earlier
        # ones, and all precede terms carried over from older segments.
        terms = tuple(reversed(list(_bias_terms(transcript, self.excluded_words, self.filter_numbers))))
        # Empty responses still advance the window. Retain each response's
        # capped distinct terms so expired responses cannot leak context.
        distinct = {}
        for term in terms:
            distinct.setdefault(_key(term), term)
            if len(distinct) == self.max_terms:
                break
        self.responses.append(tuple(distinct.values()))
        terms = chain.from_iterable(reversed(self.responses))
        selected = {}
        for term in terms:
            selected.setdefault(_key(term), term)
            if len(selected) == self.max_terms:
                break
        self.terms = tuple(selected.values())


def add_context_to_messages(messages, context, mode=DEFAULT_CONTEXT_MODE):
    """Append context inside the final user turn, before applying the chat template."""
    if not context:
        return messages
    if mode == DEFAULT_CONTEXT_MODE:
        line = f"Previous context: {context}"
    else:
        line = f"Pay extra attention to the following phrases/words: {context}."
    messages = copy.deepcopy(messages)
    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] += f"\n{line}"
            return messages
    raise ValueError("ASR previous context requires a user prompt")

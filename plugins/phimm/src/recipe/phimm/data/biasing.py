# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline positive/negative word sampling from a fixed, deduplicated vocabulary."""

import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache


def is_non_space_script_char(char):
    return any(
        start <= ord(char) <= end
        for start, end in ((0x3400, 0x9FFF), (0x0E00, 0x0E7F), (0x3040, 0x30FF), (0xAC00, 0xD7AF))
    )


def is_new_non_space_script_char(char):
    return any(
        start <= ord(char) <= end
        for start, end in ((0x0E80, 0x0FFF), (0x1000, 0x109F), (0x1780, 0x17FF), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F))
    )


@lru_cache(maxsize=500000)
def normalize_phrase(text):
    """Apply MoE-style punctuation removal and lowercase for hit matching."""
    return " ".join(re.findall(r"[\w'\-]+", text)).lower()


def _phrase_occurs(reference, normalized, reference_words):
    if not normalized:
        return False
    # Normalized ASCII words have exactly the same boundaries as split tokens.
    # Avoid rescanning the transcript for each of up to 1,000 RLBR candidates.
    if normalized.isascii() and " " not in normalized:
        return normalized in reference_words
    left_free = is_non_space_script_char(normalized[0]) or is_new_non_space_script_char(normalized[0])
    right_free = is_non_space_script_char(normalized[-1]) or is_new_non_space_script_char(normalized[-1])
    start = reference.find(normalized)
    while start >= 0:
        end = start + len(normalized)
        left_ok = left_free or start == 0 or not (reference[start - 1].isalnum() or reference[start - 1] in "_'-")
        right_ok = right_free or end == len(reference) or not (reference[end].isalnum() or reference[end] in "_'-")
        if left_ok and right_ok:
            return True
        start = reference.find(normalized, start + 1)
    return False


def matching_keywords(text, candidates):
    """Find actual candidate hits, including differently grouped phrases."""
    reference = normalize_phrase(text)
    reference_words = set(reference.split())
    return [phrase for phrase in candidates if _phrase_occurs(reference, normalize_phrase(phrase), reference_words)]


def unique_phrases(phrases):
    """Deduplicate normalized spelling while preserving the first original surface form."""
    unique = {}
    for phrase in phrases:
        normalized = normalize_phrase(phrase)
        if normalized:
            unique.setdefault(normalized, phrase)
    return list(unique.values())


class NoEligibleKeywords(ValueError):
    """The reference has no usable keyword after common-word filtering."""


@dataclass(frozen=True)
class BiasingConfig:
    max_num: int = 1000
    hit_ratio: tuple[float, float] = (0.1, 1.0)
    negative_only: bool = False

    def __post_init__(self):
        if not isinstance(self.max_num, int) or not 1 <= self.max_num <= 1000:
            raise ValueError("max_num must be an integer between 1 and 1000")
        if len(self.hit_ratio) != 2 or not 0 < self.hit_ratio[0] <= self.hit_ratio[1] <= 1:
            raise ValueError("hit_ratio must contain ordered bounds within (0, 1]")


class BiasingSimulator:
    """Each call independently samples a candidate budget and positive coverage."""

    def __init__(self, config=None, *, seed=42, common_words=()):
        self.config = config or BiasingConfig()
        self.rng = random.Random(seed)
        self.common_words = {normalize_phrase(word) for word in common_words}
        self.pool = []

    def _usable(self, word):
        normalized = normalize_phrase(word)
        return normalized and not all(token in self.common_words for token in normalized.split())

    def prepare_word_pool(self, words):
        self.pool = unique_phrases(word for word in words if self._usable(word))
        if not self.pool:
            raise ValueError("Vocabulary contains no usable candidates")

    def _negatives(self, text, count):
        reference = normalize_phrase(text)
        reference_words = set(reference.split())
        negatives, seen = [], set()
        while len(negatives) < count and len(seen) < len(self.pool):
            remaining = count - len(negatives)
            indices = self.rng.sample(range(len(self.pool)), min(max(2 * remaining, 8), len(self.pool)))
            for index in indices:
                if index in seen:
                    continue
                seen.add(index)
                phrase = self.pool[index]
                if not _phrase_occurs(reference, normalize_phrase(phrase), reference_words):
                    negatives.append(phrase)
                    if len(negatives) == count:
                        break
            if len(self.pool) - len(seen) < count - len(negatives):
                for index, phrase in enumerate(self.pool):
                    if index not in seen and not _phrase_occurs(reference, normalize_phrase(phrase), reference_words):
                        negatives.append(phrase)
                        if len(negatives) == count:
                            break
                break
        return negatives

    def sample(self, text):
        if not isinstance(text, str):
            raise ValueError("Biasing simulation requires a string transcript")
        budget = self.rng.randint(1, self.config.max_num)
        positives = []
        if not self.config.negative_only:
            usable = unique_phrases(word for word in text.split() if self._usable(word))
            if not usable:
                raise NoEligibleKeywords("Reference has no eligible keywords after common-word filtering")
            count = math.ceil(len(usable) * self.rng.uniform(*self.config.hit_ratio))
            if count > self.config.max_num:
                raise ValueError("Selected positives exceed the candidate limit")
            budget = max(budget, count)
            positives = self.rng.sample(usable, count)
        candidates = unique_phrases(positives + self._negatives(text, budget - len(positives)))
        if not candidates:
            raise ValueError("Vocabulary contains no non-matching candidates for this transcript")
        self.rng.shuffle(candidates)
        return candidates, matching_keywords(text, candidates)

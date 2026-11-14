"""Shared text normalization and alignment counts for ASR reward metrics."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_compound_pairs


@dataclass
class ErrorCounts:
    """Count aligned substitutions, deletions, insertions, and matching units."""

    n_sub: int = 0
    n_del: int = 0
    n_ins: int = 0
    n_hit: int = 0

    @property
    def n_ref(self):
        return self.n_sub + self.n_del + self.n_hit

    @property
    def n_err(self):
        return self.n_sub + self.n_del + self.n_ins

    def accuracy(self, **betas):
        """Return one minus the weighted error rate; insertions may make it negative."""
        return 1 - self.wer(**betas)

    def wer(self, **betas):
        """Divide weighted errors by at least one reference unit."""
        return self.edit_distance(**betas) / max(self.n_ref, 1)

    def edit_distance(self, **betas):
        """Weight edit operations independently for deletion/insertion metrics."""
        return (
            betas.get("sub", 1.0) * self.n_sub + betas.get("del", 1.0) * self.n_del + betas.get("ins", 1.0) * self.n_ins
        )


def normalize_text(text, name=None, lang="english"):
    """Apply an explicit normalizer or the existing language-specific default."""
    from recipe.phimm.utils.tn import default_tn_name
    from recipe.phimm.utils.tn import text_norm as apply_text_norm

    name = name or default_tn_name(lang)
    return apply_text_norm(text.strip(), name=name)


def _count_ops(ref_units, hyp_units):
    counts = ErrorCounts()
    alignment = SequenceMatcher(a=ref_units, b=hyp_units, autojunk=False)
    for tag, i1, i2, j1, j2 in alignment.get_opcodes():
        ref_length, hyp_length = i2 - i1, j2 - j1
        if tag == "equal":
            counts.n_hit += ref_length
        elif tag == "delete":
            counts.n_del += ref_length
        elif tag == "insert":
            counts.n_ins += hyp_length
        elif tag == "replace":
            counts.n_sub += min(ref_length, hyp_length)
            counts.n_del += max(ref_length - hyp_length, 0)
            counts.n_ins += max(hyp_length - ref_length, 0)
    return counts


def _split_units(text, unit="word"):
    if unit.lower() in {"word", "words"}:
        return text.split()
    if unit.lower() in {"char", "chars"}:
        return list(re.sub(r"\s+", "", text))
    raise ValueError(f"Unsupported error measure unit: {unit!r}. Expected 'word' or 'char'.")


def measure(hyp, ref, tgt_lang="english", **kwargs):
    """Normalize and align text using the recipe's existing SequenceMatcher rules."""
    norm_name = kwargs.get("text_norm")
    unit = kwargs.get("unit", "word")
    ref_text = normalize_text(ref, name=norm_name, lang=tgt_lang)
    hyp_text = normalize_text(hyp, name=norm_name, lang=tgt_lang)
    if kwargs.get("compound_norm", False):
        [ref_text], [hyp_text] = normalize_compound_pairs([ref_text], [hyp_text])
    return _count_ops(_split_units(ref_text, unit=unit), _split_units(hyp_text, unit=unit))

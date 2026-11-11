# OpenASR bracket normalization

The bracket helper and its complete annotation allowlist are copied from
[Open ASR Leaderboard PR #246](https://github.com/huggingface/open_asr_leaderboard/pull/246),
pinned to commit `f58d58a3f98eb9d5c2dd284ce85954b3cd801759`.
Only a complete allowlisted body is removed after case folding and whitespace
collapse. Ordinary enclosed words continue through the existing punctuation,
number, and spelling rules.

The shared helper is used by both English normalizers, the basic normalizer,
both basic multilingual normalizers, and the supplementary Mandarin normalizer.
It matches paired ASCII/fullwidth parentheses, square brackets, and angle brackets;
removes innermost annotations iteratively; and inserts spaces to avoid joining words.
Empty outer enclosures left by nested removal are handled by punctuation cleanup.

Existing language routing is retained, including Hindi lattice OIWER and the
supplementary normalizer's language-specific dispatch hook. This repository does
not currently vendor the upstream Armenian-specific normalizer: `hy` retains
its existing generic multilingual route.

## Validation

Run:

```bash
python -m pytest -q \
  tests/recipe/phimm/utils/test_open_asr_bracket_normalization.py \
  tests/recipe/phimm/test_openasr_expansion.py \
  tests/recipe/phimm/utils/test_tn.py
```

The regression tests cover every allowlisted tag with all six delimiter pairs,
ordinary bracketed speech, nested and mismatched delimiters, fullwidth brackets,
word boundaries, downstream number/spelling normalization, reference/prediction
symmetry, and the existing multilingual and Hindi behavior.

## Historical rescoring

Rescore saved raw references and predictions for every compared model with the
same normalizer revision. Keep response extraction, grouping, dataset coverage,
and aggregation fixed when measuring the effect of this change. Preserve raw
manifest hashes and old/new error and reference-word counts: normalization can
change the denominator as well as the errors.

The upstream score cache does not include the normalizer revision. For upstream
bucket scoring, force fresh local scores with
`--recompute_scores --no_upload_scores`. A local raw-manifest rescorer should
likewise bypass saved scores. Existing inference outputs can be reused.

LibriSpeech inclusion and ranking averages are unchanged. The separate ranking
policy change bundled in upstream PR #246 is outside this patch.

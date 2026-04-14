---
name: asr-word-error-analysis
description: Analyze a single ASR result_details JSONL file with per-utterance ref vs hyp differences, word-level alignment, and ranked error breakdowns. Use when Codex needs to inspect one specified model on one specified dataset, auto-discover the latest result_details_*.jsonl for that model and dataset under an Azure results root, diagnose transcription quality, find substitution or insertion or deletion patterns, or generate per-utterance review artifacts.
---

# ASR Word Error Analysis

Analyze a single ASR result file in depth: word-level alignment, per-utterance `ref` vs `hyp` differences, confusion pairs, pattern statistics, and playable audio for the worst utterances. Prefer the bundled script for repeatable analysis.

For *comparing two models* on the same dataset, use the **asr-detail-compare** skill instead.

## When to Use

- You have one `result_details*.jsonl` from an ASR eval and want to understand the error profile.
- You need the most common substitution confusions, deleted filler words, or hallucinated insertions.
- You want per-utterance error ranking to find the worst utterances.
- You want a visual HTML report of word-level alignments for the worst utterances.
- You need error-rate bucketed by utterance length to see where the model struggles.

## Workflow

1. Identify the `result_details*.jsonl` file by explicit path or by `--model` + `--dataset` auto-discovery.
2. When using model discovery, rely on the script to choose the most recently modified `result_details_*.jsonl` under `<results-root>/<model>/<dataset>/`.
3. Run [analyze_word_errors.py](./scripts/analyze_word_errors.py).
4. Review artifacts: start with `summary.json`, then `error_details.csv` for utterance-level `ref` vs `hyp`, then `alignment_samples.txt` or `report.html` for visual inspection.
5. When HTML is enabled, expect the script to download the ranked worst utterances' audio into the output directory and link them in the report with playable local controls.

## Script

```bash
python skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
  --input-path path/to/result_details_2025-03-01_12-00-00.jsonl \
  --dataset openasr/ami \
  --output-dir tmp/asr-word-error-analysis \
  --write-html
```

With model auto-discovery:

```bash
python skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
  --model en_hc_fy24_200k_step_7800 \
  --dataset openasr/ami \
  --output-dir tmp/asr-word-error-analysis \
  --write-html
```

See [options reference](./references/options.md) for the full flag table.

## Output Artifacts

| File | Content |
|------|---------|
| `summary.json` | Dataset-level WER, error totals, sub/del/ins rates, top confusion pairs |
| `error_details.csv` | Every utterance ranked by error count with `ref`, `hyp`, per-type counts, WER, and alignment ops |
| `substitutions.csv` | Ranked (ref_word → hyp_word) confusion pairs with counts |
| `deletions.csv` | Ranked words most frequently deleted (in ref but not hyp) |
| `insertions.csv` | Ranked words most frequently inserted (in hyp but not ref) |
| `error_patterns.csv` | Error rate bucketed by utterance ref-word count |
| `alignment_samples.txt` | Human-readable 3-line alignment (REF / HYP / OPS) for top-N worst utterances |
| `report.html` | Standalone visual report with KPI cards, confusion table, per-utterance alignment grids, and playable downloaded audio for the ranked worst utterances |

## Interpreting Results

- **High substitution rate**: model confuses similar-sounding words. Check `substitutions.csv` for systematic patterns (e.g., homophones, number formats, proper nouns).
- **High deletion rate**: model drops words, often fillers (`uh`, `um`) or short function words. Check `deletions.csv`.
- **High insertion rate**: model hallucinates extra words. Check `insertions.csv`.
- **Error rate vs length**: `error_patterns.csv` shows whether short or long utterances are harder. Very short utterances often have inflated WER from a single error.
- **Per-utterance review**: `error_details.csv`, `alignment_samples.txt`, and `report.html` are the main artifacts for checking exact `ref` vs `hyp` differences on individual utterances.

See [alignment format and JSONL schema details](./references/format.md) for alignment notation and input schema.

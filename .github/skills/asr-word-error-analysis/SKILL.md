---
name: asr-word-error-analysis
description: Analyze a single ASR JSONL output with per-utterance ref vs hyp differences, word-level alignment, and ranked error breakdowns. Use when Codex needs to inspect one specified model on one specified dataset, auto-discover local files, verl val_data_gen outputs, or the latest result_details_*.jsonl under an Azure results root, diagnose transcription quality, find substitution or insertion or deletion patterns, or generate per-utterance review artifacts.
---

# ASR Word Error Analysis

Analyze a single ASR result file in depth: word-level alignment, per-utterance `ref` vs `hyp` differences, confusion pairs, pattern statistics, and playable audio for the worst utterances. Prefer the bundled script for repeatable analysis, and default to model-based discovery when the user gives a model name and dataset instead of an explicit file.

For *comparing two models* on the same dataset, use the **asr-detail-compare** skill instead.

## When to Use

- You have one ASR JSONL output, either `result_details*.jsonl` or verl `val_data_gen`, and want to understand the error profile.
- You need the most common substitution confusions, deleted filler words, or hallucinated insertions.
- You want per-utterance error ranking to find the worst utterances.
- You want a visual HTML report of word-level alignments for the worst utterances.
- You need error-rate bucketed by utterance length to see where the model struggles.

## Workflow

1. Identify the input JSONL by explicit `--input-path` or by `--model` + `--dataset` auto-discovery.
2. When using model discovery, rely on the script's discovery order: local paths, then verl `val_data_gen` outputs, then `result_details_*.jsonl` under the results root.
3. For verl validation outputs, set `--model` to `<project_name>/<experiment_name>` matching the config's `trainer.project_name` / `trainer.experiment_name`. The script picks the latest numeric step from `<val-data-root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl` and auto-remaps `gts` to `ref` and `clean_output` to `hyp`.
4. Run [analyze_word_errors.py](./scripts/analyze_word_errors.py).
5. Review artifacts: start with `summary.json`, then `error_details.csv` for utterance-level `ref` vs `hyp`, then `alignment_samples.txt` or `report.html` for visual inspection.
6. The script generates an HTML report by default (disable with `--no-html`). It also includes the raw model output (from the `output` column) in each HTML utterance card by default. Expect the script to download the ranked worst utterances' audio into the output directory and link them in the report with playable local controls.

## Python Environment

Always use `/home/boren/.virtualenvs/openai/bin/python` to run the script. The system python (`/home/linuxbrew/.linuxbrew/bin/python3`) lacks packages such as `blobfile` and `whisper`.

## Script

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
  --input-path path/to/result_details_2025-03-01_12-00-00.jsonl \
  --dataset openasr/ami \
  --output-dir tmp/asr-word-error-analysis
```

With model auto-discovery:

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
  --model en_hc_fy24_200k_step_7800 \
  --dataset openasr/ami \
  --output-dir tmp/asr-word-error-analysis
```

With verl `val_data_gen` discovery:

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-word-error-analysis/scripts/analyze_word_errors.py \
  --model verl_repeat/eval_openasr_remax_ls_raw_nodigits_v1_full_step100 \
  --dataset ami \
  --output-dir tmp/asr-word-error-analysis/ami \
  --top-n 30
```

Useful options:
- `--results-root az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1`: override the shared `result_details` root. The script picks the latest `result_details_*.jsonl` under `<model>/<dataset>/`.
- `--val-data-root az://orngwus2cresco/data/boren/outputs`: override the verl validation outputs root. Layout: `<root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl`. The script picks the latest numeric step.
- `--input-path ...`: bypass model discovery and use one local or `az://` JSONL file directly.
- `--ref-column gts --hyp-column clean_output`: optional explicit overrides for verl JSONL files. Usually not needed because the script auto-remaps these columns to `ref` / `hyp` when needed.
- `--raw-output-column output`: column containing the raw model output to include in each HTML utterance card (default: `output`). Set to empty string to disable.
- `--no-html`: disable the HTML report (enabled by default).
- `--normalizer openasr`: normalize ref/hyp with `recipe.phimm.utils.open_asr_normalizer.eval_utils.measure_wer`'s OpenASR normalizer path before alignment. Use this for OpenASR multilingual outputs when artifact WER should match reward/eval scoring.
- `--lang German` or `--lang-column language`: override or choose the row language used by `--normalizer openasr`. Language names are mapped through `recipe.phimm.utils.languages.LANGUAGES`.
- `--top-n 30`: control how many worst utterances appear in `alignment_samples.txt` and `report.html`.

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

## Verl Validation Data Discovery (val_data_gen)

The script auto-discovers verl validation outputs from `--val-data-root` (default: `az://orngwus2cresco/data/boren/outputs`) before falling back to the `result_details` blob root. Use this when you want to inspect a single eval output from a training experiment rather than compare two experiments.

Layout: `<val-data-root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl`

The dataset directory uses the bare dataset name, so `--dataset openasr/ami` and `--dataset ami` both search `val_data_gen/ami/`. The script picks the latest numeric step filename and auto-remaps the verl schema (`gts` -> `ref`, `clean_output` -> `hyp`).

Review expectations:
- Call out the discovered input path, especially the chosen step.
- Report WER and sub/del/ins as percentages from `summary.json`.
- For RL training outputs, inspect high-insertion utterances for hallucinated content beyond the utterance boundary.

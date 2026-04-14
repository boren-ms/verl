---
name: asr-detail-compare
description: Compare two ASR `result_details*.jsonl` outputs for the same dataset, especially when Codex needs to load remote `az://` files with `blobfile`, rank the utterances that contribute most to a target model's total WER, and show baseline vs target `hyp` differences side by side. Use for model-to-model detail analysis on the same eval set, top-error investigations, utterance-level WER debugging, and generating standalone HTML comparisons as the default review artifact.
---

# ASR Detail Compare

Compare utterance-level ASR detail files without re-implementing the same merge and WER logic each time. Prefer the bundled script for repeatable comparisons, and default to model-based discovery so you only need model names and dataset.

## Workflow
1. Confirm both models write results under the same results root in the layout `<results-root>/<model>/<dataset>/result_details_*.jsonl`.
2. Run `scripts/compare_result_details.py` with `--baseline-model`, `--target-model`, and `--dataset`. Only pass explicit paths when you need to override auto-discovery.
3. Run the script with `--write-html` by default so the main deliverable is a human-friendly utterance review page.
4. Read the generated `*.summary.json` for dataset-level WER numbers and use `*.topN.html` as the primary artifact. Use `*.topN.csv` when you need the raw ranked table for follow-up analysis.
5. Spot-check a few top-ranked rows before delivering, especially if the script had to fall back to row-order joins.

## Script
Run:

```bash
python skills/asr-detail-compare/scripts/compare_result_details.py   --baseline-model baseline   --target-model en_hc_fy24_200k_step_7800   --dataset openasr/ami   --output-dir tmp/asr-detail-compare   --write-html   --join-columns audio_file
```

Useful options:
- `--results-root az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1`: override the shared model-results root. The script picks the latest `result_details_*.jsonl` under each model and dataset.
- `--top-n 20`: keep the top 20 utterances by target-model error count.
- `--join-columns audio_file`: use `audio_file` as the default explicit join key.
- `--join-columns audio_file_stem`: force the join to use the stem derived from `audio_file` when the full path is not stable across runs.
- `--ref-column ref` and `--hyp-column hyp`: override schema defaults if needed.
- `--baseline-path ...` and `--target-path ...`: bypass model discovery and use explicit files.
- `--write-html`: write the default standalone HTML review page that shows `audio_file_stem`, `baseline_wer`, `target_wer`, and side-by-side `hyp_baseline` vs `hyp_target` with word-level highlights.
- `--write-full-csv`: also save the full utterance-level joined comparison.

## HTML Review Output
- Prefer `--write-html` by default. Use the HTML output as the main deliverable unless the user explicitly asks for CSV-only output.
- The HTML output is derived from the ranked `*.topN.csv`, so it reflects the same ordering and selection logic as the CSV.
- The page highlights changed word spans separately in baseline and target hypotheses; it is optimized for quick utterance-by-utterance scanning on desktop and mobile.
- If the CSV lacks `audio_file_stem`, the renderer falls back to `comparison_id` for the card title.

## Join Rules
- Prefer `audio_file` as the join key by default when it is present and unique in both files.
- If `audio_file` is unavailable or unstable, the script can fall back to other stable keys such as `audio_file_stem`, `utt_id`, `utterance_id`, `id`, `key`, or `audio_path`.
- `audio_file_stem` is derived automatically from the basename of `audio_file` without the extension.
- If no preferred key is unique in both files, it tries the combined preferred columns.
- If that still fails and `ref` is unique in both files, it joins on `ref`.
- Only if row counts match and no better key exists does it fall back to row order via `__row_idx`.

## Ranking Logic
- The script recomputes word-level edit counts from `ref` and `hyp` for both models.
- Target utterances are ranked by absolute target error count, which is the numerator contribution to total WER on a fixed dataset.
- The output includes substitutions, deletions, insertions, utterance WER, total-WER contribution, and `error_delta` vs baseline.

## Output
- `*.summary.json`: dataset-level totals and WER for baseline and target.
- `*.topN.html`: primary review artifact for the ranked rows when `--write-html` is enabled.
- `*.topN.csv`: companion ranked table with `ref`, `hyp_baseline`, `hyp_target`, and per-model error statistics.
- `*.full.csv`: optional full joined table when `--write-full-csv` is enabled.
- Each compared model's `result_details_*.jsonl` file is copied into the output directory for local inspection, with the filename prefixed by the model name.

## Review Expectations
- Call out the join key the script chose.
- Confirm whether `ref_matches_baseline` stays true for the reviewed rows.
- Highlight whether the target model regressed or improved on the highest-contributing utterances.
- When sharing HTML output, mention that the page reflects the ranked top-N rows rather than the full comparison unless `top-n` was set to cover the full dataset.

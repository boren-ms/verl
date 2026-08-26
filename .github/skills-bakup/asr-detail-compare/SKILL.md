---
name: asr-detail-compare
description: Compare two ASR `result_details*.jsonl` or verl training-validation JSONL outputs for the same dataset, especially when Codex needs to inspect `val_data_gen`, load remote `az://` files, rank utterances by WER impact, and generate clickable standalone HTML reports. Use for model-to-model detail analysis, top-error investigations, utterance-level WER debugging, and comparing verl training checkpoints at different steps.
---

# ASR Detail Compare

Compare utterance-level ASR detail files without re-implementing the same merge and WER logic each time. Prefer the bundled script for repeatable comparisons, and default to model-based discovery so you only need model names and dataset.

## Workflow
1. For verl training-validation comparisons, resolve `trainer.default_hdfs_dir` from the effective config, then list `<trainer.default_hdfs_dir>/val_data_gen/` and `<trainer.default_hdfs_dir>/val_data_gen/<dataset>/`. Confirm the requested numeric step files exist before downloading or comparing them. Do not substitute a separate benchmark or evaluation output directory.
2. Run `scripts/compare_result_details.py` with `--baseline-model`, `--target-model`, and `--dataset`. Only pass explicit paths when you need to override auto-discovery. The model name is `<project>/<experiment>` (e.g. `verl_repeat/eval_openasr`).
3. Run the script with `--write-html` by default so the main deliverable is a human-friendly utterance review page.
4. Read the generated `*.summary.json` for dataset-level WER numbers and improved/degraded/unchanged counts. Use the three HTML reports as the primary review artifacts:
   - `*.overall-topN.html` for the biggest changes
   - `*.improved-topN.html` for wins
   - `*.degraded-topN.html` for regressions
5. Spot-check a few top-ranked rows before delivering, especially if the script had to fall back to row-order joins.

## Python Environment
Always use `/home/boren/.virtualenvs/openai/bin/python` to run the script. The system python (`/home/linuxbrew/.linuxbrew/bin/python3`) lacks `pandas`, `blobfile`, and `whisper`.

## Script
Run:

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py   --baseline-model baseline   --target-model en_hc_fy24_200k_step_7800   --dataset openasr/ami   --output-dir tmp/asr-detail-compare   --write-html   --join-columns audio_file
```

Useful options:
- `--val-data-root az://orngwus2cresco/data/boren/outputs`: override the verl validation outputs root. Layout: `<root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl`. The script picks the latest step. Default: `az://orngwus2cresco/data/boren/outputs`.
- `--top-n 20`: keep the top 20 utterances by target-model error count.
- `--join-columns audio_file`: use `audio_file` as the default explicit join key.
- `--join-columns audio_file_stem`: force the join to use the stem derived from `audio_file` when the full path is not stable across runs.
- `--ref-column ref` and `--hyp-column hyp`: override schema defaults if needed. For verl training JSONL files, use `--ref-column gts --hyp-column clean_output`.
- `--normalizer openasr`: normalize ref/hyp with the same OpenASR path used by `recipe.phimm.utils.open_asr_normalizer.eval_utils.measure_wer` before recomputing utterance errors. Use this for OpenASR multilingual outputs when summary WER should match eval/reward counts.
- `--lang German` or `--lang-column language`: override or choose the row language used by `--normalizer openasr`. Language names are mapped through `recipe.phimm.utils.languages.LANGUAGES`; if absent, the script falls back to the suffix of `data_source`.
- `--baseline-path ...` and `--target-path ...`: bypass model discovery and use explicit files.
- `--write-html`: write the default standalone HTML review page that shows `audio_file_stem`, `baseline_wer`, `target_wer`, and side-by-side `hyp_baseline` vs `hyp_target` with word-level highlights.
- `--audio-blob-root az://orngwus2cresco/data/boren/data/openasr_jsonl`: enable audio playback in HTML reports. Downloads audio files for the top-N utterances from `{blob-root}/{dataset}/audio/{index}.wav` and embeds `<audio>` controls in each card. Only effective with `--write-html`.
- `--audio-local-dir ~/data/openasr_jsonl/{dataset}/audio`: override the local cache directory for downloaded audio. Defaults to `~/data/openasr_jsonl/{dataset}/audio`. Already-cached files are reused.
- `--write-full-csv`: also save the full utterance-level joined comparison.

## HTML Review Output
- Prefer `--write-html` by default. Use the HTML output as the main deliverable unless the user explicitly asks for CSV-only output.
- The HTML output is derived from the ranked `*.topN.csv`, so it reflects the same ordering and selection logic as the CSV.
- The page highlights changed word spans separately in baseline and target hypotheses; it is optimized for quick utterance-by-utterance scanning on desktop and mobile.
- Each card shows normalized text in the main comparison grid: Whisper English normalization by default, or OpenASR normalization when `--normalizer openasr` is selected. A collapsible "Raw output" section preserves the original un-normalized reference, baseline hypothesis, and target hypothesis.
- The "Raw output" section for the target uses the `output` column if present (showing the full model output with tags like `<ASR><lang=English><TXT>...</TXT></ASR>`). When `output` only exists on the target side (not baseline), the script finds it as an unsuffixed column after the pandas merge and uses it correctly.
- If the CSV lacks `audio_file_stem`, the renderer falls back to `comparison_id` for the card title.
- When `--audio-blob-root` is set, each card includes an `<audio>` player. Audio files are downloaded from `{blob-root}/{dataset}/audio/{row_index}.wav` (flat-indexed by position in the source dataset JSONL), cached to `--audio-local-dir`, and copied to `{output-dir}/audio/`. The HTML references audio via relative `audio/{index}.wav` paths.

## Presenting HTML Reports
After generating HTML reports, read the actual filenames from the `*.summary.json` `reports` section. For each report, show both:

- A clickable markdown link whose target is the workspace-relative path, so VS Code opens it.
- The complete absolute local filesystem path on the following line, so the location is unambiguous.

Example:

- Overall: [tmp/asr-detail-compare/ami/ami-step10-vs-step70.overall-top30.html](tmp/asr-detail-compare/ami/ami-step10-vs-step70.overall-top30.html)
  `/home/boren/code/verl/tmp/asr-detail-compare/ami/ami-step10-vs-step70.overall-top30.html`
- Improved: [tmp/asr-detail-compare/ami/ami-step10-vs-step70.improved-top30.html](tmp/asr-detail-compare/ami/ami-step10-vs-step70.improved-top30.html)
  `/home/boren/code/verl/tmp/asr-detail-compare/ami/ami-step10-vs-step70.improved-top30.html`
- Degraded: [tmp/asr-detail-compare/ami/ami-step10-vs-step70.degraded-top30.html](tmp/asr-detail-compare/ami/ami-step10-vs-step70.degraded-top30.html)
  `/home/boren/code/verl/tmp/asr-detail-compare/ami/ami-step10-vs-step70.degraded-top30.html`

Do not abbreviate, truncate, or replace these paths with only a directory.

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
The script generates **three separate reports** (each as CSV, and as HTML when `--write-html` is enabled):

1. **Overall** (`*.overall-topN`): Top-N utterances with the largest absolute change in error count between baseline and target, sorted by `|error_delta|` descending. Shows the biggest movers regardless of direction.
2. **Improved** (`*.improved-topN`): Top-N utterances where the target model reduced errors (`error_delta < 0`), sorted by `baseline_errors` descending. Shows the biggest wins.
3. **Degraded** (`*.degraded-topN`): Top-N utterances where the target model increased errors (`error_delta > 0`), sorted by `target_errors` descending. Shows the worst regressions.

Additional outputs:
- `*.summary.json`: dataset-level totals, WER for baseline and target, counts of improved/degraded/unchanged utterances, and paths to all report files.
- `*.full.csv`: optional full joined table when `--write-full-csv` is enabled.
- Each compared model's `result_details_*.jsonl` file is copied into the output directory for local inspection, with the filename prefixed by the model name.

## Verl Validation Data Discovery (val_data_gen)
The script auto-discovers verl validation outputs from `--val-data-root` (default: `az://orngwus2cresco/data/boren/outputs`). When using `--baseline-model` or `--target-model`, set the model name to `<project_name>/<experiment_name>` matching the verl config's `trainer.project_name`/`trainer.experiment_name`.

Layout: `<val-data-root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl`

Before using auto-discovery or explicit paths for a training run:

1. Read the effective `trainer.project_name`, `trainer.experiment_name`, and `trainer.default_hdfs_dir`. The usual interpolation is `az://orngwus2cresco/data/boren/outputs/${trainer.project_name}/${trainer.experiment_name}`.
2. List the resolved `<trainer.default_hdfs_dir>/val_data_gen/` directory to discover the actual dataset folder name.
3. List `<trainer.default_hdfs_dir>/val_data_gen/<dataset>/` and verify both requested `<step>.jsonl` objects exist.
4. Use those exact objects as `--baseline-path` and `--target-path` for two checkpoints from the same training run. This prevents accidentally launching or reading an unrelated in-house/OpenASR benchmark directory.

The script picks the latest step (highest numeric filename). The verl schema columns (`gts`→`ref`, `clean_output`→`hyp`) are auto-remapped, so `--ref-column` / `--hyp-column` overrides are not needed.

Discovery order: local paths → val_data_gen blob → result_details blob.

Example — compare two verl experiments on ami:
```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py \
  --baseline-model verl_repeat/eval_openasr \
  --target-model verl_repeat/eval_openasr_remax_ls_raw_nodigits_v1_full_step100 \
  --dataset ami \
  --output-dir tmp/asr-detail-compare/ami \
  --write-html \
  --top-n 30
```

## Verl Training Checkpoint Comparison
When comparing verl training outputs at different steps (e.g., step0 vs step200):
- First inspect `<trainer.default_hdfs_dir>/val_data_gen/<dataset>/` and confirm both requested numeric JSONL files are present. Training validation results belong here, not under `eval_2607_reports` or a long-evaluation output root.
- Use `--baseline-path` / `--target-path` with explicit local JSONL files and `--baseline-name` / `--target-name` for labels.
- Set `--ref-column gts --hyp-column clean_output` — verl JSONL uses `gts` for reference and `clean_output` for hypothesis.
- Check for a unique stable key such as `id` first and pass it through `--join-columns` when available. Only fall back to `__row_idx` when no stable key exists, row counts match, and raw `gts` sequences are identical in order.
- Verl JSONL schema: `input`, `output` (→ `raw_output`), `gts`, `clean_output`, `score`, `step`, `data_source`, `reward`, `n_err`, `n_ref`, `n_edge`, `n_fmt`, `n_lang`.

Example:
```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py \
  --baseline-path tmp/ami_analysis/step0.jsonl \
  --target-path tmp/ami_analysis/step200.jsonl \
  --baseline-name step0 \
  --target-name step200 \
  --dataset ami \
  --ref-column gts \
  --hyp-column clean_output \
  --output-dir tmp/ami_analysis/step0_vs_step200 \
  --write-html \
  --audio-blob-root az://orngwus2cresco/data/boren/data/openasr_jsonl \
  --top-n 30
```

## Comparing Mixed-Schema Files (eval_openasr baseline vs verl target)
When the baseline uses the standard eval schema (`ref`, `hyp`, `id`) and the target uses the verl schema (`gts`, `clean_output`, `output`, `id`), the script auto-remaps `gts`→`ref` and `clean_output`→`hyp` on whichever side is missing `ref`/`hyp`. No preprocessing or `--ref-column` override is needed.

If auto-remapping is not desired, you can still **preprocess the target file** manually:

```python
import json, pathlib
src = pathlib.Path("tmp/target_model/val_data_gen/ami/300.jsonl")
dst = pathlib.Path("tmp/asr-detail-compare/ami/target_normalized.jsonl")
with open(src) as f, open(dst, 'w') as out:
    for line in f:
        row = json.loads(line)
        row['ref'] = row.get('gts', '')
        row['hyp'] = row.get('clean_output', '')
        out.write(json.dumps(row) + '\n')
```

Then compare with `--join-columns id` (both schemas have `id`):
```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py \
  --baseline-path tmp/eval_openasr_step50/ami.jsonl \
  --target-path tmp/asr-detail-compare/ami/target_normalized.jsonl \
  --baseline-name eval_openasr \
  --target-name remax_nodigits_step300 \
  --dataset ami \
  --output-dir tmp/asr-detail-compare/ami \
  --write-html \
  --join-columns id \
  --top-n 30 \
  --audio-blob-root az://orngwus2cresco/data/boren/data/openasr_jsonl
```

Key points:
- Preserving the original `output` column in the preprocessed file ensures the HTML "Raw output" section shows the full model output (with `<ASR>` tags). The script picks up `output` from the target side automatically.
- Use `--join-columns id` — both eval_openasr and verl JSONL schemas include a unique `id` field.
- The baseline eval_openasr files live locally at `tmp/eval_openasr_step50/{dataset}.jsonl` (not on blob). The verl target files are at `tmp/{model_name}/val_data_gen/{dataset}/{step}.jsonl`.
- Loop over datasets for batch comparisons:
  ```bash
  for ds in ami earnings22 gigaspeech; do
    # preprocess target, then run compare
  done
  ```

## Review Expectations
- Call out the join key the script chose.
- Confirm whether `ref_matches_baseline` stays true for the reviewed rows.
- Highlight whether the target model regressed or improved on the highest-contributing utterances.
- When sharing HTML output, mention that the page reflects the ranked top-N rows rather than the full comparison unless `top-n` was set to cover the full dataset.
- For verl checkpoint comparisons, watch for hallucinated insertions (model generating extra content beyond the utterance boundary) — a common regression pattern during RL training.

---
name: asr-detail-compare
description: Compare two ASR result_details or verl val_data_gen JSONL outputs for the same dataset, including checkpoint-to-checkpoint comparisons, and build ranked HTML review reports. Also supports an explicitly requested single-result Reference/Target HTML view. Owns result alignment and comparison, not remote evaluation, entity-span scoring, or dataset reference preparation.
---

# ASR Detail Compare

Compare utterance-level ASR detail files without re-implementing the same merge and WER logic each time. Prefer the bundled script for repeatable comparisons, and default to model-based discovery so you only need model names and dataset.

## Scope and Handoffs

- Use the available `asr-word-error-analysis` skill for a general single-result
  error diagnosis; this skill owns two-result/checkpoint comparisons and its
  specific Reference/Target HTML renderer. Do not run both pipelines for the
  same deliverable.
- Use `asr-entity-error-analysis` for EER/EWER on annotated entity spans, not
  ordinary WER from this renderer.
- Use [align-long-audio-transcription](../align-long-audio-transcription/SKILL.md)
  to create segment references. Recording-level aggregation here is an analysis
  mode, not dataset preparation.
- This skill consumes existing outputs; it does not launch training/evaluation.

## Workflow
1. Determine whether the user supplied one result or two. With two results, compare baseline vs target normally. When the user supplies only one result or dataset, always treat it as the Target, synthesize its reference baseline as described in **Single-Result Reference Baseline**, and pass `--hide-baseline` so the HTML shows only Reference and Target. Do not ask for a second model result. Use baseline-only mode only when the user explicitly requests a baseline-only review.
2. For verl training-validation comparisons, resolve `trainer.default_hdfs_dir` from the effective config, then list `<trainer.default_hdfs_dir>/val_data_gen/` and `<trainer.default_hdfs_dir>/val_data_gen/<dataset>/`. Confirm the requested numeric step files exist before downloading or comparing them. Do not substitute a separate benchmark or evaluation output directory.
3. Run `scripts/compare_result_details.py` with `--baseline-model`, `--target-model`, and `--dataset`. Only pass explicit paths when you need to override auto-discovery. The model name is `<project>/<experiment>` (e.g. `verl_repeat/eval_openasr`). For a single result, first create the reference-baseline JSONL described in **Single-Result Reference Baseline**, then pass it through `--baseline-path` and the supplied result through `--target-path`.
4. Run the script with `--write-html` by default so the main deliverable is a human-friendly utterance review page.
5. Read the generated `*.summary.json` for dataset-level WER numbers and improved/degraded/unchanged counts. Use the three HTML reports as the primary review artifacts:
   - `*.overall-topN.html` for the biggest changes
   - `*.improved-topN.html` for wins
   - `*.degraded-topN.html` for regressions
6. Spot-check a few top-ranked rows before delivering, especially if the script had to fall back to row-order joins.
7. Deliver actual generated files using [Report Path Delivery](#report-path-delivery).

## Single-Result Reference Baseline
When only one result JSONL is provided, synthesize a baseline JSONL from that same file so every baseline hypothesis equals its reference. Preserve every row and all identity, language, audio, and metadata columns so the normal join and HTML rendering remain available.

- Standard eval schema: copy `ref` into `hyp`.
- Verl validation schema: copy `gts` into `clean_output`.
- If explicit `--ref-column` and `--hyp-column` overrides are needed, copy the selected reference column into the selected hypothesis column.
- Write the derived file under the comparison output directory with a descriptive name such as `<target>.reference-baseline.jsonl`; do not modify the supplied result file.
- Run the comparison with `--baseline-name reference`, use the supplied result/model name as `--target-name`, and pass `--hide-baseline` so the scoring-only synthetic baseline is not rendered.
- Prefer a stable unique key such as `audio_file` or `id`, following the normal join rules.
- Interpret the result as target errors against ground truth: reference-baseline WER is expected to be `0%`, there are no genuine "improved" rows, and target errors appear as degradations from the perfect reference baseline.

Example for a standard eval result:
```bash
/home/boren/.virtualenvs/openai/bin/python -c '
import json, pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:3])
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        row = json.loads(line)
        row["hyp"] = row["ref"]
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
' target.jsonl target.reference-baseline.jsonl

/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py \
  --baseline-path target.reference-baseline.jsonl \
  --target-path target.jsonl \
  --baseline-name reference \
  --target-name target \
  --hide-baseline \
  --dataset DATASET \
  --output-dir tmp/asr-detail-compare/DATASET/reference-vs-target \
  --write-html \
  --join-columns audio_file
```

## Python Environment
The examples use `/home/boren/.virtualenvs/openai/bin/python` when available.
Otherwise use the configured project interpreter and verify `pandas`, `blobfile`,
`whisper`, and scoring dependencies before execution.

## Script
Run:

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py   --baseline-model baseline   --target-model en_hc_fy24_200k_step_7800   --dataset openasr/ami   --output-dir tmp/asr-detail-compare   --write-html   --join-columns audio_file
```

Useful options:
- `--val-data-root az://orngwus2cresco/data/boren/outputs`: override the verl validation outputs root. Layout: `<root>/<project>/<experiment>/val_data_gen/<dataset>/<step>.jsonl`. The script picks the latest step. Default: `az://orngwus2cresco/data/boren/outputs`.
- `--top-n 20`: keep up to 20 rows per report, using the ranking described in [Output](#output).
- `--join-columns audio_file`: use `audio_file` as the default explicit join key.
- `--join-columns audio_file_stem`: force the join to use the stem derived from `audio_file` when the full path is not stable across runs.
- `--aggregate-segments-by parent_audio_path`: for long-form files repeating the full reference, group by the actual parent-recording key, order by `seg_index`, and concatenate hypotheses before computing WER. Use `id` only if it identifies the parent, not individual segments. Use `--segment-index-column` to override ordering. For a single-result review, aggregate before synthesizing a perfect reference baseline so the full transcript is not repeated once per segment.
- `--ref-column ref` and `--hyp-column hyp`: override schema defaults if needed. For verl training JSONL files, use `--ref-column gts --hyp-column clean_output`.
- `--normalizer auto`: default. Infer each row's language, use the same HF normalization and compound-aware `kaldialign` scoring as `openasr_en_eval` for English (`en`), and use the language-specific OpenASR normalizer for every other language.
- `--normalizer english` or `--normalizer openasr`: explicitly force the English path or the OpenASR dispatch path when auto-detection is not appropriate.
- `--lang German` or `--lang-column language`: override or choose the row language used by automatic/OpenASR normalization. Language names are mapped through `recipe.phimm.utils.languages.LANGUAGES`; if absent, the script falls back to the suffix of `data_source` and then English.
- `--baseline-path ...` and `--target-path ...`: bypass model discovery and use explicit files.
- `--hide-baseline`: keep the baseline in scoring and CSV output but omit its metric and transcript pane from HTML. Use this for single-result reference-vs-target reviews.
- Omit `--target-path`/`--target-model` to generate a baseline-only report ranked by baseline errors. The Target pane starts hidden, and Reference and Baseline expand to fill its space.
- `--write-html`: write the default standalone HTML review page that shows `audio_file_stem`, `baseline_wer`, `target_wer`, and side-by-side `hyp_baseline` vs `hyp_target` with word-level highlights.
- Audio playback is optional. Neither an audio path column nor downloadable audio is required to generate CSV, summary, or HTML reports; omit all audio options when playback is unnecessary or unavailable.
- Prefer existing local audio over remote downloads. When playback is requested, first reuse a valid local path from the selected audio path column or an existing file in `--audio-local-dir`; only retrieve blob audio when no suitable local file is available.
- `--audio-blob-root az://orngwus2cresco/data/boren/data/openasr_jsonl`: optionally enable audio playback in HTML reports. For audio not already available locally, downloads the top-N files from `{blob-root}/{dataset}/audio/{index}.wav` and embeds `<audio>` controls in each card. Only effective with `--write-html`.
- `--audio-local-dir ~/data/openasr_jsonl/{dataset}/audio`: override the local audio/cache directory. Defaults to `~/data/openasr_jsonl/{dataset}/audio`; existing files must be reused rather than downloaded again.
- `--audio-path-column audio_path`: optionally resolve audio from each row's direct local or blob path instead of assuming the flat `{blob-root}/{dataset}/audio/{index}.wav` layout. Local files are copied directly under the report's `audio/` directory; remote files are downloaded only when needed.
- `--write-full-csv`: also save the full utterance-level joined comparison.

## HTML Review Output
- Prefer `--write-html` by default. Use the HTML output as the main deliverable unless the user explicitly asks for CSV-only output.
- The HTML output is derived from the ranked `*.topN.csv`, so it reflects the same ordering and selection logic as the CSV.
- The page directly aligns the normalized baseline hypothesis against the normalized target hypothesis and uses changed spans for navigation and the "Differences only" filter without adding separate model-change word styling. It also aligns each model to the reference using compound-aware `kaldialign`: Reference and correct model words remain neutral, incorrect tokens containing digits are red, inserted model words are yellow, substitutions are green, and deletion placeholders are light blue. In compact mode, each model transcript keeps only incorrect words and the five neighboring words on either side; overlapping windows are merged. "Show full context" restores the complete normalized hypotheses.
- On desktop, each card presents the available Reference, Baseline, and Target panes as synchronized columns that automatically fill the full report width. Single-result target reviews hide the synthetic Baseline pane, leaving Reference and Target. Drag a border between visible panes to redistribute their widths, use Left/Right while a border is focused for keyboard resizing, or double-click it to reset equal widths. On narrow screens, resize borders are hidden and panes stack in the same order.
- Toolbar checkboxes independently hide Reference, Baseline, or Target across every utterance, and duplicate toolbar controls stay synchronized. Sync-scroll and full/compact-context controls also apply report-wide; Previous/Next remains local because changed spans differ per utterance. Remaining visible panes automatically expand to use all freed width: two panes split it evenly and one pane fills it. In single-result target mode, Baseline starts hidden and cannot be enabled; in explicit baseline-only mode, Target does likewise.
- The report-wide "Differences only" checkbox keeps only baseline-to-target changed tokens and their aligned reference words, while hiding surrounding tokens, context ellipses, and raw sections. Duplicate controls stay synchronized across utterances. The checkbox is available only when both Baseline and Target panes exist.
- The Substitution, Deleted, Inserted, and Number error legend chips are report-wide color controls. Unchecking a type removes that color from all transcript tokens and count badges without hiding their text; duplicate controls remain synchronized across utterances. Deletion controls affect the aligned `∅` placeholders independently from substitutions.
- When audio is available, each utterance toolbar keeps its status/player and review controls in one compact line. Reports without audio remain fully functional. Audio uses only its content width; narrow viewports preserve the line with horizontal scrolling instead of wrapping it into multiple rows.
- Each utterance card uses a compact single-line metric strip. Every visible model metric keeps its label, WER, total errors, and `Sub`/`Del`/`Ins`/`Num` counts on one line; the strip scrolls horizontally instead of wrapping on narrow screens. `Num` is the count of highlighted incorrect tokens containing digits and overlaps the standard substitution/insertion edit categories rather than adding to their total.
- Every normalized reference, baseline, and target word is selectable. Clicking a word, or focusing it and pressing Enter/Space, highlights all words and gap placeholders at the corresponding aligned reference position across the three transcripts and automatically centers each counterpart in its pane. Compound matches can select multiple reference words for one model token.
- Each card uses `_HFEnglishTextNormalizer` for English text. The normalized reference remains visible, with its original text under a collapsed "Raw reference" disclosure. Each normalized baseline/target hypothesis has its own collapsed raw output directly beneath it.
- The "Raw output" section for the target uses the `output` column if present (showing the full model output with tags like `<ASR><lang=English><TXT>...</TXT></ASR>`). When `output` only exists on the target side (not baseline), the script finds it as an unsuffixed column after the pandas merge and uses it correctly.
- If the CSV lacks `audio_file_stem`, the renderer falls back to `comparison_id` for the card title.
- When audio playback is enabled, each available file is copied to `{output-dir}/audio/` and referenced by the HTML with a relative path. Reuse direct local files and files already present in `--audio-local-dir`; with `--audio-blob-root`, download `{blob-root}/{dataset}/audio/{row_index}.wav` only for missing local files. Missing audio does not block report generation.

## Join Rules
- Prefer `audio_file` as the join key by default when it is present and unique in both files.
- If `audio_file` is unavailable or unstable, the script can fall back to other stable keys such as `audio_file_stem`, `utt_id`, `utterance_id`, `id`, `key`, or `audio_path`.
- `audio_file_stem` is derived automatically from the basename of `audio_file` without the extension.
- If no preferred key is unique in both files, it tries the combined preferred columns.
- If that still fails and `ref` is unique in both files, it joins on `ref`.
- Only if row counts match and no better key exists does it fall back to row order via `__row_idx`. Before accepting that fallback, verify the reference sequence is identical in order on both sides.

## Ranking Logic
- The script recomputes word-level edit counts from `ref` and `hyp` for both models.
- Error counts measure contributions to total WER on a fixed dataset; each report uses its own ordering below.
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

## Report Path Delivery
Read actual filenames from the `*.summary.json` `reports` section. For each
generated HTML report, provide:

1. A clickable workspace link whose target is the absolute Linux path.
2. On WSL, a copyable Windows Explorer path generated with `wslpath -w`. If
   WSL path conversion is unavailable, provide only the absolute local link;
   do not guess a distribution name or fabricate a UNC path.

Example:
```bash
wslpath -w /home/boren/code/verl-mirror/tmp/asr-detail-compare/ami/baseline-vs-target.overall-top30.html
```

Present the result like this:
```markdown
- Overall: [baseline-vs-target.overall-top30.html](/home/boren/code/verl-mirror/tmp/asr-detail-compare/ami/baseline-vs-target.overall-top30.html)
  - WSL: `\\wsl.localhost\Ubuntu\home\boren\code\verl-mirror\tmp\asr-detail-compare\ami\baseline-vs-target.overall-top30.html`
```

Apply this to the overall, improved, and degraded HTML reports. If an optional report was not generated, do not invent a path for it.

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
- Verl `gts`/`clean_output` columns are auto-remapped. Explicit `--ref-column gts --hyp-column clean_output` is optional when both files use that schema; do not force it on mixed-schema inputs.
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

Compare the original files directly, using `id` only after verifying it is
present and unique on both sides:
```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/asr-detail-compare/scripts/compare_result_details.py \
  --baseline-path tmp/eval_openasr_step50/ami.jsonl \
  --target-path tmp/target_model/val_data_gen/ami/300.jsonl \
  --baseline-name eval_openasr \
  --target-name remax_nodigits_step300 \
  --dataset ami \
  --output-dir tmp/asr-detail-compare/ami \
  --write-html \
  --join-columns id \
  --top-n 30 \
  --audio-blob-root az://orngwus2cresco/data/boren/data/openasr_jsonl
```

The paths above are examples, not fixed discovery locations. The script retains
the target `output` column for raw-output rendering without preprocessing.

## Review Expectations
- Call out the join key the script chose.
- Confirm whether `ref_matches_baseline` stays true for the reviewed rows.
- Highlight whether the target model regressed or improved on the highest-contributing utterances.
- When sharing HTML output, mention that the page reflects the ranked top-N rows rather than the full comparison unless `top-n` was set to cover the full dataset.
- For verl checkpoint comparisons, watch for hallucinated insertions (model generating extra content beyond the utterance boundary) — a common regression pattern during RL training.

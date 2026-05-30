---
name: inhouse-asr-compare
description: >-
  Compare in-house ASR eval JSON dumps (records with `UtteranceId`, `UtteranceTERMetrics`, and `Metrics[*].EntityInfo`) on a chosen TER metric (default `DisfluencyTolerant_TER`) and optionally on entity recognition (EER/EWER). Two modes — (1) baseline-vs-target HTML reports that show only positions where the two models disagree, and (2) single-model HTML reports that show only positions where one model diverges from the reference. Use for utterance-level TER debugging, splitting disagreements into formatting (punc/cap/itn) vs lexical (sub/ins/del) vs entity (EER/EWER/recall/precision) categories, ranking improved/degraded (compare mode) or worst (single-model mode) utterances, and producing HTML with a sticky left sidebar, a sortable summary table, and word-aligned diff blocks. Triggers: "compare in-house eval", "TER disagreement report", "per-utterance TER diff", "improved/degraded utterances", "compare two in-house ASR JSON dumps", "EER compare", "entity error compare", "single model ref diff", "one model vs reference", "show errors vs reference".
---

# In-house ASR TER Compare

Compare two in-house ASR eval JSON files at the utterance level on a chosen `UtteranceTERMetrics` entry and emit HTML that hides any position where the two models agree (with or without the reference), so review focuses on actual model-vs-model divergence.

## Input format
Each input is a JSON list of records:

```json
{
  "UtteranceId": "...",
  "UtteranceTERMetrics": [
    {
      "MetricName": "DisfluencyTolerant_TER",
      "ter_info": {"number_of_tokens": int, "number_of_edits": int, "display_ter": float},
      "display_form_tx": "ref text",
      "display_form_hyp": "hyp text",
      "word_align": ["ref:hyp", "NULL:hyp", "ref:NULL", ...],
      "word_ter_class": [["lexical_clean"], ["punc_none_2_comma"], ["cap_upper_2_lower"], ["relaxation"], ...],
      "ter_category_info": {"ter_categories": {"punc": {...}, "cap": {...}, "itn": {...}, "lexical": {...}, "others": {...}}}
    },
    ... (other metrics: NonDisfluency_TER, NonVerbatim_TER, Verbatim_TER)
  ]
}
```

The script joins on `UtteranceId`. Both files must use the same reference (the `display_form_tx` stream).

## Python environment
Always use `/home/boren/.virtualenvs/openai/bin/python`. Standard library only — no extra deps required.

## Run

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/inhouse-asr-compare/scripts/compare_ter_disagree.py \
  --baseline-path tmp/baseline_model.json \
  --target-path  tmp/target_model.json \
  --baseline-name baseline \
  --target-name   target \
  --metric DisfluencyTolerant_TER \
  --output-dir tmp/ter_compare_disagree \
  --top-n 30
```

Options:
- `--metric` — one of `NonDisfluency_TER`, `NonVerbatim_TER`, `DisfluencyTolerant_TER` (default), `Verbatim_TER`.
- `--top-n` — cap per (category, split) page. Set high (e.g. 9999) to include every utterance.
- `--baseline-name` / `--target-name` — labels shown in HTML; default to file stems.
- `--include-entity` — also emit entity comparison pages (see "Entity comparison" below).
- `--entity-metric` — `Metrics[*].MetricName` whose `EntityInfo` block to read. Default `NonDisfluency_Simple_StandardRelax`. Other common values: `NonVerbatim_Simple_StandardRelax`, `Verbatim_Simple_StandardRelax`, `Lexical_Simple_StandardRelax`.

## Entity comparison
When `--include-entity` is passed, the script also walks each top-level record's `Metrics[*]` list, picks the entry whose `MetricName == --entity-metric`, and reads its `EntityInfo` block. Fields used:

- `NumTransEnts`, `NumRecoEnts`, `NumTransEntsMatched`, `NumRecoEntsMatched`
- `NumEntWords`, `NumEntWordSub`, `NumEntWordIns`, `NumEntWordDel`, `NumEntEdits`
- `EntityErrorRate` (EER), `EntityWordErrorRate` (EWER), `EntityRecall`, `EntityPrecision`
- `EntityAlignmentAsString` (rendered verbatim in cards)
- `ListTransUniqEnts`, `ListRecoUniqEnts` (rendered as `name×count` chips)

Two extra reports are emitted: `{stem}.entity.improved-topN.html` and `{stem}.entity.degraded-topN.html`, ranked by Δ `NumEntEdits`. Cards are filtered to utterances that have at least one entity on either side (trans or reco). The summary table shows per-utterance Δ `ent edits`, Δ `ent words`, Δ `EWER`, Δ `EER`, Δ `recall`, Δ `precision`, plus a totals row that micro-averages EER/EWER/recall/precision over the displayed utterances (so the overall row reflects pooled rates, not arithmetic means). Each entity card shows the chip stats above and a side-by-side block of each model's `EntityAlignmentAsString`.

## Output layout
Under `--output-dir`, the script writes four HTML pages plus a JSON summary (six pages with `--include-entity`):

- `{stem}.fmt.improved-topN.html`     — utterances where target made fewer formatting edits (sorted by Δ fmt edits, most-improved first)
- `{stem}.fmt.degraded-topN.html`     — target made more formatting edits (sorted by Δ fmt edits, worst first)
- `{stem}.lexical.improved-topN.html` — target made fewer lexical sub/ins/del edits
- `{stem}.lexical.degraded-topN.html` — target made more lexical sub/ins/del edits
- `{stem}.entity.improved-topN.html`  — (with `--include-entity`) target made fewer entity-word edits
- `{stem}.entity.degraded-topN.html`  — (with `--include-entity`) target made more entity-word edits
- `{stem}.summary.json`               — dataset-level TER, improved/degraded/unchanged counts, and the `reports` map of generated HTML paths

`stem = {baseline-name}__vs__{target-name}.{metric}.disagree`.

There is no "overall" page — only improved and degraded per category. Sort key is the page's own category (`lex_edits` delta on lexical pages, `fmt_edits` delta on fmt pages).

## HTML report anatomy
Each page is a single self-contained HTML file (embedded CSS, no JS).

A fixed-position **toggle button** in the top-left (`☰ hide` / `☰ show`) collapses or expands the sidebar — pure CSS via a hidden checkbox, so no script is required and the page stays self-contained.

Left **sticky sidebar** (hidden when toggle is checked):
- "Reports" nav: link to `summary.json` and to all 4 sibling category pages (current page highlighted).
- "Utterances · Δ lex" / "Δ fmt" ordered list — each entry is the `UtteranceId` plus its category-edit delta chip (red = worse, green = better). Clicking jumps to the matching card.

Main panel:
- **Summary table** (open `<details>`) listing every utterance on the page, with an "overall" row pinned at the top of the body:
  - Columns: `#`, `UtteranceId`, `Δ {cat} edits`, **per-category breakdown deltas** (`Δ sub / Δ ins / Δ del` on lexical pages; `Δ punc / Δ cap / Δ itn` on fmt pages), `Δ edits`, `Δ TER`, baseline TER, target TER.
  - Overall row shows summed deltas and micro-averaged TER over the displayed utterances.
- **Per-utterance cards** (anchor = slugified `UtteranceId`):
  - Stats chips show only the page's own category — lexical pages show `lexical: B→T (Δ)` + `lex breakdown: sub/ins/del`; fmt pages show `fmt: B→T (Δ)` + `fmt breakdown: punc/cap/itn`. TER, total edits, and disagreement-cell count are always shown.
  - Diff table is the standard 4-row block (`#`, `ref`, baseline-name, target-name), wrapped every 12 columns to prevent horizontal overflow.
  - Rows are filtered to positions where baseline and target diverge (not just where they differ from ref). On lexical pages a normalization pass (lowercase + strip surrounding punctuation) drops case-/punct-only differences. Consecutive identical insertion tokens (e.g. `+the +the +the` from forced-alignment artifacts) collapse to one.

## Category classification
For each diff row, `row_category()`:
- `lexical` — any side has `sub`, `del`, or `ins` bucket, or recorded insertions on either side.
- `fmt`     — both sides are `eq`/`relax`/`fmt`-only.

Lexical decomposition (`lex_sub / lex_ins / lex_del`) is computed by walking `word_align` and counting `lexical*` tags: `:NULL` → del, `NULL:` → ins, else sub. Per-category totals (`punc / cap / itn / lexical / fmt = punc+cap+itn`) come straight from `ter_category_info.ter_categories.*.number_of_edits`.

## Single-model mode (one model vs reference)
When only one model is provided, use `single_model_ref_diff.py` instead of `compare_ter_disagree.py`. It reads one JSON dump and emits HTML pages that show ONLY positions where the model's hypothesis diverges from the reference (no baseline/target diff, just edits vs ref).

```bash
/home/boren/.virtualenvs/openai/bin/python .github/skills/inhouse-asr-compare/scripts/single_model_ref_diff.py \
  --model-path tmp/my_model.json \
  --model-name my_model \
  --metric DisfluencyTolerant_TER \
  --output-dir tmp/my_model_ref_diff \
  --top-n 50 \
  --include-entity
```

Flags mirror the compare script except there is only `--model-path` / `--model-name` (no baseline/target pair). Each category produces a single "worst" page sorted by that category's edit count (descending), so there is no improved/degraded split:

- `{stem}.fmt.worst-topN.html`
- `{stem}.lexical.worst-topN.html`
- `{stem}.entity.worst-topN.html` (with `--include-entity`)
- `{stem}.summary.json`

`stem = {model-name}.{metric}.ref-diff`.

Per-utterance cards show a 3-row diff block (`#` / `ref` / `{model-name}`) with only ref positions where the model edited (sub/del/ins/fmt) or inserted tokens. Stats chips show absolute counts (no deltas). Summary table totals show pooled TER over displayed utterances.

## Presenting reports
After generating (compare mode), read filenames from `{stem}.summary.json` `reports` and present all four as workspace-relative markdown links:

- Lexical improved: [tmp/ter_compare_disagree/...lexical.improved-topN.html](tmp/ter_compare_disagree/baseline__vs__target.DisfluencyTolerant_TER.disagree.lexical.improved-topN.html)
- Lexical degraded: [tmp/ter_compare_disagree/...lexical.degraded-topN.html](tmp/ter_compare_disagree/baseline__vs__target.DisfluencyTolerant_TER.disagree.lexical.degraded-topN.html)
- Fmt improved:     [tmp/ter_compare_disagree/...fmt.improved-topN.html](tmp/ter_compare_disagree/baseline__vs__target.DisfluencyTolerant_TER.disagree.fmt.improved-topN.html)
- Fmt degraded:     [tmp/ter_compare_disagree/...fmt.degraded-topN.html](tmp/ter_compare_disagree/baseline__vs__target.DisfluencyTolerant_TER.disagree.fmt.degraded-topN.html)

Use the actual filenames from the summary; do not truncate.

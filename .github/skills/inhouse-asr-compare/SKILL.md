---
name: inhouse-asr-compare
description: 'Compare in-house ASR eval JSON dumps (records with `UtteranceId`, `UtteranceTERMetrics`, `Metrics[*].EntityInfo`) on a TER metric (default `DisfluencyTolerant_TER`) and optionally entity recognition (EER/EWER). Two modes: baseline-vs-target reports showing only model-disagreement positions, and single-model reports showing only positions diverging from the reference. Splits errors into formatting (punc/cap/itn), lexical (sub/ins/del), and entity categories, ranks improved/degraded or worst utterances, and emits self-contained HTML. Triggers: "compare in-house eval", "TER disagreement report", "per-utterance TER diff", "EER compare", "entity error compare", "single model ref diff".'
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

### verl long-eval `details.jsonl` (auto-detected)
Both scripts also natively read verl `main_long_eval_asr` outputs — a `.jsonl` file with one JSON object tmp/MAI-15.json vs  per line (no manual conversion needed). Format is auto-detected: a file starting with `[` is treated as an in-house JSON list, otherwise each line is parsed as a verl record. A record looks like:

```json
{
  "parent_audio_path": ".../wav/<guid>_0.wav",
  "id": "...", "data_source": "...", "language": "en",
  "ref": "ref text", "hyp": "hyp text",
  "dter": 0.1859, "dter_n_err": 7777, "dter_n_ref": 41826,
  "eer": 0.0, "eer_n_err": 0, "eer_n_ref": 0,
  "dter_detail": {"word_align": [...], "word_ter_class": [...], "ter_category_info": {...}}
}
```

Conversion to the internal schema:
- `UtteranceId` = recording GUID extracted from `parent_audio_path` (`/wav/<guid>_0.wav`), which matches the in-house `UtteranceId` for the same recording. Falls back to `id` if no GUID is found.
- A single `UtteranceTERMetrics` entry is synthesized from `dter_detail`; `display_ter = dter * 100`, `number_of_tokens = dter_n_ref`, `number_of_edits = dter_n_err`. The long-eval pipeline computes only a disfluency-tolerant TER, so the same detail is used regardless of the `--metric` requested.
- You can mix formats — e.g. a verl `details.jsonl` as `--baseline-path` and an in-house JSON as `--target-path` — as long as the recording GUIDs join.

Segment boundaries: when a verl record carries per-segment responses (the `responses` field — a list of plain segment strings — or the legacy `raw_response` field with one `<TXT>` block per segment), the **lexical and fmt pages** draw a thin amber boundary marker (`┊` with an `S{n}` label) before the reference token that begins each segment, so you can see where one segment response ends and the next starts. Boundaries are mapped onto reference positions by accumulating consumed hypothesis characters across `word_align`. Up to 2 reference words on each side of every boundary are shown as muted context rows (the same window used around errors), so the words adjacent to a boundary are always visible. `responses` is preferred when present; markers appear only for inputs that provide `responses` or `raw_response`.

Limitation: verl records carry no rich `Metrics[*].EntityInfo`, so `--include-entity` is a no-op for them (all-zero entity entries). Use in-house JSON dumps on both sides for entity comparison.

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

One extra report is emitted: `{stem}.entity.top-errors-topN.html`, ranked by absolute Δ `NumEntEdits`. Cards are filtered to utterances that have at least one entity on either side (trans or reco). The summary table shows per-utterance Δ `ent edits`, Δ `ent words`, Δ `EWER`, Δ `EER`, Δ `recall`, Δ `precision`, plus a totals row that micro-averages EER/EWER/recall/precision over the displayed utterances (so the overall row reflects pooled rates, not arithmetic means). Each entity card shows the chip stats above and a side-by-side block of each model's `EntityAlignmentAsString`.

## Output layout
Under `--output-dir`, the script writes three HTML pages plus a JSON summary (four pages with `--include-entity`):

- `{stem}.overall.top-errors-topN.html`   — largest absolute total-edit deltas, regardless of sign
- `{stem}.fmt.top-errors-topN.html`   — largest absolute formatting-edit deltas, regardless of sign
- `{stem}.lexical.top-errors-topN.html` — largest absolute lexical-edit deltas, regardless of sign
- `{stem}.entity.top-errors-topN.html` — (with `--include-entity`) largest absolute entity-word edit deltas, regardless of sign
- `{stem}.summary.json`               — dataset-level TER, improved/degraded/unchanged counts, and the `reports` map of generated HTML paths

`stem = {baseline-name}__vs__{target-name}.{metric}.disagree`.

Overall pages use the utterance's total edit delta (`target edits - baseline edits`). Category pages use the page's own category (`lex_edits` delta on lexical pages, `fmt_edits` delta on fmt pages, entity-word edit delta on entity pages). Only `top-errors` pages are emitted; improved/degraded pages are intentionally omitted. Each top-errors page is filtered to nonzero-delta utterances and sorted by absolute delta magnitude.

## HTML report anatomy
Each page is a single self-contained HTML file (embedded CSS plus one tiny inline script — no external assets).

A fixed-position **toggle button** in the top-left (`☰ hide` / `☰ show`) collapses or expands the sidebar. The show/hide itself is pure CSS via a hidden checkbox (works even with JS disabled); a small inline script only keeps the utterance currently under the viewport top fixed in place, so toggling never scroll-jumps you off the card you are reading. Native scroll anchoring is disabled (`overflow-anchor: none`) so that correction is exact even though full-width reflow changes the document height.

Left **sticky sidebar** (hidden when toggle is checked):
- "Reports" nav: link to `summary.json` and to all sibling pages (current page highlighted).
- "Utterances · Δ edits" / "Δ lex" / "Δ fmt" ordered list — each entry is the `UtteranceId` plus its page-specific delta chip (red = worse, green = better). Clicking jumps to the matching card.

Main panel:
- **Summary table** (open `<details>`) listing every utterance on the page, with an "overall" row pinned at the top of the body:
  - Overall page columns: `#`, `UtteranceId`, `Δ edits`, `Δ lex`, `Δ fmt`, `Δ TER`, baseline TER, target TER.
  - Category page columns: `#`, `UtteranceId`, `Δ {cat} edits`, **per-category breakdown deltas** (`Δ sub / Δ ins / Δ del` on lexical pages; `Δ punc / Δ cap / Δ itn` on fmt pages), `Δ edits`, `Δ TER`, baseline TER, target TER.
  - Overall row shows summed deltas and micro-averaged TER over the displayed utterances.
- **Per-utterance cards** (anchor = slugified `UtteranceId`):
  - Stats chips show the page's own focus. Overall pages show both lexical and fmt chips; lexical pages show `lexical: B→T (Δ)` + `lex breakdown: sub/ins/del`; fmt pages show `fmt: B→T (Δ)` + `fmt breakdown: punc/cap/itn`. TER, total edits, and disagreement-cell count are always shown.
  - Diff table is the standard 4-row block (`#`, `ref`, baseline-name, target-name), wrapped every 12 columns to prevent horizontal overflow.
  - Diff tokens are color-coded by edit bucket: lexical edits as `sub` / `del` / `ins`, and formatting edits by their **detailed subtype** `punc` / `cap` / `itn` / `others` (each its own color) rather than one generic "fmt" color. So on the fmt page you can see at a glance which formatting dimension diverged.
  - Each formatting token carries a small monospace **subtype badge** beneath it (from `fmt_subtype()`): punctuation/capitalization transitions render as `from→to` (e.g. `,→∅`, `.→,`, `lc→UC`), ITN edits show their kind (e.g. `num`, `money`, `ordinal`), and anything else shows `other`.
  - **Context words:** each kept divergence is surrounded by up to 2 reference words on each side (muted "ctx" cells; the model rows show a faint `·` ditto to indicate agreement at those positions). Longer agreement runs between error clusters still collapse to a `… N …` gap marker. Context width is the `n_ctx=2` argument to `filter_rows_by_category()`.
  - Rows are filtered to positions where baseline and target diverge (not just where they differ from ref). On lexical pages a normalization pass (lowercase + strip surrounding punctuation) drops case-/punct-only differences. Consecutive identical insertion tokens (e.g. `+the +the +the` from forced-alignment artifacts) collapse to one.

## Category classification
For each diff row, `row_category()`:
- `lexical` — any side has `sub`, `del`, or `ins` bucket, or recorded insertions on either side. The lexical page therefore only surfaces genuine lexical (sub/ins/del) disagreements.
- `fmt`     — both sides are `eq`/`relax`/formatting-only (`punc` / `cap` / `itn` / `others`). The fmt page surfaces only formatting disagreements and colors each token by its detailed subtype.

The per-token formatting bucket is derived in `bucket_for()` from the `word_ter_class` tag prefix: `punc_*` → `punc`, `cap_*` → `cap`, `itn_*` → `itn`, anything else → `others` (`FMT_BUCKETS = ("punc", "cap", "itn", "others")`).

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
After generating (compare mode), read filenames from `{stem}.summary.json` `reports` and present the overall, lexical, and fmt top-errors pages as workspace-relative markdown links. The labels and path patterns are:

- Overall top errors: `tmp/ter_compare_disagree/...overall.top-errors-topN.html`
- Lexical top errors: `tmp/ter_compare_disagree/...lexical.top-errors-topN.html`
- Fmt top errors: `tmp/ter_compare_disagree/...fmt.top-errors-topN.html`

Use the actual filenames from the summary; do not truncate.

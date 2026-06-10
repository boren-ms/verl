"""Compare DisfluencyTolerant_TER (or any UtteranceTERMetrics entry) between
two in-house ASR eval dumps and emit HTML that ONLY shows positions where ref,
baseline-hyp, and target-hyp do not all agree.

Strategy:
  1. For each side's `word_align`, walk it and assign each cell to a
     reference-token index (NULL:hyp cells become "insertions before next ref").
  2. Merge the two sides on ref index (both sides share the same
     `display_form_tx`, so ref-token streams align by position).
  3. For each ref position, compare ref vs baseline-hyp vs target-hyp and
     emit a row only when they disagree, with context dots for agreement runs.
  4. Also emit rows for any insertion (one side inserted something).
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path


def slugify(uid: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", uid).strip("-")
    return s or "utt"


def split_align(cell: str) -> tuple[str, str]:
    if ":" in cell:
        r, h = cell.split(":", 1)
    else:
        r = h = cell
    return r, h


# Detailed formatting buckets (subtypes of the generic "fmt" category).
FMT_BUCKETS = ("punc", "cap", "itn", "others")

# Human-readable names for punctuation / capitalization transition endpoints.
_PUNC_NAMES = {"none": "∅", "comma": ",", "period": ".",
               "question": "?", "exclamation": "!", "unknown": "?"}
_CAP_NAMES = {"lower": "lc", "upper": "UC", "title": "Tc", "mixed": "mC"}


def fmt_subtype(tag: str) -> str:
    """Return a short readable label for a formatting `word_ter_class` tag, e.g.
    `punc_comma_2_none` -> ",→∅", `cap_lower_2_upper` -> "lc→UC",
    `itn_money` -> "money". Returns "" for non-formatting / empty tags.
    """
    if not tag:
        return ""
    if tag.startswith("punc_") or tag.startswith("cap_"):
        names = _PUNC_NAMES if tag.startswith("punc_") else _CAP_NAMES
        body = tag.split("_", 1)[1]
        if "_2_" in body:
            x, y = body.split("_2_", 1)
            return f"{names.get(x, x)}→{names.get(y, y)}"
        return body
    if tag.startswith("itn_"):
        return tag[len("itn_"):]
    if tag == "others" or tag.startswith("others"):
        return "other"
    return ""


def bucket_for(tags: list[str], cell: str) -> str:
    tag = tags[0] if tags else ""
    if not tag:
        return "eq"
    if tag == "relaxation":
        return "relax"
    if tag.startswith("lexical"):
        if ":NULL" in cell:
            return "del"
        if cell.startswith("NULL:"):
            return "ins"
        return "sub"
    # Formatting edits: keep the detailed subtype (punc / cap / itn / others)
    # rather than collapsing them all into a generic "fmt" bucket.
    prefix = tag.split("_", 1)[0]
    if prefix in FMT_BUCKETS:
        return prefix
    return "others"


def index_alignment(align: list[str], classes: list[list[str]]):
    """Return:
      ref_tokens: list[str] of ref tokens in order
      per_ref: dict ref_idx -> (hyp_token_or_None, bucket, subtype)  # paired hyp for that ref slot
      insertions: dict ref_idx -> list[(hyp_token, bucket)]  # inserted before ref_idx
    """
    ref_tokens: list[str] = []
    per_ref: dict[int, tuple[str | None, str, str]] = {}
    insertions: dict[int, list[tuple[str, str]]] = {}
    ref_idx = 0
    for cell, tags in zip(align, classes):
        ref_tok, hyp_tok = split_align(cell)
        b = bucket_for(tags, cell)
        if ref_tok != "NULL":
            ref_tokens.append(ref_tok)
            sub = fmt_subtype(tags[0] if tags else "") if b in FMT_BUCKETS else ""
            per_ref[ref_idx] = (None if hyp_tok == "NULL" else hyp_tok, b, sub)
            ref_idx += 1
        else:
            insertions.setdefault(ref_idx, []).append((hyp_tok, b))
    return ref_tokens, per_ref, insertions


def _dedupe_runs(tokens: list) -> list:
    """Collapse consecutive identical tokens (case-insensitive, ignoring
    surrounding punctuation) so noisy forced-alignment repeats like
    '+the +the +the' render as a single '+the'."""
    out: list = []
    last_key = object()
    for tok in tokens:
        key = (tok or "").strip(".,!?;:\"'()[]{}\u2014\u2013-").lower()
        if key and key == last_key:
            continue
        out.append(tok)
        last_key = key
    return out


def diff_rows(b_idx, t_idx):
    """Yield row dicts describing only positions of disagreement.

    Row schema:
      kind: 'sub' (ref-paired position with disagreement) or 'ins' (insertion-only)
      ref_idx: int
      ref: str | None
      b_hyp / t_hyp: str | None    (None means deletion)
      b_bucket / t_bucket: str
      b_ins / t_ins: list[str]     (insertions before this position)
    """
    b_refs, b_per, b_ins = b_idx
    t_refs, t_per, t_ins = t_idx
    n = min(len(b_refs), len(t_refs))
    rows = []
    last_kept = -1
    for i in range(n):
        ref = b_refs[i]
        b_hyp, b_b, b_sub = b_per.get(i, (None, "eq", ""))
        t_hyp, t_b, t_sub = t_per.get(i, (None, "eq", ""))
        bi = [h for h, _ in b_ins.get(i, [])]
        ti = [h for h, _ in t_ins.get(i, [])]
        bi = _dedupe_runs(bi)
        ti = _dedupe_runs(ti)
        # Only keep positions where baseline and target produced different
        # hypotheses (token, edit class, or insertions). Skip both
        # "all three agree with ref" AND "both models made the same edit".
        if b_hyp == t_hyp and b_b == t_b and bi == ti:
            continue
        # Gap marker if we skipped equal positions
        if last_kept >= 0 and i - last_kept > 1:
            rows.append({"kind": "gap", "skipped": i - last_kept - 1})
        elif last_kept < 0 and i > 0:
            rows.append({"kind": "gap", "skipped": i})
        rows.append({
            "kind": "diff",
            "ref_idx": i,
            "ref": ref,
            "b_hyp": b_hyp, "b_bucket": b_b, "b_ins": bi, "b_sub": b_sub,
            "t_hyp": t_hyp, "t_bucket": t_b, "t_ins": ti, "t_sub": t_sub,
        })
        last_kept = i
    if last_kept >= 0 and last_kept < n - 1:
        rows.append({"kind": "gap", "skipped": n - 1 - last_kept})
    # Trailing insertions past the last ref token
    for side, ins_map, key in (("b", b_ins, "b_ins"), ("t", t_ins, "t_ins")):
        if n in ins_map:
            rows.append({
                "kind": "trailing",
                "side": side,
                "tokens": _dedupe_runs([h for h, _ in ins_map[n]]),
            })
    return rows


def row_category(row: dict) -> str | None:
    """Bucket a diff/trailing row into 'fmt' (punc/cap/itn-only disagreement)
    or 'lexical' (any sub/del/ins involved on either side).
    """
    if row["kind"] == "trailing":
        return "lexical"
    if row["kind"] != "diff":
        return None
    buckets = {row["b_bucket"], row["t_bucket"]}
    if row["b_ins"] or row["t_ins"] or buckets & {"sub", "del", "ins"}:
        return "lexical"
    if buckets & set(FMT_BUCKETS):
        return "fmt"
    # Both sides eq/relax but somehow recorded differently — lump with fmt.
    return "fmt"


def _norm_token(tok) -> str:
    """Normalize a token for lexical comparison: lowercase + strip leading/
    trailing punctuation, so case- or punctuation-only differences are hidden
    on the lexical page.
    """
    if tok is None:
        return ""
    return tok.strip(".,!?;:\"'()[]{}\u2014\u2013-").lower()


def _norm_seq(tokens) -> tuple:
    return tuple(_norm_token(t) for t in tokens if _norm_token(t))


def filter_rows_by_category(rows: list, category: str,
                            b_idx=None, t_idx=None, n_ctx: int = 2,
                            seg_starts=None) -> list:
    """Keep only rows of the given category and regenerate gap markers.

    For category == 'lexical' we additionally drop rows where the two models'
    hypotheses are identical after lowercase + strip-punctuation normalization
    (so pure case / punctuation disagreements never leak into the lexical page).

    `b_idx` / `t_idx` are the `(ref_tokens, per_ref, insertions)` triples; when
    provided, up to `n_ctx` reference words on each side of every kept
    divergence are emitted as muted "ctx" rows so the error is shown in context.
    `seg_starts` (list of ``[ref_idx, seg_num]``) additionally forces up to
    `n_ctx` reference words on each side of every segment boundary to be shown,
    so the words adjacent to a boundary are visible even when no error is near.
    """
    # 1) Select the divergence rows (and trailing) that belong on this page.
    kept_diffs: list = []
    trailing: list = []
    for r in rows:
        if r["kind"] == "gap":
            continue
        if r["kind"] == "trailing":
            if category == "lexical":
                trailing.append(r)
            continue
        if row_category(r) != category:
            continue
        if category == "lexical":
            b_norm = (_norm_token(r["b_hyp"]),) + _norm_seq(r["b_ins"])
            t_norm = (_norm_token(r["t_hyp"]),) + _norm_seq(r["t_ins"])
            if b_norm == t_norm:
                continue
        kept_diffs.append(r)

    if not kept_diffs and not trailing:
        return []

    kept_map = {r["ref_idx"]: r for r in kept_diffs}

    # 2) Without ref-token context, fall back to the old gap-only layout.
    if not b_idx or not t_idx or n_ctx <= 0:
        out: list = []
        prev_idx = -1
        for i in sorted(kept_map):
            if prev_idx >= 0 and i - prev_idx > 1:
                out.append({"kind": "gap", "skipped": i - prev_idx - 1})
            elif prev_idx < 0 and i > 0:
                out.append({"kind": "gap", "skipped": i})
            out.append(kept_map[i])
            prev_idx = i
        out.extend(trailing)
        return out

    # 3) Expand each kept divergence with up to n_ctx context words per side.
    b_refs, b_per, _ = b_idx
    t_refs, _, _ = t_idx
    n = min(len(b_refs), len(t_refs))
    show: set[int] = set()
    for i in kept_map:
        for j in range(i - n_ctx, i + n_ctx + 1):
            if 0 <= j < n:
                show.add(j)
    # Also reveal n_ctx words on each side of every segment boundary so the
    # words adjacent to a boundary are visible even without a nearby error.
    for r, _seg in (seg_starts or []):
        b = int(r)
        for j in range(b - n_ctx, b + n_ctx + 1):
            if 0 <= j < n:
                show.add(j)

    out = []
    prev_idx = -1
    for i in sorted(show):
        if prev_idx >= 0 and i - prev_idx > 1:
            out.append({"kind": "gap", "skipped": i - prev_idx - 1})
        elif prev_idx < 0 and i > 0:
            out.append({"kind": "gap", "skipped": i})
        if i in kept_map:
            out.append(kept_map[i])
        else:
            b_hyp, _, _ = b_per.get(i, (None, "eq", ""))
            out.append({"kind": "ctx", "ref_idx": i, "ref": b_refs[i],
                        "hyp": b_hyp})
        prev_idx = i
    if prev_idx >= 0 and prev_idx < n - 1:
        out.append({"kind": "gap", "skipped": n - 1 - prev_idx})
    out.extend(trailing)
    return out


# ---- HTML ---------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; }
html { overflow-anchor: none; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; margin: 0; }
.sb-toggle { position: absolute; opacity: 0; pointer-events: none; }
.sb-toggle-btn { position: fixed; top: 8px; left: 208px; z-index: 20; cursor: pointer; background: #fff; border: 1px solid #d0d7de; border-radius: 4px; padding: 2px 8px; font-size: 14px; line-height: 1.4; box-shadow: 0 1px 2px rgba(0,0,0,0.08); user-select: none; transition: left 0.08s ease-out; will-change: left; color: #1f2328; }
.sb-toggle-btn:hover { background: #f6f8fa; }
.sb-toggle-btn::before { content: "\\2630  hide"; }
.sb-toggle:checked ~ .sb-toggle-btn { left: 8px; }
.sb-toggle:checked ~ .sb-toggle-btn::before { content: "\\2630  show"; }
.sb-toggle:checked ~ .layout .sidebar { display: none; }
.sb-toggle:checked ~ .layout .main { margin-left: 0; }
.layout { display: block; }
.sidebar { position: fixed; top: 0; left: 0; bottom: 0; z-index: 10; width: 240px; overflow-y: auto; border-right: 1px solid #d0d7de; background: #f6f8fa; padding: 16px 12px; font-size: 12px; box-sizing: border-box; }
.sidebar h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #555; margin: 0 0 8px 0; }
.sidebar .nav { margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid #d0d7de; }
.sidebar .nav a { display: block; padding: 3px 6px; border-radius: 4px; text-decoration: none; color: #0969da; font-size: 11px; }
.sidebar .nav a:hover { background: #eaeef2; }
.sidebar .nav a.current { color: #1f2328; font-weight: 600; background: #eaeef2; }
.sidebar ol { list-style: none; padding: 0; margin: 0; counter-reset: utt; }
.sidebar li { counter-increment: utt; margin: 0; }
.sidebar li a { display: block; padding: 3px 6px; border-radius: 4px; text-decoration: none; color: #1f2328; font-family: ui-monospace, Menlo, monospace; font-size: 11px; word-break: break-all; line-height: 1.35; }
.sidebar li a::before { content: counter(utt) ". "; color: #888; }
.sidebar li a:hover { background: #eaeef2; }
.sidebar .d-pos { color: #b00020; font-weight: 600; margin-left: 4px; }
.sidebar .d-neg { color: #006622; font-weight: 600; margin-left: 4px; }
.sidebar .d-zero { color: #888; margin-left: 4px; }
.main { margin-left: 240px; padding: 24px; max-width: 1400px; min-width: 0; }
h1 { font-size: 20px; }
.meta { color: #666; font-size: 13px; margin-bottom: 14px; line-height: 1.55; }
.legend { font-size: 12px; color: #444; margin-bottom: 18px; }
.legend span { padding: 1px 6px; border-radius: 3px; margin-right: 8px; }
.card { border: 1px solid #d0d7de; border-radius: 8px; padding: 14px 18px; margin-bottom: 22px; background: #fafbfc; }
.card h2 { font-size: 13px; margin: 0 0 10px 0; font-family: ui-monospace, Menlo, monospace; word-break: break-all; }
.stats { font-size: 12px; color: #333; margin-bottom: 10px; }
.stats span { display: inline-block; margin-right: 12px; padding: 2px 8px; background: #eef2f7; border-radius: 4px; }
.delta-pos { color: #b00020; font-weight: 600; }
.delta-neg { color: #006622; font-weight: 600; }
table.diff { border-collapse: separate; border-spacing: 4px 0; font-family: ui-monospace, Menlo, monospace; font-size: 13px; margin-top: 6px; width: 100%; table-layout: fixed; }
table.diff td.cell { border: 1px solid #eaeef2; border-radius: 4px; padding: 3px 6px; vertical-align: middle; text-align: center; white-space: normal; word-break: break-word; }
table.diff td.row-label { background: transparent; color: #555; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; text-align: right; padding: 3px 8px; border: none; width: 70px; }
table.diff td.idx-cell { color: #999; font-size: 10px; padding: 2px 4px; text-align: center; border: none; background: transparent; }
table.diff td.gap-cell { color: #888; font-style: italic; font-size: 11px; background: #f9fafb; border: 1px dashed #d8dde3; }
table.diff td.diff-col { background: #fffbf0; }
table.diff td.both-same { background: #f0f7ff; }
table.diff tr.ref td.cell { background: #ffffff; font-weight: 600; color: #1f2328; }
table.diff tr.ref td.cell.diff-col { background: #f5f9ff; }
table.diff tr.ref td.cell.both-same { background: #f5f9ff; }
.row-block { margin-bottom: 16px; }
.tok-eq    { color: #1f2328; }
.tok-sub   { background: #fff3c4; color: #6a4a00; padding: 0 3px; border-radius: 3px; }
.tok-del   { background: #ffd7d5; color: #82071e; text-decoration: line-through; padding: 0 3px; border-radius: 3px; }
.tok-ins   { background: #d1f4d4; color: #0a5f1e; padding: 0 3px; border-radius: 3px; }
.tok-fmt   { background: #e0e7ff; color: #2c2e7a; padding: 0 3px; border-radius: 3px; }
.tok-punc  { background: #e0e7ff; color: #2c2e7a; padding: 0 3px; border-radius: 3px; }
.tok-cap   { background: #d6f0f5; color: #0b5566; padding: 0 3px; border-radius: 3px; }
.tok-itn   { background: #fde2f3; color: #8a1d63; padding: 0 3px; border-radius: 3px; }
.tok-others{ background: #ececec; color: #444; padding: 0 3px; border-radius: 3px; }
.tok-relax { background: #f0e6ff; color: #5a2a99; padding: 0 3px; border-radius: 3px; }
.none      { color: #b88; font-style: italic; }
.fmt-sub   { display: block; font-size: 9px; line-height: 1.2; margin-top: 2px; color: #555; font-family: ui-monospace, Menlo, monospace; letter-spacing: -0.02em; }
table.diff td.ctx-cell { background: #fcfcfd; color: #99a; }
table.diff tr.ref td.cell.ctx-cell { background: #fcfcfd; color: #8a909a; font-weight: 500; }
.ctx-ditto { color: #c2c8d0; }
table.diff td.seg-cell { background: #fff7ed; border: none; padding: 0 2px; width: 1px; }
table.diff tr.ref td.cell.seg-cell { background: #fff7ed; }
.seg-bar { display: inline-block; color: #d97706; font-weight: 700; }
.seg-num { display: inline-block; font-size: 9px; font-weight: 700; color: #b45309; font-family: ui-monospace, Menlo, monospace; }
.ins-chip  { display: inline-block; margin-right: 2px; }
.ent-block { margin-top: 10px; padding: 8px 10px; border: 1px solid #d8dde3; border-radius: 5px; background: #fff; }
.ent-label { font-size: 11px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.ent-ali { font-family: ui-monospace, Menlo, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-word; margin: 0 0 6px 0; color: #1f2328; }
.ent-meta { font-size: 11px; color: #555; }
table.toc { border-collapse: collapse; font-size: 12px; margin: 4px 0 22px 0; width: 100%; }
table.toc th, table.toc td { border: 1px solid #d8dde3; padding: 4px 8px; text-align: left; }
table.toc th { background: #eef2f7; font-weight: 600; }
table.toc td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, monospace; }
table.toc td.uid { font-family: ui-monospace, Menlo, monospace; word-break: break-all; }
table.toc a { text-decoration: none; color: #0969da; }
table.toc a:hover { text-decoration: underline; }
details.toc-wrap { margin-bottom: 18px; }
details.toc-wrap > summary { cursor: pointer; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
"""


# Tiny self-contained script: keep the utterance under the viewport top fixed
# in place when the sidebar is toggled. The CSS checkbox still does the actual
# show/hide, so the page degrades gracefully (just with a scroll jump) if JS is
# disabled. We continuously track the first card at/below the viewport top and,
# on toggle, re-pin it to the same vertical offset after the width reflow.
TOGGLE_JS = """
(function () {
  var cb = document.getElementById('sb-toggle');
  if (!cb) return;
  var anchor = null, anchorTop = 0, ticking = false;
  function capture() {
    var cards = document.getElementsByClassName('card');
    for (var i = 0; i < cards.length; i++) {
      var r = cards[i].getBoundingClientRect();
      if (r.bottom > 0) { anchor = cards[i]; anchorTop = r.top; return; }
    }
    anchor = null;
  }
  window.addEventListener('scroll', function () {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(function () { capture(); ticking = false; });
    }
  }, { passive: true });
  cb.addEventListener('change', function () {
    if (!anchor) capture();
    if (!anchor) return;
    var a = anchor, top = anchorTop;
    requestAnimationFrame(function () {
      var nt = a.getBoundingClientRect().top;
      window.scrollBy(0, nt - top);
      capture();
    });
  });
  capture();
})();
"""


def render_token(tok: str | None, bucket: str, sub: str = "") -> str:
    if tok is None:
        base = '<span class="none">∅</span>'
    else:
        base = f'<span class="tok-{bucket}">{html.escape(tok)}</span>'
    if sub:
        base += f'<span class="fmt-sub">{html.escape(sub)}</span>'
    return base


def render_ins_list(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return " ".join(
        f'<span class="ins-chip tok-ins">+{html.escape(t)}</span>' for t in tokens
    )


def render_diff_table(rows: list, baseline_name: str, target_name: str,
                      cols_per_row: int = 12, seg_starts=None) -> str:
    """Row-wise layout that wraps: each visual block has 3 rows
    (ref / baseline / target), and we emit a new block every `cols_per_row`
    disagreement positions. No horizontal overflow.

    When `seg_starts` (list of ``[ref_idx, seg_num]``) is supplied, a thin
    segment-boundary marker column is injected before the reference token that
    begins each segment, so per-segment response boundaries are visible.
    """
    if not rows:
        return '<div class="meta">No disagreements — ref, baseline, and target all match.</div>'

    boundaries = sorted(([int(r), int(n)] for r, n in (seg_starts or [])),
                        key=lambda x: x[0])
    _bi = [0]

    cols = []

    def _flush_segs(upto_ref):
        while _bi[0] < len(boundaries) and boundaries[_bi[0]][0] <= upto_ref:
            n = boundaries[_bi[0]][1]
            cols.append({
                "idx": f'<span class="seg-num">S{n}</span>',
                "ref": '<span class="seg-bar">┊</span>',
                "b": '<span class="seg-bar">┊</span>',
                "t": '<span class="seg-bar">┊</span>',
                "cls": "seg-cell",
            })
            _bi[0] += 1

    for row in rows:
        if row["kind"] in ("diff", "ctx"):
            _flush_segs(row["ref_idx"])
        if row["kind"] == "gap":
            cols.append({
                "idx": "…",
                "ref": f'… {row["skipped"]} …',
                "b": "",
                "t": "",
                "cls": "gap-cell",
            })
            continue
        if row["kind"] == "trailing":
            tokens_html = render_ins_list(row["tokens"])
            cols.append({
                "idx": "end",
                "ref": '<span class="none">∅</span>',
                "b": tokens_html if row["side"] == "b" else "",
                "t": tokens_html if row["side"] == "t" else "",
                "cls": "diff-col",
            })
            continue
        if row["kind"] == "ctx":
            ditto = '<span class="ctx-ditto">·</span>'
            cols.append({
                "idx": str(row["ref_idx"]),
                "ref": html.escape(row["ref"]),
                "b": ditto,
                "t": ditto,
                "cls": "ctx-cell",
            })
            continue
        both_same = (row["b_hyp"] == row["t_hyp"]
                     and row["b_bucket"] == row["t_bucket"]
                     and row["b_ins"] == row["t_ins"])
        cls = "both-same" if both_same else "diff-col"
        b_cell = (render_ins_list(row["b_ins"]) + (" " if row["b_ins"] else "")
                  + render_token(row["b_hyp"], row["b_bucket"], row.get("b_sub", "")))
        t_cell = (render_ins_list(row["t_ins"]) + (" " if row["t_ins"] else "")
                  + render_token(row["t_hyp"], row["t_bucket"], row.get("t_sub", "")))
        cols.append({
            "idx": str(row["ref_idx"]),
            "ref": html.escape(row["ref"]),
            "b": b_cell,
            "t": t_cell,
            "cls": cls,
        })

    _flush_segs(float("inf"))

    def chunk(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    blocks = []
    for group in chunk(cols, cols_per_row):
        idx_row = ['<td class="row-label">#</td>'] + [
            f'<td class="idx-cell">{c["idx"]}</td>' for c in group
        ]
        ref_row = ['<td class="row-label">ref</td>'] + [
            f'<td class="cell {c["cls"]}">{c["ref"]}</td>' for c in group
        ]
        b_row = [f'<td class="row-label">{html.escape(baseline_name)}</td>'] + [
            f'<td class="cell {c["cls"]}">{c["b"]}</td>' for c in group
        ]
        t_row = [f'<td class="row-label">{html.escape(target_name)}</td>'] + [
            f'<td class="cell {c["cls"]}">{c["t"]}</td>' for c in group
        ]
        blocks.append(
            '<table class="diff row-block"><tbody>'
            + '<tr>' + ''.join(idx_row) + '</tr>'
            + '<tr class="ref">' + ''.join(ref_row) + '</tr>'
            + '<tr>' + ''.join(b_row) + '</tr>'
            + '<tr>' + ''.join(t_row) + '</tr>'
            + '</tbody></table>'
        )
    return ''.join(blocks)


def _iter_raw_records(path: Path):
    """Yield raw dict records from either an in-house JSON list dump or a verl
    long-eval `details.jsonl` (one JSON object per line). Detection is by the
    first non-whitespace character: `[` -> JSON array, otherwise line-delimited
    JSON. A single top-level JSON object is also accepted.
    """
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        for rec in data:
            yield rec
        return
    if stripped.startswith("{") and path.suffix.lower() != ".jsonl":
        # Could be a single object or JSONL with a leading `{`; try whole-file
        # parse first, fall back to line-by-line.
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                yield data
                return
            for rec in data:
                yield rec
            return
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _is_verl_long_eval(rec: dict) -> bool:
    """True for verl `main_long_eval_asr` per-recording records, which carry a
    `dter_detail` block instead of `UtteranceTERMetrics`."""
    return "dter_detail" in rec and "UtteranceTERMetrics" not in rec


def _verl_uid(rec: dict) -> str:
    """Recording id for a verl long-eval record. Prefer the audio GUID embedded
    in `parent_audio_path` (e.g. `.../wav/<guid>_0.wav`), which matches the
    in-house `UtteranceId`. Fall back to `id` or the raw parent path."""
    p = rec.get("parent_audio_path") or ""
    m = re.search(r"/([0-9a-fA-F][0-9a-fA-F-]{7,})(?:_\d+)?\.wav", p)
    if m:
        return m.group(1)
    return rec.get("id") or p or ""


def _segment_ref_starts(align: list, segments) -> list:
    """Map verl per-segment responses onto reference-token positions.

    The long-eval `hyp` is the space-join of each segment's text, so segment
    boundaries fall at known offsets in the hypothesis token stream. We walk
    `word_align`, accumulating consumed hypothesis characters, and record the
    reference-token index at which each new segment begins.

    `segments` is the per-segment text list — either the `responses` field
    (plain segment strings) or the legacy `raw_response` field (each item an
    `<ASR>...<TXT>...</TXT></ASR>` wrapper). When a `<TXT>...</TXT>` block is
    present it is extracted; otherwise the string is used as-is.

    Returns a list of ``[ref_idx, seg_num]`` pairs (``seg_num`` 1-based) meaning
    "segment ``seg_num`` starts immediately before reference token ``ref_idx``".
    Returns ``[]`` when there are fewer than two segments.
    """
    segs = []
    for s in segments or []:
        s = s or ""
        m = re.search(r"<TXT>(.*?)</TXT>", s, re.S)
        text = m.group(1) if m else s
        segs.append(re.sub(r"\s+", "", text))
    if len(segs) <= 1:
        return []
    seg_char_ends = []
    acc = 0
    for s in segs:
        acc += len(s)
        seg_char_ends.append(acc)
    consumed = 0
    seg_idx = 0
    ref_count = 0
    starts = []
    for cell in align:
        r, h = split_align(cell)
        if h != "NULL":
            consumed += len(re.sub(r"\s+", "", h))
        if r != "NULL":
            ref_count += 1
        while seg_idx < len(seg_char_ends) - 1 and consumed >= seg_char_ends[seg_idx]:
            starts.append([ref_count, seg_idx + 1])
            seg_idx += 1
    return starts


def _verl_to_inhouse(rec: dict, metric: str) -> dict:
    """Convert one verl long-eval record into an in-house-style record carrying a
    single `UtteranceTERMetrics` entry (named `metric`) built from `dter_detail`.
    The long-eval pipeline only computes a disfluency-tolerant TER, so the same
    detail is surfaced regardless of the requested `metric` name."""
    dd = rec.get("dter_detail") or {}
    align = dd.get("word_align") or []
    refs, hyps = [], []
    for cell in align:
        if ":" in cell:
            r, h = cell.split(":", 1)
        else:
            r = h = cell
        if r != "NULL":
            refs.append(r)
        if h != "NULL":
            hyps.append(h)
    return {
        "UtteranceId": _verl_uid(rec),
        "DataSetID": rec.get("data_source"),
        "_seg_starts": _segment_ref_starts(
            align, rec.get("responses") or rec.get("raw_response")),
        "UtteranceTERMetrics": [{
            "MetricName": metric,
            "ter_info": {
                "number_of_tokens": int(rec.get("dter_n_ref") or 0),
                "number_of_edits": int(rec.get("dter_n_err") or 0),
                "display_ter": float(rec.get("dter") or 0.0) * 100.0,
            },
            "display_form_tx": " ".join(refs),
            "display_form_hyp": " ".join(hyps),
            "word_align": align,
            "word_ter_class": dd.get("word_ter_class") or [],
            "ter_category_info": dd.get("ter_category_info") or {},
        }],
    }


def load_entity_info(path: Path, entity_metric: str) -> dict:
    """Return {uid: entity_dict} extracted from top-level Metrics[*].EntityInfo
    matching MetricName == entity_metric. Records without that metric, or with
    empty EntityInfo, get an all-zero entry so deltas are well-defined.

    Verl long-eval `details.jsonl` records carry no rich EntityInfo block, so
    they yield all-zero entity entries (entity comparison is a no-op for them).
    """
    out = {}
    for rec in _iter_raw_records(path):
        if _is_verl_long_eval(rec):
            uid = _verl_uid(rec)
            if uid:
                out[uid] = {
                    "n_trans_ents": 0, "n_reco_ents": 0,
                    "n_trans_matched": 0, "n_reco_matched": 0,
                    "n_ent_words": 0, "n_ent_ins": 0, "n_ent_del": 0,
                    "n_ent_sub": 0, "n_ent_edits": 0,
                    "eer": -1.0, "ewer": -1.0, "recall": -1.0, "precision": -1.0,
                    "alignment_str": "", "trans_uniq": {}, "reco_uniq": {},
                }
            continue
        uid = rec.get("UtteranceId")
        if not uid:
            continue
        ei = None
        for m in rec.get("Metrics") or []:
            if m.get("MetricName") == entity_metric:
                ei = m.get("EntityInfo") or {}
                break
        ei = ei or {}

        def _f(k, default=0.0):
            v = ei.get(k)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        def _i(k, default=0):
            v = ei.get(k)
            try:
                return int(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        out[uid] = {
            "n_trans_ents": _i("NumTransEnts"),
            "n_reco_ents": _i("NumRecoEnts"),
            "n_trans_matched": _i("NumTransEntsMatched"),
            "n_reco_matched": _i("NumRecoEntsMatched"),
            "n_ent_words": _i("NumEntWords"),
            "n_ent_ins": _i("NumEntWordIns"),
            "n_ent_del": _i("NumEntWordDel"),
            "n_ent_sub": _i("NumEntWordSub"),
            "n_ent_edits": _i("NumEntEdits"),
            "eer": _f("EntityErrorRate", -1.0),
            "ewer": _f("EntityWordErrorRate", -1.0),
            "recall": _f("EntityRecall", -1.0),
            "precision": _f("EntityPrecision", -1.0),
            "alignment_str": ei.get("EntityAlignmentAsString") or "",
            "trans_uniq": ei.get("ListTransUniqEnts") or {},
            "reco_uniq": ei.get("ListRecoUniqEnts") or {},
        }
    return out


def load_records(path: Path, metric: str) -> dict:
    out = {}
    for rec in _iter_raw_records(path):
        if _is_verl_long_eval(rec):
            rec = _verl_to_inhouse(rec, metric)
        uid = rec.get("UtteranceId")
        chosen = None
        for m in rec.get("UtteranceTERMetrics") or []:
            if m.get("MetricName") == metric:
                chosen = m
                break
        if not uid or chosen is None:
            continue
        info = chosen.get("ter_info") or {}
        cats = (chosen.get("ter_category_info") or {}).get("ter_categories") or {}
        cat_edits = {k: int(v.get("number_of_edits") or 0) for k, v in cats.items()}
        # Decompose lexical into sub/ins/del by walking word_align with class.
        align = chosen.get("word_align") or []
        classes = chosen.get("word_ter_class") or []
        n_sub = n_ins = n_del = 0
        for cell, tags in zip(align, classes):
            tag = tags[0] if tags else ""
            if not tag.startswith("lexical"):
                continue
            if ":NULL" in cell:
                n_del += 1
            elif cell.startswith("NULL:"):
                n_ins += 1
            else:
                n_sub += 1
        out[uid] = {
            "align": align,
            "classes": classes,
            "tokens": int(info.get("number_of_tokens") or 0),
            "edits": int(info.get("number_of_edits") or 0),
            "ter": float(info.get("display_ter") or 0.0),
            "cat_edits": cat_edits,
            "fmt_edits": cat_edits.get("punc", 0) + cat_edits.get("cap", 0) + cat_edits.get("itn", 0),
            "lex_edits": cat_edits.get("lexical", 0),
            "lex_sub": n_sub, "lex_ins": n_ins, "lex_del": n_del,
            "punc_edits": cat_edits.get("punc", 0),
            "cap_edits": cat_edits.get("cap", 0),
            "itn_edits": cat_edits.get("itn", 0),
            "seg_starts": rec.get("_seg_starts") or [],
            "entity": None,
        }
    return out


def card_html(uid: str, b: dict, t: dict, baseline_name: str, target_name: str,
              category: str | None = None) -> str | None:
    if category == "entity":
        return entity_card_html(uid, b, t, baseline_name, target_name)
    b_idx = index_alignment(b["align"], b["classes"])
    t_idx = index_alignment(t["align"], t["classes"])
    if len(b_idx[0]) != len(t_idx[0]):
        warn = (f'<div class="meta" style="color:#b00020">'
                f'⚠ ref-token counts differ: baseline={len(b_idx[0])} '
                f'target={len(t_idx[0])} — showing min length.</div>')
    else:
        warn = ""
    rows = diff_rows(b_idx, t_idx)
    seg_starts = b.get("seg_starts") if category in (None, "fmt", "lexical") else None
    if category is not None:
        rows = filter_rows_by_category(rows, category, b_idx, t_idx,
                                       seg_starts=seg_starts)
    n_diff = sum(1 for r in rows if r["kind"] in ("diff", "trailing"))
    if n_diff == 0:
        return None
    delta = t["ter"] - b["ter"]
    sign = "+" if delta > 0 else ""
    delta_cls = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "")

    def cat_chip(label: str, b_n: int, t_n: int) -> str:
        d = t_n - b_n
        sgn = "+" if d > 0 else ""
        cls = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "")
        return (f'<span>{label}: {b_n} → {t_n} '
                f'(<span class="{cls}">{sgn}{d}</span>)</span>')

    def compound_chip(label: str, parts: list[tuple[str, int, int]]) -> str:
        """parts: list of (sub_label, baseline_val, target_val)."""
        inner = []
        for sub, bn, tn in parts:
            d = tn - bn
            sgn = "+" if d > 0 else ""
            cls = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "")
            inner.append(f'{sub}={bn}→{tn} <span class="{cls}">({sgn}{d})</span>')
        return f'<span>{label}: ' + " | ".join(inner) + '</span>'

    parts = [
        f'<span>{baseline_name} TER: <b>{b["ter"]:.2f}%</b> '
        f'({b["edits"]}/{b["tokens"]})</span>',
        f'<span>{target_name} TER: <b>{t["ter"]:.2f}%</b> '
        f'({t["edits"]}/{t["tokens"]})</span>',
        f'<span class="{delta_cls}">ΔTER: {sign}{delta:.2f}pp</span>',
        f'<span class="{delta_cls}">Δedits: {sign}{t["edits"] - b["edits"]}</span>',
    ]
    if category in (None, "lexical"):
        parts.append(cat_chip("lexical", b["lex_edits"], t["lex_edits"]))
        parts.append(compound_chip("lex breakdown", [
            ("sub", b["lex_sub"], t["lex_sub"]),
            ("ins", b["lex_ins"], t["lex_ins"]),
            ("del", b["lex_del"], t["lex_del"]),
        ]))
    if category in (None, "fmt"):
        parts.append(cat_chip("fmt", b["fmt_edits"], t["fmt_edits"]))
        parts.append(compound_chip("fmt breakdown", [
            ("punc", b["punc_edits"], t["punc_edits"]),
            ("cap", b["cap_edits"], t["cap_edits"]),
            ("itn", b["itn_edits"], t["itn_edits"]),
        ]))
    parts.append(f'<span>disagreement cells: {n_diff}</span>')
    stats = "".join(parts)
    return (
        f'<div class="card">'
        f'<h2 id="{html.escape(slugify(uid))}">{html.escape(uid)}</h2>'
        f'<div class="stats">{stats}</div>'
        f'{warn}'
        f'{render_diff_table(rows, baseline_name, target_name, seg_starts=seg_starts)}'
        f'</div>'
    )


def _delta_chip(label: str, b_v, t_v, fmt="{:+d}", val_fmt="{}") -> str:
    d = t_v - b_v
    cls = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "")
    sgn = fmt.format(d) if isinstance(d, int) else fmt.format(d)
    return (f'<span>{label}: {val_fmt.format(b_v)} → {val_fmt.format(t_v)} '
            f'(<span class="{cls}">{sgn}</span>)</span>')


def entity_card_html(uid: str, b: dict, t: dict, baseline_name: str, target_name: str) -> str | None:
    be = b.get("entity")
    te = t.get("entity")
    if not be or not te:
        return None
    # Skip utterances with no entities on either side.
    if be["n_trans_ents"] == 0 and te["n_trans_ents"] == 0 \
            and be["n_reco_ents"] == 0 and te["n_reco_ents"] == 0:
        return None
    parts = [
        _delta_chip("ent edits", be["n_ent_edits"], te["n_ent_edits"]),
        _delta_chip("ent words", be["n_ent_words"], te["n_ent_words"]),
        _delta_chip("ent sub", be["n_ent_sub"], te["n_ent_sub"]),
        _delta_chip("ent ins", be["n_ent_ins"], te["n_ent_ins"]),
        _delta_chip("ent del", be["n_ent_del"], te["n_ent_del"]),
        _delta_chip("EWER", be["ewer"], te["ewer"], fmt="{:+.2f}pp", val_fmt="{:.2f}%"),
        _delta_chip("EER", be["eer"], te["eer"], fmt="{:+.2f}pp", val_fmt="{:.2f}%"),
        _delta_chip("recall", be["recall"], te["recall"], fmt="{:+.2f}pp", val_fmt="{:.2f}%"),
        _delta_chip("precision", be["precision"], te["precision"], fmt="{:+.2f}pp", val_fmt="{:.2f}%"),
        f'<span>trans ents: {be["n_trans_ents"]} → {te["n_trans_ents"]}</span>',
        f'<span>reco ents: {be["n_reco_ents"]} → {te["n_reco_ents"]}</span>',
    ]
    stats = "".join(parts)

    def _ent_list(d: dict) -> str:
        if not d:
            return '<span class="none">∅</span>'
        items = ", ".join(f'{html.escape(str(k))}×{v}' for k, v in d.items())
        return html.escape(items) if False else items  # already escaped above

    def _ali(label: str, ent: dict) -> str:
        ali = ent["alignment_str"] or "(empty)"
        return (f'<div class="ent-block"><div class="ent-label">{html.escape(label)}</div>'
                f'<pre class="ent-ali">{html.escape(ali)}</pre>'
                f'<div class="ent-meta">trans entities: {_ent_list(ent["trans_uniq"])} '
                f'· reco entities: {_ent_list(ent["reco_uniq"])}</div></div>')

    body = _ali(baseline_name, be) + _ali(target_name, te)
    return (
        f'<div class="card">'
        f'<h2 id="{html.escape(slugify(uid))}">{html.escape(uid)}</h2>'
        f'<div class="stats">{stats}</div>'
        f'{body}'
        f'</div>'
    )


def _render_entity_toc(entries: list) -> str:
    def cls(v):
        return "delta-pos" if v > 0 else ("delta-neg" if v < 0 else "")

    def num(v, fmt="{:+d}"):
        return fmt.format(v)

    rows = []
    for i, e in enumerate(entries, 1):
        be, te = e["b"]["entity"], e["t"]["entity"]
        d_edits = te["n_ent_edits"] - be["n_ent_edits"]
        d_words = te["n_ent_words"] - be["n_ent_words"]
        d_ewer = te["ewer"] - be["ewer"]
        d_eer = te["eer"] - be["eer"]
        d_recall = te["recall"] - be["recall"]
        d_prec = te["precision"] - be["precision"]
        anchor = slugify(e["utt_id"])
        rows.append(
            f'<tr>'
            f'<td class="num">{i}</td>'
            f'<td class="uid"><a href="#{html.escape(anchor)}">{html.escape(e["utt_id"])}</a></td>'
            f'<td class="num {cls(d_edits)}">{num(d_edits)}</td>'
            f'<td class="num {cls(d_words)}">{num(d_words)}</td>'
            f'<td class="num {cls(d_ewer)}">{d_ewer:+.2f}pp</td>'
            f'<td class="num {cls(d_eer)}">{d_eer:+.2f}pp</td>'
            f'<td class="num {cls(d_recall)}">{d_recall:+.2f}pp</td>'
            f'<td class="num {cls(d_prec)}">{d_prec:+.2f}pp</td>'
            f'<td class="num">{be["n_trans_ents"]}/{te["n_trans_ents"]}</td>'
            f'<td class="num">{be["n_reco_ents"]}/{te["n_reco_ents"]}</td>'
            f'</tr>'
        )

    tot_b_edits = sum(e["b"]["entity"]["n_ent_edits"] for e in entries)
    tot_t_edits = sum(e["t"]["entity"]["n_ent_edits"] for e in entries)
    tot_b_words = sum(e["b"]["entity"]["n_ent_words"] for e in entries)
    tot_t_words = sum(e["t"]["entity"]["n_ent_words"] for e in entries)
    tot_b_trans = sum(e["b"]["entity"]["n_trans_ents"] for e in entries)
    tot_t_trans = sum(e["t"]["entity"]["n_trans_ents"] for e in entries)
    tot_b_match = sum(e["b"]["entity"]["n_trans_matched"] for e in entries)
    tot_t_match = sum(e["t"]["entity"]["n_trans_matched"] for e in entries)
    tot_b_reco = sum(e["b"]["entity"]["n_reco_ents"] for e in entries)
    tot_t_reco = sum(e["t"]["entity"]["n_reco_ents"] for e in entries)
    tot_b_reco_match = sum(e["b"]["entity"]["n_reco_matched"] for e in entries)
    tot_t_reco_match = sum(e["t"]["entity"]["n_reco_matched"] for e in entries)
    b_ewer = (tot_b_edits / tot_b_words * 100.0) if tot_b_words else 0.0
    t_ewer = (tot_t_edits / tot_t_words * 100.0) if tot_t_words else 0.0
    b_eer = ((tot_b_trans - tot_b_match) / tot_b_trans * 100.0) if tot_b_trans else 0.0
    t_eer = ((tot_t_trans - tot_t_match) / tot_t_trans * 100.0) if tot_t_trans else 0.0
    b_recall = (tot_b_match / tot_b_trans * 100.0) if tot_b_trans else 0.0
    t_recall = (tot_t_match / tot_t_trans * 100.0) if tot_t_trans else 0.0
    b_prec = (tot_b_reco_match / tot_b_reco * 100.0) if tot_b_reco else 0.0
    t_prec = (tot_t_reco_match / tot_t_reco * 100.0) if tot_t_reco else 0.0
    d_edits = tot_t_edits - tot_b_edits
    d_words = tot_t_words - tot_b_words
    totals_row = (
        f'<tr style="background:#eef2f7;font-weight:600">'
        f'<td class="num">Σ</td>'
        f'<td class="uid">overall ({len(entries)} utts)</td>'
        f'<td class="num {cls(d_edits)}">{num(d_edits)}</td>'
        f'<td class="num {cls(d_words)}">{num(d_words)}</td>'
        f'<td class="num {cls(t_ewer - b_ewer)}">{t_ewer - b_ewer:+.2f}pp '
        f'<small>({b_ewer:.2f}→{t_ewer:.2f})</small></td>'
        f'<td class="num {cls(t_eer - b_eer)}">{t_eer - b_eer:+.2f}pp '
        f'<small>({b_eer:.2f}→{t_eer:.2f})</small></td>'
        f'<td class="num {cls(t_recall - b_recall)}">{t_recall - b_recall:+.2f}pp '
        f'<small>({b_recall:.2f}→{t_recall:.2f})</small></td>'
        f'<td class="num {cls(t_prec - b_prec)}">{t_prec - b_prec:+.2f}pp '
        f'<small>({b_prec:.2f}→{t_prec:.2f})</small></td>'
        f'<td class="num">{tot_b_trans}/{tot_t_trans}</td>'
        f'<td class="num">{tot_b_reco}/{tot_t_reco}</td>'
        f'</tr>'
    )
    return (
        f'<details class="toc-wrap" open>'
        f'<summary>Utterances ({len(entries)}) — click row to jump</summary>'
        f'<table class="toc"><thead><tr>'
        f'<th>#</th><th>UtteranceId</th>'
        f'<th>Δ ent edits</th><th>Δ ent words</th>'
        f'<th>Δ EWER</th><th>Δ EER</th>'
        f'<th>Δ recall</th><th>Δ precision</th>'
        f'<th>trans (B/T)</th><th>reco (B/T)</th>'
        f'</tr></thead><tbody>'
        + totals_row
        + "".join(rows)
        + '</tbody></table></details>'
    )


def render_toc(entries: list, cat_key: str) -> str:
    """entries: list of dicts with utt_id, cat_delta, ter_delta, edits_delta, b, t."""
    if not entries:
        return ""
    if cat_key == "entity":
        return _render_entity_toc(entries)
    show_total_edit_col = cat_key != "overall"
    if cat_key == "lexical":
        cat_label = "Δ lex edits"
        sub_keys = [("sub", "lex_sub"), ("ins", "lex_ins"), ("del", "lex_del")]
    elif cat_key == "fmt":
        cat_label = "Δ fmt edits"
        sub_keys = [("punc", "punc_edits"), ("cap", "cap_edits"), ("itn", "itn_edits")]
    else:
        cat_label = "Δ edits"
        sub_keys = [("lex", "lex_edits"), ("fmt", "fmt_edits")]

    def cls(v):
        return "delta-pos" if v > 0 else ("delta-neg" if v < 0 else "")

    def sgn(v, fmt="{:+d}"):
        return fmt.format(v)

    rows = []
    for i, e in enumerate(entries, 1):
        uid = e["utt_id"]
        anchor = slugify(uid)
        cd = e["cat_delta"]
        td = e["ter_delta"]
        ed = e["edits_delta"]
        sub_cells = "".join(
            f'<td class="num {cls(e["t"][k] - e["b"][k])}">'
            f'{sgn(e["t"][k] - e["b"][k])}</td>'
            for _, k in sub_keys
        )
        edit_cell = (
            f'<td class="num {cls(ed)}">{sgn(ed)}</td>'
            if show_total_edit_col else ""
        )
        rows.append(
            f'<tr>'
            f'<td class="num">{i}</td>'
            f'<td class="uid"><a href="#{html.escape(anchor)}">{html.escape(uid)}</a></td>'
            f'<td class="num {cls(cd)}">{sgn(cd)}</td>'
            + sub_cells
            + edit_cell
            + f'<td class="num {cls(td)}">{td:+.2f}pp</td>'
            f'<td class="num">{e["b"]["ter"]:.2f}%</td>'
            f'<td class="num">{e["t"]["ter"]:.2f}%</td>'
            f'</tr>'
        )
    tot_cd = sum(e["cat_delta"] for e in entries)
    tot_ed = sum(e["edits_delta"] for e in entries)
    tot_b_edits = sum(e["b"]["edits"] for e in entries)
    tot_t_edits = sum(e["t"]["edits"] for e in entries)
    tot_b_tokens = sum(e["b"]["tokens"] for e in entries)
    tot_t_tokens = sum(e["t"]["tokens"] for e in entries)
    b_ter_overall = (tot_b_edits / tot_b_tokens * 100.0) if tot_b_tokens else 0.0
    t_ter_overall = (tot_t_edits / tot_t_tokens * 100.0) if tot_t_tokens else 0.0
    ter_delta_overall = t_ter_overall - b_ter_overall
    tot_sub_cells = "".join(
        f'<td class="num {cls(sum(e["t"][k] - e["b"][k] for e in entries))}">'
        f'{sgn(sum(e["t"][k] - e["b"][k] for e in entries))}</td>'
        for _, k in sub_keys
    )
    totals_row = (
        f'<tr style="background:#eef2f7;font-weight:600">'
        f'<td class="num">Σ</td>'
        f'<td class="uid">overall ({len(entries)} utts)</td>'
        f'<td class="num {cls(tot_cd)}">{tot_cd:+d}</td>'
        + tot_sub_cells
        + (f'<td class="num {cls(tot_ed)}">{tot_ed:+d}</td>' if show_total_edit_col else "")
        + f'<td class="num {cls(ter_delta_overall)}">{ter_delta_overall:+.2f}pp</td>'
        f'<td class="num">{b_ter_overall:.2f}%</td>'
        f'<td class="num">{t_ter_overall:.2f}%</td>'
        f'</tr>'
    )
    sub_headers = "".join(f'<th>Δ {lbl}</th>' for lbl, _ in sub_keys)
    edit_header = '<th>Δ edits</th>' if show_total_edit_col else ''
    return (
        f'<details class="toc-wrap" open>'
        f'<summary>Utterances ({len(entries)}) — click row to jump</summary>'
        f'<table class="toc"><thead><tr>'
        f'<th>#</th><th>UtteranceId</th><th>{cat_label}</th>'
        + sub_headers
        + edit_header
        + '<th>Δ TER</th>'
        '<th>baseline TER</th><th>target TER</th>'
        '</tr></thead>'
        '<tbody>'
        + totals_row
        + "".join(rows)
        + '</tbody></table></details>'
    )


def render_sidebar(entries: list, cat_key: str, nav_html: str = "") -> str:
    if not entries and not nav_html:
        return ""
    items = []
    for e in entries:
        uid = e["utt_id"]
        anchor = slugify(uid)
        cd = e["cat_delta"]
        cls = "d-pos" if cd > 0 else ("d-neg" if cd < 0 else "d-zero")
        sign = "+" if cd > 0 else ""
        items.append(
            f'<li><a href="#{html.escape(anchor)}">{html.escape(uid)}'
            f'<span class="{cls}">{sign}{cd}</span></a></li>'
        )
    label = {
        "overall": "Δ edits",
        "lexical": "Δ lex",
        "fmt": "Δ fmt",
        "entity": "Δ ent edits",
    }.get(cat_key, "Δ")
    body = (
        f'<h3>Utterances · {label}</h3><ol>{"".join(items)}</ol>'
        if entries else ""
    )
    return f'<aside class="sidebar">{nav_html}{body}</aside>'


def render_page(title: str, items: list, baseline_name: str, target_name: str, summary: dict, toc_html: str = "", sidebar_html: str = "") -> str:
    legend = (
        '<div class="legend">'
        '<span class="tok-sub">substitution</span>'
        '<span class="tok-del">deletion</span>'
        '<span class="tok-ins">insertion</span>'
        '<span class="tok-punc">punc</span>'
        '<span class="tok-cap">cap</span>'
        '<span class="tok-itn">itn</span>'
        '<span class="tok-others">other fmt</span>'
        '<span class="tok-relax">relaxation (forgiven)</span>'
        '<span class="none">∅ = absent / deleted</span>'
        '<br>Rows shaded blue: baseline and target produced the same (non-ref) output. '
        'Rows shaded amber: baseline and target diverged.'
        '</div>'
    )
    head = (
        f'<h1>{html.escape(title)}</h1>'
        f'<div class="meta">'
        f'metric = <code>{html.escape(summary["metric"])}</code> &nbsp;|&nbsp; '
        f'baseline = <code>{html.escape(baseline_name)}</code> &nbsp;|&nbsp; '
        f'target = <code>{html.escape(target_name)}</code><br>'
        f'overall: baseline TER {summary["b_ter_overall"]:.2f}% → '
        f'target TER {summary["t_ter_overall"]:.2f}% '
        f'(Δ {summary["t_ter_overall"] - summary["b_ter_overall"]:+.2f}pp) &nbsp;|&nbsp; '
        f'utterances: {summary["n_compared"]} '
        f'(improved {summary["n_improved"]}, degraded {summary["n_degraded"]}, '
        f'unchanged {summary["n_unchanged"]})'
        f'</div>{legend}{toc_html}'
    )
    body = "\n".join(items)
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title><style>{CSS}</style></head>'
        f'<body><input type="checkbox" id="sb-toggle" class="sb-toggle">'
        f'<label for="sb-toggle" class="sb-toggle-btn" title="Toggle sidebar"></label>'
        f'<div class="layout">{sidebar_html}<div class="main">{head}{body}</div></div>'
        f'<script>{TOGGLE_JS}</script></body></html>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-path", required=True)
    ap.add_argument("--target-path", required=True)
    ap.add_argument("--baseline-name", default=None)
    ap.add_argument("--target-name", default=None)
    ap.add_argument("--metric", default="DisfluencyTolerant_TER",
                    choices=["NonDisfluency_TER", "NonVerbatim_TER",
                             "DisfluencyTolerant_TER", "Verbatim_TER"])
    ap.add_argument("--output-dir", default="tmp/ter_compare_disagree")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--include-entity", action="store_true",
                    help="Also emit entity-comparison pages from Metrics[*].EntityInfo.")
    ap.add_argument("--entity-metric", default="NonDisfluency_Simple_StandardRelax",
                    help="Metrics[*].MetricName whose EntityInfo to use for entity comparison.")
    args = ap.parse_args()

    bpath = Path(args.baseline_path).expanduser()
    tpath = Path(args.target_path).expanduser()
    bname = args.baseline_name or bpath.stem
    tname = args.target_name or tpath.stem

    base = load_records(bpath, args.metric)
    targ = load_records(tpath, args.metric)
    if args.include_entity:
        b_ent = load_entity_info(bpath, args.entity_metric)
        t_ent = load_entity_info(tpath, args.entity_metric)
        for uid, rec in base.items():
            rec["entity"] = b_ent.get(uid)
        for uid, rec in targ.items():
            rec["entity"] = t_ent.get(uid)
    common = sorted(set(base) & set(targ))

    summaries = []
    for uid in common:
        b, t = base[uid], targ[uid]
        summaries.append({
            "utt_id": uid,
            "b": b, "t": t,
            "ter_delta": t["ter"] - b["ter"],
            "edits_delta": t["edits"] - b["edits"],
        })

    tok_b = sum(s["b"]["tokens"] for s in summaries)
    tok_t = sum(s["t"]["tokens"] for s in summaries)
    ed_b = sum(s["b"]["edits"] for s in summaries)
    ed_t = sum(s["t"]["edits"] for s in summaries)
    summary = {
        "metric": args.metric,
        "n_compared": len(summaries),
        "b_ter_overall": (ed_b / tok_b * 100.0) if tok_b else 0.0,
        "t_ter_overall": (ed_t / tok_t * 100.0) if tok_t else 0.0,
        "n_improved": sum(1 for s in summaries if s["edits_delta"] < 0),
        "n_degraded": sum(1 for s in summaries if s["edits_delta"] > 0),
        "n_unchanged": sum(1 for s in summaries if s["edits_delta"] == 0),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{bname}__vs__{tname}.{args.metric}.disagree"
    summary_filename = f"{stem}.summary.json"

    # Per-focus deltas drive top-error sorting: overall pages use total edit
    # delta, lexical pages use lex_edits delta, fmt pages use fmt_edits delta.
    def cat_delta(s, cat_key):
        if cat_key == "lexical":
            return s["t"]["lex_edits"] - s["b"]["lex_edits"]
        if cat_key == "overall":
            return s["edits_delta"]
        if cat_key == "entity":
            be, te = s["b"].get("entity"), s["t"].get("entity")
            if not be or not te:
                return 0
            return te["n_ent_edits"] - be["n_ent_edits"]
        return s["t"]["fmt_edits"] - s["b"]["fmt_edits"]

    def cat_target_edits(s, cat_key):
        """Return the target model's absolute error count for a category."""
        if cat_key == "lexical":
            return s["t"]["lex_edits"]
        if cat_key == "entity":
            te = s["t"].get("entity")
            return te["n_ent_edits"] if te else 0
        if cat_key == "fmt":
            return s["t"]["fmt_edits"]
        return s["t"]["edits"]

    reports = {}
    pages: list = []
    categories = [
        ("overall", "overall edits"),
        ("fmt", "formatting (punc/cap/itn)"),
        ("lexical", "lexical (sub/ins/del)"),
    ]
    if args.include_entity:
        categories.append(("entity", f"entity ({args.entity_metric})"))
    for cat_key, cat_label in categories:
        subset = sorted(
            [s for s in summaries if cat_delta(s, cat_key) != 0],
            key=lambda s: abs(cat_delta(s, cat_key)),
            reverse=True,
        )[: args.top_n]
        items = []
        toc_entries = []
        card_category = None if cat_key == "overall" else cat_key
        for s in subset:
            card = card_html(s["utt_id"], s["b"], s["t"], bname, tname,
                             category=card_category)
            if card is not None:
                items.append(card)
                toc_entries.append({
                    "utt_id": s["utt_id"],
                    "b": s["b"], "t": s["t"],
                    "cat_delta": cat_delta(s, cat_key),
                    "ter_delta": s["ter_delta"],
                    "edits_delta": s["edits_delta"],
                    "target_edits": cat_target_edits(s, cat_key),
                })
        label = "top-errors"
        filename = f"{stem}.{cat_key}.{label}-top{len(subset)}.html"
        title = (f"TER disagreement — {cat_label} — top errors "
                 f"(sorted by |{cat_key} \u0394|) "
                 f"({len(items)}/{len(subset)} utterances) [{args.metric}]")
        pages.append({
            "cat_key": cat_key, "cat_label": cat_label, "label": label,
            "filename": filename, "title": title,
            "items": items, "toc_entries": toc_entries,
        })
        reports.setdefault(cat_key, {})[label] = str(out_dir / filename)

    def build_nav(current_filename: str) -> str:
        rows = ['<a href="' + html.escape(summary_filename) + '">summary.json</a>']
        for pg in pages:
            cls = "current" if pg["filename"] == current_filename else ""
            txt = f'{pg["cat_key"]} · {pg["label"]} ({len(pg["items"])})'
            rows.append(
                f'<a class="{cls}" href="{html.escape(pg["filename"])}">{html.escape(txt)}</a>'
            )
        return '<div class="nav"><h3>Reports</h3>' + "".join(rows) + '</div>'

    for pg in pages:
        nav_html = build_nav(pg["filename"])
        toc_html = render_toc(pg["toc_entries"], pg["cat_key"])
        sidebar_html = render_sidebar(pg["toc_entries"], pg["cat_key"], nav_html)
        (out_dir / pg["filename"]).write_text(
            render_page(pg["title"], pg["items"], bname, tname, summary, toc_html, sidebar_html)
        )

    summary["reports"] = reports
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

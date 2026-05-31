"""Single-model variant of compare_ter_disagree: show one model's hypothesis
diffed against the reference, emitting HTML that ONLY surfaces positions
where the model diverges from ref (substitutions, deletions, insertions,
formatting edits).

Layout mirrors compare_ter_disagree.py but with a single "model" column
instead of baseline/target, and "worst" pages sorted by edits descending
(no improved/degraded split).
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

# Reuse helpers from the compare script (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ter_disagree import (  # noqa: E402
    CSS,
    FMT_BUCKETS,
    TOGGLE_JS,
    _dedupe_runs,
    bucket_for,
    fmt_subtype,
    index_alignment,
    load_entity_info,
    load_records,
    render_ins_list,
    render_token,
    slugify,
    split_align,
)


# ---- diff rows (single side vs ref) -----------------------------------------


def diff_rows_single(m_idx):
    """Yield row dicts describing only positions where model diverges from ref.

    Row schema:
      kind: 'diff' | 'gap' | 'trailing'
      ref_idx, ref, m_hyp, m_bucket, m_ins
    """
    refs, per, ins_map = m_idx
    rows = []
    last_kept = -1
    n = len(refs)
    for i in range(n):
        ref = refs[i]
        m_hyp, m_b, m_sub = per.get(i, (None, "eq", ""))
        mi = _dedupe_runs([h for h, _ in ins_map.get(i, [])])
        # Skip positions where the model matches ref exactly (no edit, no ins).
        if m_b in ("eq", "relax") and m_hyp == ref and not mi:
            continue
        if last_kept >= 0 and i - last_kept > 1:
            rows.append({"kind": "gap", "skipped": i - last_kept - 1})
        elif last_kept < 0 and i > 0:
            rows.append({"kind": "gap", "skipped": i})
        rows.append({
            "kind": "diff",
            "ref_idx": i,
            "ref": ref,
            "m_hyp": m_hyp,
            "m_bucket": m_b,
            "m_ins": mi,
            "m_sub": m_sub,
        })
        last_kept = i
    if last_kept >= 0 and last_kept < n - 1:
        rows.append({"kind": "gap", "skipped": n - 1 - last_kept})
    if n in ins_map:
        rows.append({
            "kind": "trailing",
            "tokens": _dedupe_runs([h for h, _ in ins_map[n]]),
        })
    return rows


def row_category_single(row: dict) -> str | None:
    if row["kind"] == "trailing":
        return "lexical"
    if row["kind"] != "diff":
        return None
    if row["m_ins"] or row["m_bucket"] in {"sub", "del", "ins"}:
        return "lexical"
    if row["m_bucket"] in FMT_BUCKETS:
        return "fmt"
    return "fmt"


def filter_rows_by_category_single(rows: list, category: str,
                                   m_idx=None, n_ctx: int = 2,
                                   seg_starts=None) -> list:
    # 1) Select the divergence rows (and trailing) for this page.
    kept_diffs: list = []
    trailing: list = []
    for r in rows:
        if r["kind"] == "gap":
            continue
        if r["kind"] == "trailing":
            if category == "lexical":
                trailing.append(r)
            continue
        if row_category_single(r) != category:
            continue
        kept_diffs.append(r)

    if not kept_diffs and not trailing:
        return []

    kept_map = {r["ref_idx"]: r for r in kept_diffs}

    # 2) No ref context available -> old gap-only layout.
    if not m_idx or n_ctx <= 0:
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
    refs, per, _ = m_idx
    n = len(refs)
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
            m_hyp, _, _ = per.get(i, (None, "eq", ""))
            out.append({"kind": "ctx", "ref_idx": i, "ref": refs[i],
                        "hyp": m_hyp})
        prev_idx = i
    if prev_idx >= 0 and prev_idx < n - 1:
        out.append({"kind": "gap", "skipped": n - 1 - prev_idx})
    out.extend(trailing)
    return out


def render_diff_table_single(rows: list, model_name: str, cols_per_row: int = 12,
                             seg_starts=None) -> str:
    if not rows:
        return '<div class="meta">No disagreements — model matches ref everywhere.</div>'

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
                "m": '<span class="seg-bar">┊</span>',
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
                "m": "",
                "cls": "gap-cell",
            })
            continue
        if row["kind"] == "trailing":
            cols.append({
                "idx": "end",
                "ref": '<span class="none">∅</span>',
                "m": render_ins_list(row["tokens"]),
                "cls": "diff-col",
            })
            continue
        if row["kind"] == "ctx":
            cols.append({
                "idx": str(row["ref_idx"]),
                "ref": html.escape(row["ref"]),
                "m": '<span class="ctx-ditto">·</span>',
                "cls": "ctx-cell",
            })
            continue
        m_cell = (render_ins_list(row["m_ins"]) + (" " if row["m_ins"] else "")
                  + render_token(row["m_hyp"], row["m_bucket"], row.get("m_sub", "")))
        cols.append({
            "idx": str(row["ref_idx"]),
            "ref": html.escape(row["ref"]),
            "m": m_cell,
            "cls": "diff-col",
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
        m_row = [f'<td class="row-label">{html.escape(model_name)}</td>'] + [
            f'<td class="cell {c["cls"]}">{c["m"]}</td>' for c in group
        ]
        blocks.append(
            '<table class="diff row-block"><tbody>'
            + '<tr>' + ''.join(idx_row) + '</tr>'
            + '<tr class="ref">' + ''.join(ref_row) + '</tr>'
            + '<tr>' + ''.join(m_row) + '</tr>'
            + '</tbody></table>'
        )
    return ''.join(blocks)


def card_html_single(uid: str, m: dict, model_name: str, category: str | None = None) -> str | None:
    if category == "entity":
        return entity_card_html_single(uid, m, model_name)
    m_idx = index_alignment(m["align"], m["classes"])
    rows = diff_rows_single(m_idx)
    seg_starts = m.get("seg_starts") if category in ("fmt", "lexical") else None
    if category is not None:
        rows = filter_rows_by_category_single(rows, category, m_idx,
                                              seg_starts=seg_starts)
    n_diff = sum(1 for r in rows if r["kind"] in ("diff", "trailing"))
    if n_diff == 0:
        return None
    parts = [
        f'<span>{html.escape(model_name)} TER: <b>{m["ter"]:.2f}%</b> '
        f'({m["edits"]}/{m["tokens"]})</span>',
    ]
    if category in (None, "lexical"):
        parts.append(f'<span>lexical: <b>{m["lex_edits"]}</b></span>')
        parts.append(
            f'<span>lex breakdown: sub={m["lex_sub"]} ins={m["lex_ins"]} '
            f'del={m["lex_del"]}</span>'
        )
    if category in (None, "fmt"):
        parts.append(f'<span>fmt: <b>{m["fmt_edits"]}</b></span>')
        parts.append(
            f'<span>fmt breakdown: punc={m["punc_edits"]} '
            f'cap={m["cap_edits"]} itn={m["itn_edits"]}</span>'
        )
    parts.append(f'<span>diff cells: {n_diff}</span>')
    stats = "".join(parts)
    seg_starts = m.get("seg_starts") if category in ("fmt", "lexical") else None
    return (
        f'<div class="card">'
        f'<h2 id="{html.escape(slugify(uid))}">{html.escape(uid)}</h2>'
        f'<div class="stats">{stats}</div>'
        f'{render_diff_table_single(rows, model_name, seg_starts=seg_starts)}'
        f'</div>'
    )


def entity_card_html_single(uid: str, m: dict, model_name: str) -> str | None:
    e = m.get("entity")
    if not e:
        return None
    if e["n_trans_ents"] == 0 and e["n_reco_ents"] == 0:
        return None
    parts = [
        f'<span>ent edits: <b>{e["n_ent_edits"]}</b></span>',
        f'<span>ent words: {e["n_ent_words"]}</span>',
        f'<span>sub/ins/del: {e["n_ent_sub"]}/{e["n_ent_ins"]}/{e["n_ent_del"]}</span>',
        f'<span>EWER: {e["ewer"]:.2f}%</span>',
        f'<span>EER: {e["eer"]:.2f}%</span>',
        f'<span>recall: {e["recall"]:.2f}%</span>',
        f'<span>precision: {e["precision"]:.2f}%</span>',
        f'<span>trans ents: {e["n_trans_ents"]}</span>',
        f'<span>reco ents: {e["n_reco_ents"]}</span>',
    ]
    stats = "".join(parts)

    def _ent_list(d):
        if not d:
            return '<span class="none">∅</span>'
        return ", ".join(f'{html.escape(str(k))}×{v}' for k, v in d.items())

    ali = e["alignment_str"] or "(empty)"
    body = (
        f'<div class="ent-block"><div class="ent-label">{html.escape(model_name)}</div>'
        f'<pre class="ent-ali">{html.escape(ali)}</pre>'
        f'<div class="ent-meta">trans entities: {_ent_list(e["trans_uniq"])} '
        f'· reco entities: {_ent_list(e["reco_uniq"])}</div></div>'
    )
    return (
        f'<div class="card">'
        f'<h2 id="{html.escape(slugify(uid))}">{html.escape(uid)}</h2>'
        f'<div class="stats">{stats}</div>'
        f'{body}'
        f'</div>'
    )


# ---- TOC + sidebar ----------------------------------------------------------


def render_toc_single(entries: list, cat_key: str) -> str:
    if not entries:
        return ""
    if cat_key == "entity":
        return _render_entity_toc_single(entries)
    cat_label = "lex edits" if cat_key == "lexical" else "fmt edits"
    if cat_key == "lexical":
        sub_keys = [("sub", "lex_sub"), ("ins", "lex_ins"), ("del", "lex_del")]
    else:
        sub_keys = [("punc", "punc_edits"), ("cap", "cap_edits"), ("itn", "itn_edits")]

    rows = []
    for i, e in enumerate(entries, 1):
        uid = e["utt_id"]
        anchor = slugify(uid)
        m = e["m"]
        cat_val = e["cat_val"]
        sub_cells = "".join(
            f'<td class="num">{m[k]}</td>' for _, k in sub_keys
        )
        rows.append(
            f'<tr>'
            f'<td class="num">{i}</td>'
            f'<td class="uid"><a href="#{html.escape(anchor)}">{html.escape(uid)}</a></td>'
            f'<td class="num">{cat_val}</td>'
            + sub_cells
            + f'<td class="num">{m["edits"]}</td>'
            f'<td class="num">{m["ter"]:.2f}%</td>'
            f'<td class="num">{m["tokens"]}</td>'
            f'</tr>'
        )
    tot_cat = sum(e["cat_val"] for e in entries)
    tot_edits = sum(e["m"]["edits"] for e in entries)
    tot_tokens = sum(e["m"]["tokens"] for e in entries)
    ter_overall = (tot_edits / tot_tokens * 100.0) if tot_tokens else 0.0
    tot_sub_cells = "".join(
        f'<td class="num">{sum(e["m"][k] for e in entries)}</td>'
        for _, k in sub_keys
    )
    totals_row = (
        f'<tr style="background:#eef2f7;font-weight:600">'
        f'<td class="num">Σ</td>'
        f'<td class="uid">overall ({len(entries)} utts)</td>'
        f'<td class="num">{tot_cat}</td>'
        + tot_sub_cells
        + f'<td class="num">{tot_edits}</td>'
        f'<td class="num">{ter_overall:.2f}%</td>'
        f'<td class="num">{tot_tokens}</td>'
        f'</tr>'
    )
    sub_headers = "".join(f'<th>{lbl}</th>' for lbl, _ in sub_keys)
    return (
        f'<details class="toc-wrap" open>'
        f'<summary>Utterances ({len(entries)}) — click row to jump</summary>'
        f'<table class="toc"><thead><tr>'
        f'<th>#</th><th>UtteranceId</th><th>{cat_label}</th>'
        + sub_headers
        + f'<th>edits</th><th>TER</th><th>tokens</th>'
        f'</tr></thead><tbody>'
        + totals_row + "".join(rows)
        + '</tbody></table></details>'
    )


def _render_entity_toc_single(entries: list) -> str:
    rows = []
    for i, e in enumerate(entries, 1):
        en = e["m"]["entity"]
        anchor = slugify(e["utt_id"])
        rows.append(
            f'<tr>'
            f'<td class="num">{i}</td>'
            f'<td class="uid"><a href="#{html.escape(anchor)}">{html.escape(e["utt_id"])}</a></td>'
            f'<td class="num">{en["n_ent_edits"]}</td>'
            f'<td class="num">{en["n_ent_words"]}</td>'
            f'<td class="num">{en["ewer"]:.2f}%</td>'
            f'<td class="num">{en["eer"]:.2f}%</td>'
            f'<td class="num">{en["recall"]:.2f}%</td>'
            f'<td class="num">{en["precision"]:.2f}%</td>'
            f'<td class="num">{en["n_trans_ents"]}</td>'
            f'<td class="num">{en["n_reco_ents"]}</td>'
            f'</tr>'
        )
    tot_edits = sum(e["m"]["entity"]["n_ent_edits"] for e in entries)
    tot_words = sum(e["m"]["entity"]["n_ent_words"] for e in entries)
    tot_trans = sum(e["m"]["entity"]["n_trans_ents"] for e in entries)
    tot_reco = sum(e["m"]["entity"]["n_reco_ents"] for e in entries)
    tot_trans_match = sum(e["m"]["entity"]["n_trans_matched"] for e in entries)
    tot_reco_match = sum(e["m"]["entity"]["n_reco_matched"] for e in entries)
    ewer = (tot_edits / tot_words * 100.0) if tot_words else 0.0
    eer = ((tot_trans - tot_trans_match) / tot_trans * 100.0) if tot_trans else 0.0
    recall = (tot_trans_match / tot_trans * 100.0) if tot_trans else 0.0
    precision = (tot_reco_match / tot_reco * 100.0) if tot_reco else 0.0
    totals_row = (
        f'<tr style="background:#eef2f7;font-weight:600">'
        f'<td class="num">Σ</td>'
        f'<td class="uid">overall ({len(entries)} utts)</td>'
        f'<td class="num">{tot_edits}</td>'
        f'<td class="num">{tot_words}</td>'
        f'<td class="num">{ewer:.2f}%</td>'
        f'<td class="num">{eer:.2f}%</td>'
        f'<td class="num">{recall:.2f}%</td>'
        f'<td class="num">{precision:.2f}%</td>'
        f'<td class="num">{tot_trans}</td>'
        f'<td class="num">{tot_reco}</td>'
        f'</tr>'
    )
    return (
        f'<details class="toc-wrap" open>'
        f'<summary>Utterances ({len(entries)}) — click row to jump</summary>'
        f'<table class="toc"><thead><tr>'
        f'<th>#</th><th>UtteranceId</th>'
        f'<th>ent edits</th><th>ent words</th>'
        f'<th>EWER</th><th>EER</th><th>recall</th><th>precision</th>'
        f'<th>trans</th><th>reco</th>'
        f'</tr></thead><tbody>'
        + totals_row + "".join(rows)
        + '</tbody></table></details>'
    )


def render_sidebar_single(entries: list, cat_key: str, nav_html: str = "") -> str:
    if not entries and not nav_html:
        return ""
    items = []
    for e in entries:
        uid = e["utt_id"]
        anchor = slugify(uid)
        cv = e["cat_val"]
        items.append(
            f'<li><a href="#{html.escape(anchor)}">{html.escape(uid)}'
            f'<span class="d-pos">{cv}</span></a></li>'
        )
    label = {"lexical": "lex edits", "fmt": "fmt edits", "entity": "ent edits"}.get(cat_key, "edits")
    body = f'<h3>Utterances · {label}</h3><ol>{"".join(items)}</ol>' if entries else ""
    return f'<aside class="sidebar">{nav_html}{body}</aside>'


def render_page_single(title: str, items: list, model_name: str, summary: dict,
                       toc_html: str = "", sidebar_html: str = "") -> str:
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
        '<br>Showing only positions where the model diverges from the reference.'
        '</div>'
    )
    head = (
        f'<h1>{html.escape(title)}</h1>'
        f'<div class="meta">'
        f'metric = <code>{html.escape(summary["metric"])}</code> &nbsp;|&nbsp; '
        f'model = <code>{html.escape(model_name)}</code><br>'
        f'overall TER: <b>{summary["ter_overall"]:.2f}%</b> '
        f'({summary["edits_overall"]}/{summary["tokens_overall"]}) &nbsp;|&nbsp; '
        f'utterances: {summary["n_utts"]} '
        f'(with errors: {summary["n_with_errors"]})'
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


# ---- main -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--metric", default="DisfluencyTolerant_TER",
                    choices=["NonDisfluency_TER", "NonVerbatim_TER",
                             "DisfluencyTolerant_TER", "Verbatim_TER"])
    ap.add_argument("--output-dir", default="tmp/ter_ref_diff")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--include-entity", action="store_true")
    ap.add_argument("--entity-metric", default="NonDisfluency_Simple_StandardRelax")
    args = ap.parse_args()

    mpath = Path(args.model_path).expanduser()
    mname = args.model_name or mpath.stem

    recs = load_records(mpath, args.metric)
    if args.include_entity:
        ent = load_entity_info(mpath, args.entity_metric)
        for uid, r in recs.items():
            r["entity"] = ent.get(uid)

    summaries = [
        {"utt_id": uid, "m": m}
        for uid, m in sorted(recs.items())
    ]
    tok = sum(s["m"]["tokens"] for s in summaries)
    ed = sum(s["m"]["edits"] for s in summaries)
    summary = {
        "metric": args.metric,
        "n_utts": len(summaries),
        "tokens_overall": tok,
        "edits_overall": ed,
        "ter_overall": (ed / tok * 100.0) if tok else 0.0,
        "n_with_errors": sum(1 for s in summaries if s["m"]["edits"] > 0),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{mname}.{args.metric}.ref-diff"
    summary_filename = f"{stem}.summary.json"

    def cat_val(s, cat_key):
        if cat_key == "lexical":
            return s["m"]["lex_edits"]
        if cat_key == "entity":
            e = s["m"].get("entity")
            return e["n_ent_edits"] if e else 0
        return s["m"]["fmt_edits"]

    categories = [
        ("fmt", "formatting (punc/cap/itn)"),
        ("lexical", "lexical (sub/ins/del)"),
    ]
    if args.include_entity:
        categories.append(("entity", f"entity ({args.entity_metric})"))

    reports = {}
    pages: list = []
    for cat_key, cat_label in categories:
        worst = sorted(
            [s for s in summaries if cat_val(s, cat_key) > 0],
            key=lambda s: cat_val(s, cat_key),
            reverse=True,
        )[: args.top_n]
        items = []
        toc_entries = []
        for s in worst:
            card = card_html_single(s["utt_id"], s["m"], mname, category=cat_key)
            if card is not None:
                items.append(card)
                toc_entries.append({
                    "utt_id": s["utt_id"],
                    "m": s["m"],
                    "cat_val": cat_val(s, cat_key),
                })
        filename = f"{stem}.{cat_key}.worst-top{len(worst)}.html"
        title = (f"Ref diff — {cat_label} — worst "
                 f"({len(items)}/{len(worst)} utterances) [{args.metric}]")
        pages.append({
            "cat_key": cat_key, "label": "worst",
            "filename": filename, "title": title,
            "items": items, "toc_entries": toc_entries,
        })
        reports.setdefault(cat_key, {})["worst"] = str(out_dir / filename)

    def build_nav(current_filename: str) -> str:
        rows = ['<a href="' + html.escape(summary_filename) + '">summary.json</a>']
        for pg in pages:
            cls = "current" if pg["filename"] == current_filename else ""
            txt = f'{pg["cat_key"]} · worst ({len(pg["items"])})'
            rows.append(
                f'<a class="{cls}" href="{html.escape(pg["filename"])}">{html.escape(txt)}</a>'
            )
        return '<div class="nav"><h3>Reports</h3>' + "".join(rows) + '</div>'

    for pg in pages:
        nav_html = build_nav(pg["filename"])
        toc_html = render_toc_single(pg["toc_entries"], pg["cat_key"])
        sidebar_html = render_sidebar_single(pg["toc_entries"], pg["cat_key"], nav_html)
        (out_dir / pg["filename"]).write_text(
            render_page_single(pg["title"], pg["items"], mname, summary, toc_html, sidebar_html)
        )

    summary["reports"] = reports
    (out_dir / summary_filename).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

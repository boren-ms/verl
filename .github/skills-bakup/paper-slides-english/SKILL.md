---
name: paper-slides-english
description: "Read an academic paper (arXiv URL/ID or local PDF) and produce a compact slides-style English HTML deck covering key insights, figures, tables, and equations. Prefer TeX source for faithful figure/table extraction. Use when: paper slides, slides HTML, paper deck, slides English, read paper English, paper walkthrough slides, arXiv slides, generate slides HTML, paper summary slides, 2xxx.xxxxx slides."
argument-hint: "arXiv abs/html/pdf URL, arXiv ID (e.g. 2601.13409), or local PDF path"
---

# Paper Slides (English HTML Deck)

Produce a self-contained slides-style **HTML file in English** that presents an academic paper as a compact, visual deck — each "slide" is a card covering one key topic, with inline figures, tables, and equations. The goal is dense-but-readable: every slide should stand alone like a real presentation slide.

## When to Use

- User gives an arXiv URL/ID and asks for "slides", "paper deck", "slides HTML", or "paper summary in English"
- Producing a shareable single-file HTML presentation of a paper for team review or seminars
- Quick paper walkthrough emphasizing figures/tables over prose

## Inputs

One of:
1. arXiv URL (`abs`, `html`, or `pdf` form), e.g. `https://arxiv.org/abs/2601.13409`
2. Bare arXiv ID, e.g. `2601.13409` (optionally with `v1`, `v2`, ...)
3. A direct paper URL on another site, or a local PDF path

If none provided, ask the user for the paper URL/ID.

## Source Fetching Strategy (TeX-first → HTML → PDF fallback)

For arXiv papers, **always attempt the TeX source first**. It preserves exact equations, figure paths, and table source — far superior to the rendered HTML and dramatically better than PDF text.

### Step 1 — Normalize Input

From any arXiv input, derive:
- `abs_url`  = `https://arxiv.org/abs/<id>`
- `tex_url`  = `https://arxiv.org/src/<id>` (same as `https://arxiv.org/e-print/<id>`)
- `html_url` = `https://arxiv.org/html/<id>` (try `v1`, `v2`, ... if bare id 404s)
- `pdf_url`  = `https://arxiv.org/pdf/<id>`

### Step 2 — Try TeX Source (preferred)

```bash
mkdir -p /tmp/paper_<id> && cd /tmp/paper_<id>
curl -sSL -A "Mozilla/5.0" "https://arxiv.org/src/<id>" -o source.tar.gz
file source.tar.gz
( tar -xzf source.tar.gz 2>/dev/null ) || ( gunzip -c source.tar.gz > main.tex )
ls -la
```

Identify the main `.tex` file (the one containing `\documentclass` and `\begin{document}`). If multiple `.tex` files exist, the entry point typically contains `\title{...}` and `\include{}`/`\input{}` calls.

**Extract from TeX:**
- Title, authors, abstract — `\title`, `\author`, `\begin{abstract}`
- Section structure — `\section`, `\subsection`
- **Figures**: every `\includegraphics{path}` with its `\caption{...}` and `\label{fig:...}`. Convert non-PNG/JPG assets:
  - `.pdf` figure → `pdftoppm -png -r 150 fig.pdf fig_out` → `fig_out-1.png`
  - `.eps` figure → `convert -density 150 fig.eps fig.png` (ImageMagick)
  - `.png`/`.jpg` → use directly, embed as base64 for self-contained output
- **Tables**: copy `\begin{table}...\end{table}` LaTeX blocks; convert to HTML tables (use `pandoc -f latex -t html5` on each snippet, or reconstruct manually)
- **Equations**: copy `equation`/`align`/`gather` environments verbatim — MathJax renders them when wrapped in `\[ ... \]` or `$$ ... $$`

### Step 3 — Fall Back to HTML

If TeX is unavailable (PDF-only submission or withdrawn):

Fetch `html_url` via `fetch_webpage`; if blocked, try `open_browser_page` + `read_page`.

Collect:
- All figure image URLs (resolve to absolute, typically `https://arxiv.org/html/<id>/x1.png` etc.)
- All `<table>` blocks and inline math spans
- Section text

### Step 4 — Fall Back to PDF

Only when both TeX and HTML fail:
1. `curl -sSL "$pdf_url" -o /tmp/paper_<id>.pdf`
2. Text: `pdftotext -layout /tmp/paper_<id>.pdf -`
3. Figures: `pdfimages -all /tmp/paper_<id>.pdf /tmp/paper_<id>_img`; fallback: `pdftoppm -png -r 110 paper.pdf page` then crop with PIL
4. Embed figures as base64 `data:image/...;base64,...`
5. Mark tables as "reconstructed from PDF"

Tell the user which source was used (TeX / HTML / PDF).

## Output: Slides HTML File

Write a **single self-contained `.html` file** to:

```
paper_notes/<arxiv_id_or_slug>_slides.html
```

Use [slides_template.html](./assets/slides_template.html) as the starting point. Fill in all `{{PLACEHOLDER}}` blocks.

The `paper_notes/` directory is git-ignored; create it if absent.

### Slide Deck Structure

Build the deck as a vertical sequence of **slide cards** (each `<section class="slide slide-N">`). Recommended slide order:

| # | Slide | Contents |
|---|-------|----------|
| 1 | **Title** | Paper title, authors, venue/year, arXiv badge, abs/html/pdf links |
| 2 | **TL;DR** | 2–4 bullet one-liners: problem → gap → proposal → result. No prose. |
| 3 | **Overview Figure** | The main architecture/method diagram (Fig. 1 or Fig. 2 from paper). Full-width image + 3–5 bullet annotations keyed to parts of the figure (e.g., "① Encoder takes mel spectrogram → …"). |
| 4 | **Motivation & Problem** | What problem, why hard, what's missing in prior work. 4–6 tight bullets. |
| 5–N | **Method slides** | One slide per major component. Layout: figure/equation on left, bullet list on right (CSS grid). Include key equations inline with MathJax. |
| N+1 | **Main Results** | Primary result table embedded in slide. Highlight best numbers (`class="best"`). 2–3 bullet takeaways above or below. |
| N+2 | **Ablations / Analysis** | Ablation table or chart + 3–5 interpretation bullets. |
| N+3 | **Key Insights & Takeaways** | 5–8 bullets, each one a concrete, self-contained insight (no vague claims). Bold the key term in each bullet. |
| N+4 | **Limitations & Future Work** | Authors' admitted limitations + 2–3 of your own critiques. |

**Rules:**
- Each slide is **self-contained**: a reader looking at only one slide should understand its point without context from adjacent slides.
- **Every figure must appear in the slide where it is discussed**, not in a separate "figures" appendix.
- Every table must have a 1-line caption explaining what it shows.
- Equations must have a 1-line English explanation immediately below.
- Do **not** pad slides with background or related work prose unless it is directly needed to explain a technical choice.

### Design Requirements

**Layout:**
- Full-width (`width: 100%; max-width: 100%`) with `clamp(16px, 2.5vw, 40px)` side padding
- Each slide: white card, 16px gap between slides, `border-radius: 10px`, subtle box-shadow
- Two-column grid for method slides (`grid-template-columns: 1fr 1fr`); figure in left column, bullets in right
- ≤900px: all columns collapse to single-column
- Left-side **slide index** (sticky nav) listing slide titles, auto-highlighted on scroll

**Typography:**
- Body: 16px, line-height 1.65, system sans-serif (`-apple-system, "Segoe UI", "Helvetica Neue", sans-serif`)
- Slide title: 20px bold, accent color left-border (4px solid)
- Bullet text: 15px, tight list (`li { margin: 5px 0 }`)
- Table: 13–14px, zebra rows, `thead` in light blue
- Figure caption / equation label: 13px muted gray
- Code/pseudocode: `14px monospace`, light background block

**Color palette:**
- Page background: `#f0f2f5`
- Card background: `#ffffff`
- Accent A (blue): `#1a6fb5` — title slide, nav highlight
- Accent B (teal): `#0d7d6b` — method slides
- Accent C (indigo): `#4f46e5` — results & insights
- Accent D (rose): `#be123c` — limitations, best-value highlights
- TL;DR bar: amber `#d97706` background strip

**Slide header bar:**
- Each `<section class="slide">` has a `<div class="slide-header">` with colored left-border + emoji + slide title

**Figures:**
- `<figure>` inside `.figure-pane` div; `max-width: 100%; height: auto`
- Caption: `<figcaption>` below image, 13px muted
- Images: embed as base64 for full self-containment, or use absolute `https://arxiv.org/html/<id>/...` URLs if file would exceed ~20MB

**Tables:**
- `thead` background `#dbeafe` (light blue), sticky top
- Zebra: `tbody tr:nth-child(even)` → `#f8fafc`
- Best value: `<td class="best">` → red bold
- Overflow: horizontal scroll wrapper

**Equations:**
- Wrap in `<div class="eq-block">` with light gray background
- Label top-right: `<span class="eq-label">(Eq. N)</span>`
- Core equation: add `class="eq-star"` → amber left-border + subtle highlight
- Explanation: `<p class="eq-explain">` immediately below

**MathJax:** Include CDN script in `<head>`:
```html
<script>MathJax = { tex: { inlineMath: [['\\(','\\)'],['$','$']], displayMath: [['$$','$$'],['\\[','\\]']] } };</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

**Self-contained:** Zero external CSS/JS dependencies except MathJax CDN. All images base64-embedded unless HTML size would exceed ~20MB (then write to `paper_notes/<id>_assets/` and use relative paths; note this in the header).

## Procedure

1. Normalize arXiv input → derive `tex_url`, `html_url`, `pdf_url`.
2. **Try TeX source** (`arxiv.org/src/<id>`); unpack and locate main `.tex` + figure files. Convert non-PNG figures to PNG. Convert TeX tables to HTML.
3. If TeX unavailable → try HTML render; collect figure URLs, table HTML, section text.
4. If HTML also unavailable → fall back to PDF extraction.
5. Read the paper end-to-end; identify all sections, figures, tables, key equations.
6. Plan the slide sequence (see table above). Assign each figure to the slide where it is explained.
7. Draft each slide as a `<section class="slide">` block. Keep each slide tight: title + bullets + at most one figure or table per slide (split into two slides if needed).
8. Fill in [slides_template.html](./assets/slides_template.html) and save to `paper_notes/<id>_slides.html`.
9. Report the output path and which source path was used (TeX / HTML / PDF). Offer to open in browser preview.

## Quality Checklist

- [ ] All text in English; technical abbreviations expanded on first use
- [ ] Each slide has a clear, specific title (not "Method" but "Biased Reward Computation")
- [ ] Overview figure is on Slide 3, full-width, with annotated callouts
- [ ] Every figure appears in the slide that discusses it; no orphaned figures appendix
- [ ] Every table has a 1-line caption; best values marked `class="best"`
- [ ] Every key equation has a 1-line plain-English explanation
- [ ] Key Insights slide has ≥5 concrete, falsifiable bullets (not vague praise)
- [ ] No slide is a wall of text — max ~8 bullets per slide; break longer content into two slides
- [ ] HTML renders correctly with MathJax (no broken LaTeX)
- [ ] File is self-contained (base64 images) or assets folder documented in header comment

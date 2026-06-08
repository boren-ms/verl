---
name: paper-reading
description: "Read an academic paper (arXiv or other URL/PDF) and produce TWO well-designed HTML slide decks (reveal.js): first an English deck, then a Chinese (简体中文) deck. Both use reveal.js with vertical layout, MathJax formulas, and figures/tables from TeX source. Use when: read paper, paper reading, paper summary, paper report, 读论文, 论文精读, summarize arxiv paper, 中文论文解读, generate paper HTML report, arxiv 2xxx.xxxxx, paper walkthrough, 论文笔记, paper slides."
argument-hint: "arXiv abs/html/pdf URL, arXiv ID (e.g. 2405.22263), or local PDF path"
---

# Paper Reading (Dual-Language reveal.js Slide Decks)

Produce **two** self-contained reveal.js HTML slide decks for an academic paper:

1. **English slides** (`_en.html`) — generated **first**
2. **Chinese slides** (`_zh.html`) — generated **second**, using the same extracted data

Both decks use reveal.js (CDN-loaded) with MathJax equations, inline figures, and faithful tables from the original source.

## When to Use

- User gives an arXiv URL / ID / PDF and asks for a reading note, summary, or report
- User says "read this paper" / "paper reading" / "paper summary" / "generate report"
- User says "读这篇论文" / "论文精读" / "用中文总结" / "生成 HTML 报告"
- Producing shareable single-file HTML presentations of a paper

## Inputs

One of:
1. arXiv URL (`abs`, `html`, or `pdf` form), e.g. `https://arxiv.org/abs/2405.22263`
2. Bare arXiv ID, e.g. `2405.22263` (optionally with `v1`, `v2`, ...)
3. A direct paper URL on another site, or a local PDF path

If none provided, ask the user for the paper URL/ID.

## Source Fetching Strategy (TeX-first, HTML-next, PDF-fallback)

For arXiv papers, **prefer the TeX source first**, then the rendered HTML, then PDF only as a last resort. TeX source preserves the exact equations, table source, figure file references, and section structure that the authors wrote — far richer than the HTML render and dramatically better than PDF text extraction.

For non-arXiv URLs / local PDFs, skip to the HTML or PDF step directly (TeX is usually unavailable).

### Step 1 — Normalize the input

From any arXiv input, derive:
- `abs_url`    = `https://arxiv.org/abs/<id>`
- `tex_url`    = `https://arxiv.org/src/<id>` (also try `https://arxiv.org/e-print/<id>` — same content, gzip tar)
- `html_url`   = `https://arxiv.org/html/<id>` (also try `https://arxiv.org/html/<id>v1`, `v2`, … if bare id fails)
- `pdf_url`    = `https://arxiv.org/pdf/<id>`

For non-arXiv URLs, use the given URL as-is; if it looks like a PDF, jump straight to Step 4.

### Step 2 — Try TeX source (preferred for arXiv)

Download and unpack:

```bash
mkdir -p /tmp/paper_<id> && cd /tmp/paper_<id>
curl -sSL -A "Mozilla/5.0" "$tex_url" -o source.tar.gz
file source.tar.gz
( tar -xzf source.tar.gz 2>/dev/null ) || ( gunzip -c source.tar.gz > main.tex )
ls -la
```

Identify the main `.tex` file (containing `\documentclass` and `\begin{document}`).

Extract from the TeX source:
- **Title, authors, abstract** (`\title`, `\author`, `\begin{abstract}`)
- **Section structure** (`\section`, `\subsection`)
- **Figures**: `\includegraphics{path}` — record each figure file path and its `\caption{...}`. Convert non-PNG figures to PNG:
  - `.pdf` → `pdftoppm -png -r 150 fig.pdf fig`
  - `.eps` → `convert -density 150 fig.eps fig.png`
  - `.png` / `.jpg` → use directly
- **Tables**: copy `\begin{table}...\end{table}` blocks; reconstruct as HTML tables
- **Equations**: copy `equation`, `align`, `gather` environments verbatim for MathJax rendering

### Step 3 — Fall back to HTML

If TeX source is unavailable, fetch `html_url`. Record figure URLs, table HTML, section structure, and key equations.

### Step 4 — Fall back to PDF

Only if both TeX and HTML are unavailable:
1. Download: `curl -sSL "$pdf_url" -o /tmp/paper_<id>.pdf`
2. Text: `pdftotext -layout /tmp/paper_<id>.pdf -`
3. Figures: `pdfimages -all` or `pdftoppm -png -r 110`
4. Embed figures as base64 or write to `paper_notes/<id>_assets/` if HTML exceeds ~20 MB

Tell the user which source was used (TeX / HTML / PDF).

---

## Output: Two HTML Slide Decks

Write **two** self-contained `.html` files to `paper_notes/`:

```
paper_notes/<arxiv_id_or_slug>_en.html    ← English (generated first)
paper_notes/<arxiv_id_or_slug>_zh.html    ← Chinese (generated second)
paper_notes/<arxiv_id_or_slug>_assets/    ← Extracted figures (shared)
```

The `paper_notes/` directory is git-ignored. Create it if absent.

Both decks share the same figures. Both use the reveal.js slide format described below.

---

## Part A — English Slides (`_en.html`)

Generate this deck **first**. All text in English.

### Slide Structure (English)

Each slide is a `<section>` inside reveal.js. Use **vertical layout** — content stacked top-to-bottom within each slide (no side-by-side two-column layouts). Bullets always appear **below** figures/equations/tables.

> **Layout principles:**
> - **Vertical stacking:** Figures, equations, and tables at top; bullets below. Never use two-column `flexbox` layouts.
> - **Figure-first:** Overview/method diagrams appear prominently before explanatory text.
> - **Full-width tables:** Use original column names from the paper source (e.g., "test-clean" not "cl"). Tables span full slide width. Include all data columns and values from original — never abbreviate.
> - **Centered formulas:** Equation boxes use `width: fit-content; margin: auto` to shrink to content and center.
> - **Spacing:** Leave clear vertical gaps between sections (headings, equations, tables, bullets).

**Slide 1 — Title + TL;DR** (combined on one slide):
- Title, authors, affiliations, venue, arXiv links as badges
- TL;DR as a translucent info box at bottom: Problem → Proposal → Result (3 lines)
- Uses dark gradient background

**Slide 2 — Overview Figure:**
- Paper's most representative figure displayed prominently (centered)
- 2–3 bullets below explaining what the figure shows

**Slide 3 — Motivation & Problem:**
- 4–6 bullet points explaining why this work is needed
- Reference specific numbers from the paper's tables

**Slides 4–6 — Method slides** (one per key technique):
- Each slide: equation box(es) or figure at top → bullets below
- Key equations in `.eq-box` with label and 1-line explanation
- Core equation gets `.eq-star` (gold left border)
- Figures with `<figcaption>` referencing figure number

**Slide 7 — Main Results:**
- Full results table with original column names and all values from paper
- Use `BWER (WER/UWER)` format if paper uses it — preserve cell structure
- Table caption + 2–3 takeaway bullets below

**Slide 8 — Ablation Studies:**
- Tables stacked vertically (not side-by-side)
- Each table with caption, then summary bullets below

**Slide 9 — Key Insights & Contributions:**
- Tagged list: `[Novel]` / `[Claim]` / `[Novel+Claim]`
- Each with left border accent via `.insights li` style

**Slide 10 — Limitations & Future Work:**
- Bullet list of stated limitations + critical observations
- Brief "Future" note at bottom

### Optional Slides (English)

Include only if the paper warrants:
- **Reproduction Notes** — Key hyperparams, data scale, training recipe
- **Related Work Comparison** — Side-by-side table with closest baselines

---

## Part B — Chinese Slides (`_zh.html`)

Generate this deck **second**, after the English slides. All body text in Chinese (简体中文). English technical terms should be given a Chinese translation on first occurrence.

Use the same slide structure as Part A, translated to Chinese. Same reveal.js framework, same vertical layout rules, same figure/table/equation placement.

### Slide Structure (Chinese)

> **排版总原则：**
> - **垂直堆叠：** 图/公式/表在上，要点在下。不使用双栏布局。
> - **图优先：** 概览图居中显示在解释文字之前。
> - **完整表格：** 使用论文原始列名，包含所有数据列和数值。
> - **公式居中：** 公式框自适应内容宽度并居中显示。

**幻灯片 1 — 标题 + TL;DR**（合并为一页）:
- 英文标题 + 中文副标题，作者，机构，会议，arXiv 链接
- TL;DR 信息框：问题 → 方案 → 结果

**幻灯片 2 — 概览图**

**幻灯片 3 — 研究动机与问题**

**幻灯片 4–6 — 方法**（每项关键技术一页）:
- 公式/图在上，要点在下

**幻灯片 7 — 主要结果**

**幻灯片 8 — 消融实验**

**幻灯片 9 — 核心要点与贡献**:
- 标签：`[创新]` / `[论断]` / `[创新+论断]`

**幻灯片 10 — 局限与个人评价**

### Optional Slides (Chinese)

- **复现要点** — 关键超参、数据规模、训练配方
- **相关工作对比** — 与最相近基线的对比表

---

## Design Requirements — reveal.js Slide Decks (Both Languages)

Both decks use **reveal.js 5.2.1** (CDN-loaded) with the **white theme**, MathJax 3, and a consistent CSS stylesheet.

### reveal.js Setup

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.2.1/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.2.1/dist/theme/white.css">
<script>window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]}}</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
```

```javascript
Reveal.initialize({
  hash: true, slideNumber: 'c/t',
  width: 1280, height: 720, margin: 0.08,
  transition: 'slide', center: false, plugins: []
});
Reveal.on('slidechanged', () => {
  if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise();
});
```

### CSS Design System

```css
/* Base */
.reveal { font-size: 24px; }
.reveal .slides section { overflow: hidden; padding: 20px 30px; }

/* Headings — generous bottom margin for spacing */
.reveal h2 { font-size: 1.15em; border-bottom: 2px solid #1a6fb5; margin-bottom: 18px; }

/* Content — margin: 10px+ between sections */
.reveal ul, .reveal ol { font-size: 0.78em; line-height: 1.45; margin: 10px 0; }
.reveal p { font-size: 0.8em; margin-bottom: 8px; }
.reveal figure { margin: 0 0 12px 0; }

/* Tables — full width, original column names, generous spacing */
.reveal table { font-size: 0.52em; width: 100%; white-space: nowrap; margin: 10px auto 14px; }
.reveal table th { background: #dbeafe; }
.reveal table .best { color: #be123c; font-weight: 700; }
.table-caption { font-size: 0.48em; margin-bottom: 14px; }

/* Equation boxes — fit content width, centered, with spacing */
.eq-box { width: fit-content; max-width: 100%; margin: 10px auto;
           font-size: 0.72em; overflow-x: auto; text-align: center; }
.eq-box.eq-star { border-left: 4px solid #d97706; background: #fffbeb; }

/* Tags for contributions */
.tag { font-size: 0.52em; font-weight: 700; padding: 1px 7px; border-radius: 10px; }
.tag-new { background: #fce7f3; color: #9d174d; }
.tag-claim { background: #dbeafe; color: #1e40af; }
.tag-both { background: #fef3c7; color: #92400e; }

/* Title slide */
.title-slide { background: linear-gradient(135deg, #0f2a5c 0%, #1a6fb5 60%, #0d7d6b 100%); }
/* TL;DR box on title slide: semi-transparent white */
/* background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); */
```

### Key Layout Rules

1. **Vertical stacking only** — no two-column `flexbox` / `.cols` layouts. Content flows top-to-bottom: figure/equation → bullets.
2. **Title + TL;DR combined** on slide 1 (not separate slides).
3. **Tables use original names** from the paper source — never abbreviate column headers (e.g., use "test-clean" not "cl", "Edit Level" not "Edit", "Reference Aware" not "Ref").
4. **Include all table values** from original paper. If paper shows BWER with (WER/UWER), include all three.
5. **Equation boxes centered** with `width: fit-content; margin: auto`.
6. **Clear spacing** between content sections — 10-18px margins between headings, equations, tables, and bullet lists.
7. **Section titles with emoji** prefix: 🎯 📊 🔬 ⚙️ ⚠️ 💡 📌 🧪
8. **Images**: external files in `_assets/` directory, `max-height` constrained (200-280px).
9. **Chinese slides** use Chinese font stack: `'PingFang SC', 'Microsoft YaHei', ...`
10. **Horizontal navigation** (left/right arrows) — slides are siblings, not nested.

---

## Procedure Summary

1. Normalize input → derive `tex_url` / `html_url` / `pdf_url`.
2. **Try TeX source first**; if unavailable, try HTML; if that also fails, fall back to PDF.
3. Read the paper end-to-end; identify sections, figures, tables, key equations.
4. Convert TeX figures to PNG (`pdftoppm` / ImageMagick); save to `paper_notes/<id>_assets/`.
5. **Generate English slides first** (`_en.html`): ~10 reveal.js slides using vertical layout.
6. **Generate Chinese slides second** (`_zh.html`): same structure, all text in Chinese.
7. Save both files to `paper_notes/`.
8. Report both output paths and which source was used (TeX / HTML / PDF).

## Quality Checklist

### English Slides (`_en.html`)
- [ ] All body text in English
- [ ] reveal.js slide deck with horizontal navigation
- [ ] Vertical content layout within each slide (no two-column flexbox)
- [ ] Title + TL;DR combined on slide 1
- [ ] Overview figure on slide 2 with explanatory bullets below
- [ ] Tables use original column names from paper — no abbreviations
- [ ] All table values from paper preserved (never omit data)
- [ ] Equation boxes centered with `width: fit-content`
- [ ] Clear spacing between content sections (10-18px margins)
- [ ] MathJax renders correctly; re-renders on slide change
- [ ] Badge links to arXiv abs / html / pdf on title slide

### Chinese Slides (`_zh.html`)
- [ ] 所有正文为中文；英文术语首次出现时给出中译
- [ ] reveal.js 幻灯片，水平导航
- [ ] 每页内容垂直堆叠（不使用双栏布局）
- [ ] 标题 + TL;DR 合并为第一页
- [ ] 概览图在第二页，下方有中文要点
- [ ] 表格使用论文原始列名，数据完整不省略
- [ ] 公式框居中，自适应内容宽度
- [ ] 内容区域间距清晰（10-18px）
- [ ] MathJax 公式正确渲染
- [ ] 顶部 badge 区链接回 arXiv abs / html / pdf

## Anti-patterns

- ❌ **Generating only one language** — must produce both English and Chinese slides
- ❌ **Generating Chinese first** — English slides must be generated first
- ❌ **Two-column layouts** — never use `flexbox` `.cols` for side-by-side content; stack vertically
- ❌ **Abbreviated column names** — use "test-clean" not "cl", "Edit Level" not "Edit"
- ❌ **Omitting table values** — include all data from original paper tables
- ❌ **Separate TL;DR slide** — merge into title slide
- ❌ Pasting raw abstract without analysis — must provide synthesis and interpretation
- ❌ **Skipping TeX source** for arXiv papers — TeX has the richest data
- ❌ Skipping HTML and going straight to PDF — information loss is severe
- ❌ Figures/tables without explanation — every visual must have a caption/explanation
- ❌ Outputting Markdown — must be well-designed HTML
- ❌ Standalone "Key Equations" or "Key Tables" sections — embed where discussed
- ❌ Vague takeaway bullets (e.g., "method is effective") — each must cite figure/table/equation

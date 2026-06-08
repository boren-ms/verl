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

## Design Requirements — Poster-style Layout (Both Languages)

Both English and Chinese reports use the **academic poster** visual style — dense but clear, letting the reader grasp the paper in a single screen.

**Layout skeleton (top to bottom):**

1. **Top Banner**: Dark gradient header; left side shows title (large) + authors/venue; right side has badge area with arXiv ID, year, abs/html/pdf link buttons
2. **TL;DR block**: Gold/warm highlight bar, 1–2 sentences
3. **Multi-column card grid (CSS Grid)**: 2–3 column grid of **cards**, each with a colored header bar + section title with emoji. **First grid row must be hero figures** (1–2 key figures spanning 2 columns)
4. **Subsequent rows**: Cards grouped by topic — key takeaways / contributions / method / main table / ablations / setup / limitations / reviewer notes

**Visual requirements:**

- Font: 16–17px body, 17–18px TL;DR, 13–14px tables/captions, 17px section titles
- Line height: 1.6–1.7
- Card gap: 12–14px; card padding: 12–16px
- Colors: light warm background + white cards + dark-blue/burgundy/olive-green accent colors
- Section titles with emoji prefix (🎯 📊 🔬 ⚙️ ⚠️ 💡 📌 🧪 🏆)
- Contribution tags: pink `[Novel/创新]`, blue `[Claim/论断]`, gold `[Novel+Claim/创新+论断]` with left color bar
- Key equations: light background box + `(N) Name` label; ★ for core equations with red border
- Tables: light-blue header, zebra rows, best values in red bold `class="best"`
- Single-file HTML: zero external dependencies except MathJax CDN; images as external URLs or base64
- **Full-width responsive**: `width: 100%; max-width: 100%` with `clamp(16px, 2.5vw, 40px)` padding — fills any screen width
- Responsive: ≤900px collapses to single column

**Information integrity:**

- **Never omit key numbers** from main/ablation tables for aesthetics
- **Never duplicate** — same info appears only once across sections
- Convert long paragraphs to bullet lists; bold key terms
- Equation explanations: 2–3 lines max; table captions: 3 lines max

---

## Procedure Summary

1. Normalize input → derive `tex_url` / `html_url` / `pdf_url`.
2. **Try TeX source first**; if unavailable, try HTML; if that also fails, fall back to PDF.
3. Read the paper end-to-end; identify sections, figures, tables, key equations.
4. Convert TeX figures to PNG (`pdftoppm` / ImageMagick); convert TeX tables to HTML.
5. **Generate English report first** (`_en.html`): draft all sections in English using the poster layout.
6. **Generate Chinese report second** (`_zh.html`): reuse the same figures/tables/equations, write all sections in Chinese.
7. Save both files to `paper_notes/`.
8. Report both output paths and which source was used (TeX / HTML / PDF).

## Quality Checklist

### English Report (`_en.html`)
- [ ] All body text in English
- [ ] Poster-style multi-column card layout (Banner + TL;DR + Grid Cards)
- [ ] Overview figure at top (hero position in first grid row) with explanatory paragraph
- [ ] Other figures placed inline in relevant section cards
- [ ] Key Takeaways each reference specific figure/table/equation numbers; tables embedded as evidence
- [ ] Core Contributions use `[Novel]/[Claim]/[Novel+Claim]` tags; no duplication with other sections
- [ ] Tables embedded under takeaways/results, not in standalone section; key numbers preserved
- [ ] Equations inline in method only when necessary
- [ ] HTML renders correctly with MathJax, collapses to single column at ≤900px
- [ ] Top badge area links to arXiv abs / html / pdf

### Chinese Report (`_zh.html`)
- [ ] 所有正文为中文；英文术语首次出现时给出中译
- [ ] 海报式多列卡片布局（Banner + TL;DR + Grid Cards）
- [ ] 概览图置顶（位于卡片网格第一行 hero 位），并有中文导读
- [ ] 其它图分散到相关章节卡片就近放置
- [ ] 核心要点每条都引用了具体的图/表/公式编号；适合用数字说明的要点就近嵌入表格
- [ ] 核心贡献用 `[创新]/[论断]/[创新+论断]` 标签，未与其它章节重复
- [ ] 表格作为论据嵌入到要点/结果之下；关键数字未省略
- [ ] 公式仅在方法叙述需要时就近嵌入
- [ ] HTML 单文件可直接打开，公式正确渲染，≤900px 自动单列
- [ ] 顶部 badge 区链接回 arXiv abs / html / pdf

## Anti-patterns

- ❌ **Generating only one language** — must produce both English and Chinese reports
- ❌ **Generating Chinese first** — English report must be generated first
- ❌ Pasting raw abstract without analysis — must provide synthesis and interpretation
- ❌ **Skipping TeX source** for arXiv papers — TeX has the richest data
- ❌ Skipping HTML and going straight to PDF — information loss is severe
- ❌ Figures/tables without explanation — every visual must have a caption/explanation
- ❌ Outputting Markdown — must be well-designed HTML
- ❌ Multiple output files per language — single HTML file per language (images external or base64)
- ❌ Dumping all figures into a "Key Figures" section — place inline where discussed
- ❌ Standalone "Key Equations" or "Key Tables" sections — embed where needed
- ❌ Separate "Innovations" and "Claims" sections with duplicated content — merge into one "Core Contributions" section
- ❌ Vague takeaway bullets (e.g., "method is effective") — each must cite figure/table/equation

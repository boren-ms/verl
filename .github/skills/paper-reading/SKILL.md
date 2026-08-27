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

After converting/copying all figures to `_assets/`, **trim whitespace borders** so figures render as large as possible in slides:
```bash
cd paper_notes/<short-name>_assets && for f in *.png; do convert "$f" -trim +repage "$f"; done
```

### Step 3 — Fall back to HTML

If TeX source is unavailable, fetch `html_url`. Record figure URLs, table HTML, section structure, and key equations.

### Step 4 — Fall back to PDF

Only if both TeX and HTML are unavailable:
1. Download: `curl -sSL "$pdf_url" -o /tmp/paper_<id>.pdf`
2. Text: `pdftotext -layout /tmp/paper_<id>.pdf -`
3. Figures: `pdfimages -all` or `pdftoppm -png -r 110`
4. Embed figures as base64 or write to `paper_notes/<short-name>_assets/` if HTML exceeds ~20 MB

Tell the user which source was used (TeX / HTML / PDF).

---

## Output: Two HTML Slide Decks

Write **two** self-contained `.html` files to `paper_notes/`:

```
paper_notes/<short-name>_en.html    ← English (generated first)
paper_notes/<short-name>_zh.html    ← Chinese (generated second)
paper_notes/<short-name>_assets/    ← Extracted figures (shared)
```

**Naming convention:** Use a short, descriptive kebab-case name derived from the paper title — not just the arXiv ID. Examples:
- `rlbr-contextual-biasing` (not `2601.13409`)
- `on-policy-distillation` (not `2601.18734`)
- `grpo-math-reasoning` (not `2402.03300`)

Keep it to 2–4 words that capture the paper's core topic. This makes files easier to find and distinguish.

The `paper_notes/` directory is git-ignored. Create it if absent.

Both decks share the same figures. Both use the reveal.js slide format described below.

---

## Part A — English Slides (`_en.html`)

Generate this deck **first**. All text in English.

### Slide Structure (English)

The deck follows a narrative arc: **hook → gap → idea → mechanics → evidence → dissection → takeaways**. Slides are built around figures, formulas, and tables — visual elements are the primary content, with compact bullets explaining them.

Each slide is a `<section>` inside reveal.js. Use **vertical layout** — content stacked top-to-bottom within each slide (no side-by-side two-column layouts).

> **Core principle — Visual-centric, compact bullets:**
> Slides are built **around** figures, formulas, and tables — these are the primary content. Bullets **surround** them (above/below) to explain, interpret, or annotate. Keep bullet text **short and compact**: one line per bullet, no full sentences when a phrase suffices.
>
> - **Figure/formula/table first:** Place the visual element prominently; bullets explain it.
> - **Maximize figure usage:** Every figure extracted from the paper should appear on at least one slide. If a figure relates to a slide's topic, include it — even on Detail slides that also have equations. Prefer showing a figure over leaving a text-only slide.
> - **Compact bullets:** Use terse, information-dense phrasing — e.g., `$\lambda\!=\!5$ → biasing errors weighted 6× vs general` instead of a full sentence.
> - **Full-width tables:** Use original column names from the paper source (e.g., "test-clean" not "cl"). Include all data columns and values — never abbreviate.
> - **Centered formulas:** Equation boxes use `width: fit-content; margin: auto` to shrink to content and center.
> - **Spacing:** Leave clear vertical gaps between sections (headings, equations, tables, bullets).

**Language switch link** — visible on **every** slide (not just the title):
- Place `<div class="lang-switch">` inside `<div class="reveal">` but **before** `<div class="slides">` so it persists across all slides
- EN deck: `<div class="lang-switch"><a href="<short-name>_zh.html">🌐 中文版</a></div>`
- ZH deck: `<div class="lang-switch"><a href="<short-name>_en.html">🌐 English</a></div>`

**Slide 1 — Title & Highlight** (combined on one slide):
- Light background (`#f8fafc`) — clean, readable, professional
- Title, authors, affiliations, venue, arXiv links as colored badges
- Leave a clear vertical gap (margin/spacer) between the title/author/badges section and the TL;DR box — they should feel like two distinct zones
- TL;DR box at bottom with left blue accent border: Problem → Proposal → Result (3 lines, generous `line-height: 1.85`, `padding: 18px 24px`)
- Use color highlights in TL;DR: `.hl-red` for problem, `.hl-orange` for method, `.hl-green` for results
- This is the "hook" — the audience should immediately know what the paper achieved

**Slide 2 — Problem & Motivation:**
- 4–6 bullet points explaining what problem exists and why this work is needed
- Reference specific numbers from the paper's tables to quantify the gap
- Focus on: what fails today, why it matters, what metric captures the gap

**Slide 3 — Solution (High-Level):**
- Paper's most representative overview figure displayed prominently (centered)
- 3–4 compact bullets below stating the core idea / approach at a high level
- No equations yet — this is the "big picture" slide
- Answer: "What is the key insight?" and "How does it differ from prior work?"

**Slides 4–6 — Details** (one per key technique, typically 1–3 slides):
- Each slide: equation box(es) or figure at top → compact bullets below explaining it
- **Include the relevant figure on every detail slide that has one** — if the paper has a figure illustrating a technique, it must appear on that technique's detail slide alongside its equations
- Key equations in `.eq-box` with label; core equation gets `.eq-star` (gold left border)
- Keep bullets terse: what the equation does, what each term means, key hyperparameter values
- Figures with `<figcaption>` referencing figure number
- Adjust the number of detail slides to match the paper's complexity (1 for simple papers, 3 for multi-component methods)

**Slide 7 — Results:**
- Full results table with original column names and all values from paper
- Table caption + 2–3 compact takeaway bullets citing specific numbers
- Every bullet references a concrete number or comparison from the table

**Slide 8+ — Ablation** (one slide per table/experiment):
- **One topic per slide**: if the paper has multiple ablation tables (e.g., Table 2 for hyperparameter sweep, Table 3 for strategy comparison), put each on its **own slide** with a descriptive subtitle (e.g., "Ablation: Biasing Weight λ", "Ablation: RLBR Strategy")
- **Same rule for figures**: if an ablation slide would contain multiple figures covering different topics (e.g., model scaling vs approximation quality), split them into separate slides. Only combine figures on the same slide when they support the same statement.
- Each slide: table or figure with caption → 2–3 compact takeaway bullets below
- If the paper has no ablation, merge insights into the Results slide and skip this

**Contributions slide:**
- Tagged list with `[Novel]` / `[Claim]` / `[Novel+Claim]`
  - Each with left border accent via `.insights li` style
- This slide answers: what are the paper's key contributions?

**Limitations & Future Work slide** (separate from Contributions):
- 3–5 bullet points covering stated limitations + your critical observations
- Brief "Future" note at the end
- This slide answers: what remains open or unverified?

### Optional Slides (English)

Include only if the paper warrants:
- **Context (Related Work)** — Include only when prior art is essential to understanding the contribution.
- **Reproduction Notes** — Key hyperparams, data scale, training recipe (if not covered in Details slides)

---

## Part B — Chinese Slides (`_zh.html`)

Generate this deck **second**, after the English slides. All body text in Chinese (简体中文). English technical terms should be given a Chinese translation on first occurrence.

Use the same slide structure as Part A, translated to Chinese. Same reveal.js framework, same vertical layout rules, same figure/table/equation placement.

### Slide Structure (Chinese)

中文幻灯片遵循与英文相同的叙事弧线：**引子 → 问题 → 方案 → 细节 → 结果 → 消融 → 总结**。围绕图/公式/表组织内容，要点简洁。

> **排版总原则：**
> - **图/公式/表为主，要点为辅：** 视觉元素居上，简洁要点在下解释。
> - **垂直堆叠：** 不使用双栏布局。
> - **完整表格：** 使用论文原始列名，包含所有数据列和数值。
> - **公式居中：** 公式框自适应内容宽度并居中显示。

**幻灯片 1 — 标题与亮点**（合并为一页）:
- 浅色背景（`#f8fafc`），整洁专业
- 英文标题 + 中文副标题，作者，机构，会议，arXiv 彩色链接徽章
- 标题/作者区与亮点信息框之间留出明显垂直间距（margin/spacer），形成两个独立区域
- 亮点信息框：左侧蓝色边框，问题 → 方案 → 结果（3 行，宽行高 `1.85`，大内边距 `18px 24px`）
- 使用颜色高亮：`.hl-red` 标注问题，`.hl-orange` 标注方法，`.hl-green` 标注结果

**幻灯片 2 — 问题与动机**:
- 4–6 个要点，用具体数字量化当前方法的不足

**幻灯片 3 — 方案（高层概览）**:
- 概览图居中，下方 2–4 条要点说明核心思路
- 不涉及公式——先讲"做什么"再讲"怎么做"

**幻灯片 4–6 — 细节**（每项关键技术一页，1–3 页）:
- 公式/图在上，简洁要点在下
- **每页细节如有对应图，必须包含该图** — 图与公式可同时出现在一页上
- 包含实现细节：超参数、训练配置、架构选择

**幻灯片 7 — 结果**:
- 完整结果表，使用论文原始列名和所有数值

**幻灯片 8+ — 消融实验**（每张表/实验独立一页）:
- **每页只讲一个主题**：若论文有多张消融表（如表 2 超参搜索、表 3 策略对比），每张表单独一页，标题加描述性副标题
- **图片同理**：若一页包含多张讲述不同主题的图（如模型缩放 vs 近似质量），拆分为独立幻灯片。仅当多图支持同一论点时才可合并
- 每页：表格或图片 + 标题 → 下方 2–3 条简洁要点

**贡献页**（独立一页）:
- 贡献列表，标签 `[创新]` / `[论断]` / `[创新+论断]`

**局限与未来工作页**（独立一页，与贡献分开）:
- 3–5 条要点：论文明确局限 + 审视性观察 + 未来方向

### Optional Slides (Chinese)

- **背景（相关工作）** — 仅在前人工作对理解贡献至关重要时才包含
- **复现要点** — 关键超参、数据规模、训练配方（若细节页未涵盖）

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
// Keep lang-switch href in sync with current slide
var langLink = document.querySelector('.lang-switch a');
if (langLink) {
  var base = langLink.getAttribute('href').split('#')[0];
  function syncLang() { langLink.href = base + '#/' + Reveal.getIndices().h; }
  Reveal.on('slidechanged', syncLang);
  syncLang();
}
```

### CSS Design System

```css
/* Base */
.reveal { font-size: 24px; }
.reveal .slides section { overflow: hidden; padding: 20px 30px; }

/* Headings — generous bottom margin for spacing */
.reveal h2 { font-size: 1.15em; border-bottom: 2px solid #1a6fb5; margin-bottom: 18px; }

/* Content — unified 0.68em for all body text (bullets, tables, paragraphs) */
.reveal ul, .reveal ol { font-size: 0.68em; line-height: 1.45; margin: 10px 0; }
.reveal p { font-size: 0.68em; margin-bottom: 8px; }
.reveal figure { margin: 0 0 12px 0; }
.reveal figcaption { font-size: 0.58em; }

/* Tables — same 0.68em as body text for visual consistency */
.reveal table { font-size: 0.68em; width: 100%; white-space: nowrap; margin: 10px auto 14px; }
.reveal table th { background: #dbeafe; }
.reveal table .best { color: #be123c; font-weight: 700; }
.table-caption { font-size: 0.62em; margin-bottom: 14px; }

/* Equation boxes — fit content width, centered, with spacing */
.eq-box { width: fit-content; max-width: 100%; margin: 10px auto;
           font-size: 0.72em; overflow-x: auto; text-align: center; }
.eq-box.eq-star { border-left: 4px solid #d97706; background: #fffbeb; }

/* Tags for contributions */
.tag { font-size: 0.52em; font-weight: 700; padding: 1px 7px; border-radius: 10px; }
.tag-new { background: #fce7f3; color: #9d174d; }
.tag-claim { background: #dbeafe; color: #1e40af; }
.tag-both { background: #fef3c7; color: #92400e; }

/* Title slide — light theme */
.title-slide { background: #f8fafc; }
.title-slide h1 { color: #0f172a; border: none; text-align: center; }
.title-slide .meta { color: #475569; text-align: center; }
.title-slide .badges a { color: #1e40af; background: #dbeafe; border: 1px solid #93c5fd; }
/* Language switch link — visible on EVERY slide; placed inside .reveal but outside .slides */
.lang-switch { position: absolute; top: 18px; right: 30px; font-size: 12px; z-index: 100; }
.lang-switch a { color: #1e40af; text-decoration: none; background: #dbeafe;
                  border: 1px solid #93c5fd; padding: 3px 10px; border-radius: 12px; }
/* TL;DR box — left-accent, generous padding, wide row spacing, separated from title */
.tldr-box { background: #f0f9ff; border-left: 4px solid #1a6fb5;
            padding: 18px 24px; margin-top: 72px; font-size: 0.68em;
            line-height: 3.7; text-align: left; }
.tldr-box strong { color: #1a6fb5; }

/* Color highlights — used on every slide to mark key content */
.hl-red { color: #dc2626; font-weight: 700; }    /* problems, failures, regressions */
.hl-green { color: #16a34a; font-weight: 700; }  /* improvements, positive results */
.hl-orange { color: #d97706; font-weight: 700; } /* methods, innovations, key hyperparams */
```

### Color Highlighting

Use `<span class="hl-red|hl-green|hl-orange">...</span>` on **every slide** to draw attention to key content:

| Class | Color | Use for |
|-------|-------|---------|
| `.hl-red` | Red (#dc2626) | Problems, failures, regressions, limitations, degraded metrics |
| `.hl-green` | Green (#16a34a) | Improvements, positive results, best numbers, gains |
| `.hl-orange` | Orange (#d97706) | Proposed methods, innovations, key hyperparameters, future directions |

**Guidelines:**
- Every slide should have at least 1–2 highlighted spans — not just the TL;DR box.
- Highlight specific numbers, method names, and key phrases — not entire sentences.
- On Problem slides: red for failure metrics, orange for unexplored opportunities.
- On Solution/Detail slides: orange for method names and hyperparams, green for benefits.
- On Results slides: green for best numbers and improvements, red for regressions.
- On Ablation slides: green for best configs, red for worst/degraded configs.
- On Conclusion slides: green for claimed gains, red for limitations.

### Key Layout Rules

1. **Vertical stacking only** — no two-column `flexbox` / `.cols` layouts. Content flows top-to-bottom: figure/equation → bullets.
2. **Title + TL;DR combined** on slide 1 (not separate slides).
3. **One topic per slide** — never combine two distinct topics (e.g., two ablation tables, or contributions + limitations, or figures about different experiments) on the same slide. Split into separate slides. Only combine multiple figures/tables when they support the same statement.
4. **Context (Related Work) is optional** — include only when prior art is essential to understanding the contribution.
5. **Tables use original names** from the paper source — never abbreviate column headers (e.g., use "test-clean" not "cl", "Edit Level" not "Edit", "Reference Aware" not "Ref").
4. **Include all table values** from original paper. If paper shows BWER with (WER/UWER), include all three.
5. **Equation boxes centered** with `width: fit-content; margin: auto`.
6. **Clear spacing** between content sections — 10-18px margins between headings, equations, tables, and bullet lists.
7. **Unified font size** — tables, bullets, and paragraphs all use **0.68em** so text and data share the same visual weight on every slide.
8. **Section titles with emoji** prefix: 🎯 📊 🔬 ⚙️ ⚠️ 💡 📌 🧪
9. **Images**: external files in `_assets/` directory, `max-height` constrained (200-280px).
    - **Use every extracted figure** on at least one slide. Do not leave figures unused in `_assets/` — each one should appear where it is most relevant.
10. **Chinese slides** use Chinese font stack: `'PingFang SC', 'Microsoft YaHei', ...`
10. **Horizontal navigation** (left/right arrows) — slides are siblings, not nested.
11. **Compact bullet style** — terse, information-dense phrasing. One line per bullet when possible. Use `\!=\!` for tight spacing in inline math.

### MathJax Formula Rendering Best Practices

Formulas must render correctly in the browser. Follow these rules:

1. **HTML-escape `<` in math:** Inside `$$...$$` or `$...$`, use `&lt;` for the `<` symbol (e.g., `o_{i,&lt;t}`) since the browser parses HTML before MathJax processes the math.
2. **Use `\!` for negative thin space** (e.g., `1\!-\!\epsilon`) — MathJax 3 supports this.
3. **`\text{}` and `\mathrm{}`** both work in MathJax 3 for roman-text inside math.
4. **`\mathcal{}` renders correctly** — use for calligraphic letters like `\mathcal{ED}`, `\mathcal{J}`.
5. **Display math `$$...$$`** inside `.eq-box` divs — the div provides styling; the `$$` triggers MathJax block rendering.
6. **Re-render on slide change** — the `slidechanged` event handler calls `MathJax.typesetPromise()` to ensure formulas render when navigating.
7. **Avoid `\begin{align}` inside `$$`** — use `\begin{aligned}` (the `*`-free environment) inside `$$...$$` for multi-line aligned equations.
8. **Test complex formulas** — if a formula has subscripts with `<`, fractions with `\frac`, or calligraphic fonts, verify they render by checking the HTML in a browser.

---

## Procedure Summary

1. Normalize input → derive `tex_url` / `html_url` / `pdf_url`.
2. **Try TeX source first**; if unavailable, try HTML; if that also fails, fall back to PDF.
3. Read the paper end-to-end; identify sections, figures, tables, key equations.
4. Convert TeX figures to PNG (`pdftoppm` / ImageMagick); save to `paper_notes/<short-name>_assets/`.
5. **Generate English slides first** (`_en.html`): ~10–12 reveal.js slides following the narrative arc (Title & Highlight → Problem & Motivation → Solution → Details → Results → Ablation (one slide per table) → Contributions → Limitations & Future). Optionally add Context (Related Work) if essential.
6. **Generate Chinese slides second** (`_zh.html`): same structure, all text in Chinese.
7. Save both files to `paper_notes/`.
8. Report both output paths and which source was used (TeX / HTML / PDF).

## Quality Checklist

### English Slides (`_en.html`)
- [ ] All body text in English
- [ ] reveal.js slide deck with horizontal navigation
- [ ] Vertical content layout within each slide (no two-column flexbox)
- [ ] Slide 1: Title & Highlight combined (not separate slides)
- [ ] Slide 2: Problem & Motivation with specific numbers quantifying the gap
- [ ] Slide 3: Solution overview figure with high-level bullets (no equations yet)
- [ ] Slides 4–6: Details with equations, figures, and compact explanatory bullets
- [ ] Every extracted figure from the paper appears on at least one slide
- [ ] Tables use original column names from paper — no abbreviations
- [ ] All table values from paper preserved (never omit data)
- [ ] Equation boxes centered with `width: fit-content`
- [ ] Clear spacing between content sections (10-18px margins)
- [ ] Color highlights (red/green/orange) on every slide — key numbers, methods, failures marked
- [ ] Ablation: each table/experiment on its own slide with descriptive subtitle
- [ ] Contributions slide: tagged list ([Novel] / [Claim] / [Novel+Claim])
- [ ] Limitations & Future Work: separate slide from Contributions
- [ ] MathJax renders correctly (HTML-escaped `<` in subscripts, `\mathcal` works)
- [ ] Re-renders on slide change via `slidechanged` event
- [ ] Badge links to arXiv abs / html / pdf on title slide
- [ ] Language switch link on title slide (top-right, links to `_zh.html`)

### Chinese Slides (`_zh.html`)
- [ ] 所有正文为中文；英文术语首次出现时给出中译
- [ ] reveal.js 幻灯片，水平导航
- [ ] 每页内容垂直堆叠（不使用双栏布局）
- [ ] 幻灯片 1：标题与亮点合并为一页
- [ ] 幻灯片 2：问题与动机，用具体数字量化差距
- [ ] 幻灯片 3：方案概览图，下方有中文要点
- [ ] 所有提取的图均出现在至少一页幻灯片中
- [ ] 表格使用论文原始列名，数据完整不省略
- [ ] 公式框居中，自适应内容宽度
- [ ] 内容区域间距清晰（10-18px）
- [ ] 每页均有颜色高亮（红/绿/橙）标注关键数字、方法、问题
- [ ] 消融实验：每张表/实验独立一页，标题含描述性副标题
- [ ] 贡献页：独立一页，带标签列表（[创新] / [论断] / [创新+论断]）
- [ ] 局限与未来工作：独立一页，与贡献分开
- [ ] MathJax 公式正确渲染（下标中 `<` 已 HTML 转义）
- [ ] 顶部 badge 区链接回 arXiv abs / html / pdf
- [ ] 标题页右上角有语言切换链接（指向 `_en.html`）

## Anti-patterns

- ❌ **Generating only one language** — must produce both English and Chinese slides
- ❌ **Generating Chinese first** — English slides must be generated first
- ❌ **Two-column layouts** — never use `flexbox` `.cols` for side-by-side content; stack vertically
- ❌ **Abbreviated column names** — use "test-clean" not "cl", "Edit Level" not "Edit"
- ❌ **Omitting table values** — include all data from original paper tables
- ❌ **Separate TL;DR slide** — merge into title slide
- ❌ **Skipping Related Work** — Context slide is optional; include only when it helps tell the story
- ❌ **Verbose bullet text** — keep bullets compact and information-dense; one line per bullet
- ❌ **Equations on the Solution slide** — slide 3 is high-level only; equations go in Details slides
- ❌ **Combining contributions and limitations on one slide** — they should be separate slides
- ❌ **Combining figures about different topics** on one slide — split into separate slides; only combine when figures support the same statement
- ❌ Pasting raw abstract without analysis — must provide synthesis and interpretation
- ❌ **Skipping TeX source** for arXiv papers — TeX has the richest data
- ❌ Skipping HTML and going straight to PDF — information loss is severe
- ❌ Figures/tables without explanation — every visual must have a caption/explanation
- ❌ **Omitting available figures** — if a figure was extracted to `_assets/`, it must appear on a slide; do not leave figures unused
- ❌ Outputting Markdown — must be well-designed HTML
- ❌ Standalone "Key Equations" or "Key Tables" sections — embed where discussed
- ❌ Vague takeaway bullets (e.g., "method is effective") — each must cite figure/table/equation
- ❌ **No color highlights** — every slide must use `.hl-red` / `.hl-green` / `.hl-orange` to mark key content

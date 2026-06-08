---
name: paper-reading
description: "Read an academic paper (arXiv or other URL/PDF) and produce TWO well-designed HTML reports: first an English report, then a Chinese (简体中文) report. Both use a poster-style layout with figures, tables, and formulas. Use when: read paper, paper reading, paper summary, paper report, 读论文, 论文精读, summarize arxiv paper, 中文论文解读, generate paper HTML report, arxiv 2xxx.xxxxx, paper walkthrough, 论文笔记."
argument-hint: "arXiv abs/html/pdf URL, arXiv ID (e.g. 2405.22263), or local PDF path"
---

# Paper Reading (Dual-Language HTML Reports)

Produce **two** self-contained, well-designed HTML reports for an academic paper:

1. **English report** (`_en.html`) — generated **first**
2. **Chinese report** (`_zh.html`) — generated **second**, using the same extracted data

Both reports use an academic-poster visual style with inline figures, tables, and MathJax equations.

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

## Output: Two HTML Reports

Write **two** self-contained `.html` files to `paper_notes/`:

```
paper_notes/<arxiv_id_or_slug>_en.html    ← English (generated first)
paper_notes/<arxiv_id_or_slug>_zh.html    ← Chinese (generated second)
```

The `paper_notes/` directory is git-ignored. Create it if absent.

Both reports share the same figures and source data. Both use the poster-style layout described below.

---

## Part A — English Report (`_en.html`)

Generate this report **first**. All text in English.

### Required Sections (English)

> **Layout principles:**
> - **Figure-first:** The paper's overview/method diagram (typically Fig. 1 or 2) must appear at the **top** of the report (right after TL;DR) as the visual entry point.
> - **Explain via figure:** Key Takeaways and Method sections must **explicitly reference figure numbers** (e.g., "see Fig. 1 ②").
> - **No duplication:** Innovations and Claims are **merged** into a single "Core Contributions" section. Each item is tagged and written only once.

1. **Paper Info** — Title, authors, affiliations, arXiv ID, publication date, links (abs/html/pdf)
2. **TL;DR** — 1–2 sentence summary: what was done + key result
3. **Overview Figure** — The paper's most representative figure displayed prominently; with a short paragraph explaining what it shows and where the core idea sits in it
4. **Key Takeaways** — 3–6 bullet points; **each must reference a specific figure/table/equation number** and preferably include an inline table as evidence when numeric comparison is relevant
5. **Core Contributions** — Merged innovations + claims, each prefixed with a tag:
   - `[Novel]` — methodological/algorithmic novelty (vs. prior work)
   - `[Claim]` — experimentally-supported conclusion (must cite §/Fig./Table)
   - `[Novel+Claim]` — both novel and experimentally validated
6. **Method** — Describe the method flow in English; **each sub-step must reference its corresponding figure**; include key equations inline with MathJax (`$$ ... $$`) and 1–2 line explanations. Do NOT create a separate "Key Equations" section.
7. **Results** — Summarize main numbers and comparisons; embed the primary results table inline with English captions
8. **Limitations & Future Work** — Authors' stated limitations + your critical observations
9. **Reviewer Notes** — Applicability, reproducibility assessment, relation to current work

> **Figure/Table/Equation placement rules:**
> - Figures, tables, and equations are **never in standalone sections**
> - Figures: overview figure at top; others placed inline in the section that discusses them
> - Tables: embedded as evidence under Key Takeaways or Results, each with a 1-line caption explaining its point
> - Equations: only when necessary for understanding the method, placed inline; omit if not essential

### Optional Sections (English)

Include only if the paper warrants or user requests deeper coverage:
- **Reproduction Notes / Pseudo-code** — Key hyperparams, data scale, training recipe
- **Related Work Comparison** — Side-by-side comparison table with closest baselines
- **Ablation Analysis** — Per-component ablation interpretation
- **Datasets & Benchmarks** — List of data/metrics used

---

## Part B — Chinese Report (`_zh.html`)

Generate this report **second**, after the English report. All body text in Chinese (简体中文). English technical terms should be given a Chinese translation on first occurrence.

Use [report_template.html](./assets/report_template.html) as reference (but apply the poster-style layout below).

### Required Sections (in Chinese)

> **排版总原则：**
> - **图优先 (figure-first)：** 概览图必须放到报告**最顶部**（紧跟 TL;DR 之后）
> - **以图说事 (explain-via-figure)：** 各章节要**显式引用图编号**
> - **不重复 (no-duplication)：** 「创新点」与「主要论断」**合并为一节**「核心贡献」

1. **论文信息 (Paper Info)** — 标题(原文+中译)、作者、机构、arXiv ID、发表时间、原文链接(abs/html/pdf)
2. **TL;DR 一句话总结** — 1–2 句中文概括
3. **🖼️ 概览图 (Overview Figure)** — **置顶**显示论文最能代表全局思路的一张图；配中文导读
4. **核心要点 (Key Takeaways)** — 3–6 条要点；**每条挂接到具体的图/表/公式编号**，适合用数字对比说明时**就近嵌入表格**
5. **核心贡献 (Contributions)** — **合并"创新点"+"主要论断"**：
   - `[创新]` 方法/算法/工程上的新颖之处
   - `[论断]` 实验证据声明的结论（必附 §/Fig./Table 出处）
   - `[创新+论断]` 既是新方法又被实验验证
6. **方法 (Method)** — 中文叙述方法流程；**每个子步骤引用对应图编号**；必要公式就近嵌入
7. **实验结果 (Results)** — 中文小结主要数字与对比；就近嵌入主结果表
8. **局限与未来工作 (Limitations & Future Work)** — 作者承认的局限 + 批判性思考
9. **个人评价 (Reviewer Notes)** — 适用场景、是否值得复现、与当前工作的关联

> **关于图/表/公式的放置：**
> - 图、表、公式都**不单独成节**
> - 图：概览图置顶，其它就近放到相关章节
> - 表：作为要点/结果的论据嵌入，配中文解读
> - 公式：仅在方法叙述需要时就近嵌入

### Optional Sections (Chinese)

- **复现要点 / 伪代码** — 关键超参、数据规模、训练配方
- **相关工作对比** — 与最相近基线的对比表
- **消融实验解读** — 逐项消融的中文解释
- **数据集与评测** — 数据/指标清单

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

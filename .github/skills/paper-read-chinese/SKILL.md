---
name: paper-read-chinese
description: "Read an academic paper (arXiv or other URL/PDF) and produce a well-designed Chinese HTML reading report covering key takeaways (核心要点), innovations (创新点), claims (主要论断), with preserved figures, tables, and formulas — each accompanied by a Chinese explanation. Use when: 读论文, paper reading in Chinese, 论文精读, summarize arxiv paper in Chinese, 中文论文解读, generate paper HTML report, arxiv 2xxx.xxxxx, paper walkthrough, 论文笔记."
argument-hint: "arXiv abs/html/pdf URL, arXiv ID (e.g. 2405.22263), or local PDF path"
---

# Paper Reading (Chinese HTML Report)

Produce a self-contained, well-designed HTML report **written in Chinese (简体中文)** that walks the reader through an academic paper, preserving its figures, tables, and key formulas with inline explanations.

## When to Use

- User gives an arXiv URL / ID / PDF and asks for a 中文 reading note or 解读
- User says "读这篇论文" / "paper reading" / "用中文总结" / "生成 HTML 报告"
- Producing a shareable single-file HTML that summarizes a paper for a Chinese-speaking audience

## Inputs

One of:
1. arXiv URL (`abs`, `html`, or `pdf` form), e.g. `https://arxiv.org/abs/2405.22263`
2. Bare arXiv ID, e.g. `2405.22263` (optionally with `v1`, `v2`, ...)
3. A direct paper URL on another site, or a local PDF path

If none provided, ask the user for the paper URL/ID.

## Source Fetching Strategy (HTML-first, PDF-fallback)

**Always try the HTML version first.** It preserves structure, math (MathJax), figure URLs, and tables far better than PDF text extraction.

### Step 1 — Normalize the input

From any arXiv input, derive:
- `abs_url`  = `https://arxiv.org/abs/<id>`
- `html_url` = `https://arxiv.org/html/<id>` (also try `https://arxiv.org/html/<id>v1`, `v2`, … if bare id fails)
- `pdf_url`  = `https://arxiv.org/pdf/<id>`

For non-arXiv URLs, try the given URL as-is; if it looks like a PDF, skip to PDF fallback.

### Step 2 — Try HTML

Fetch `html_url` (try `fetch_webpage`, then `open_browser_page` + `read_page` if blocked).

A successful HTML fetch should contain section headings, paragraphs, `<figure>`/`<img>` tags, `<table>` tags, and MathJax/LaTeX spans (`\(...\)`, `$$...$$`, or `<math>`).

Record:
- All figure image URLs (resolve to absolute URLs, typically `https://arxiv.org/html/<id>/extracted/...` or `.../x1.png`)
- All table HTML blocks (keep original `<table>` markup)
- Section structure (Abstract, Introduction, Method, Experiments, …)
- Key display equations

### Step 3 — Fallback to PDF

Only if the HTML version is unavailable (404, empty, or clearly not rendered):
1. Download the PDF: `curl -sSL "$pdf_url" -o /tmp/paper_<id>.pdf`
2. Extract text: `pdftotext -layout /tmp/paper_<id>.pdf -` (or `pdfplumber` in Python)
3. Extract figures as images: `pdfimages -all /tmp/paper_<id>.pdf /tmp/paper_<id>_img`, then **embed each figure as a base64 `data:image/...;base64,...` URI** inside the `<img src="...">` so the report stays a single file (matches the HTML-path single-file guarantee).
   - **Auto-switch rule:** if the resulting HTML would exceed ~20 MB (e.g. many large figures), instead write images to a sibling folder `paper_notes/<id>_assets/` and reference them via relative path. Note this in the report header so the user knows the HTML is no longer standalone.
4. Tables from PDF are unreliable — reconstruct the most important 1–3 tables manually from the extracted text and clearly mark them as "重建表格 (reconstructed from PDF)".

Tell the user briefly which path was used (HTML vs PDF fallback).

## Output: HTML Report

Write a **single self-contained `.html` file** (one file, inline CSS, CDN-hosted MathJax) to:

```
paper_notes/<arxiv_id_or_slug>_zh.html
```

The `paper_notes/` directory is git-ignored.

Use [report_template.html](./assets/report_template.html) as the starting point. Fill in the placeholders.

### Required Sections (in Chinese)

1. **论文信息 (Paper Info)** — 标题(原文+中译)、作者、机构、arXiv ID、发表时间、原文链接(abs/html/pdf)
2. **TL;DR 一句话总结** — 1–2 句中文概括
3. **核心要点 (Key Takeaways)** — 3–6 条要点 bullet list
4. **创新点 (Innovations)** — 与已有工作的差异、技术新意，逐条列出
5. **主要论断 (Claims / Contributions)** — 作者明确声明的贡献，逐条列出并标注其在论文中的依据 (e.g. §3.2 / Fig.4 / Table 2)
6. **方法 (Method)** — 用中文叙述方法流程；穿插关键公式与解释（见下）
7. **关键公式 (Key Formulas)** — 提炼 3–8 个核心公式，每个公式：
   - 用 MathJax 渲染 (`$$ ... $$`)
   - 紧跟一段 **中文解释**：符号含义、为什么这样设计、与基线的差别
8. **关键图 (Figures)** — 保留论文图片（直接 `<img src="...">` 指向原 arXiv URL；若 PDF fallback 则 base64 内嵌）
   - 每张图配 **中文图注 + 解释**：这张图想说明什么？x/y 轴含义？关键趋势/对比？
9. **关键表 (Tables)** — 保留原表（HTML 时直接复制 `<table>` 标记并加 class 样式）
   - 每张表配 **中文表注 + 解释**：列含义、最佳结果加粗的指标、可读出的结论
10. **实验结果 (Results)** — 中文小结主要数字与对比
11. **局限与未来工作 (Limitations & Future Work)** — 作者承认的局限 + 你的批判性思考
12. **个人评价 (Reviewer Notes)** — 适用场景、是否值得复现、与当前工作的关联

### Optional Sections (include only if the paper warrants or the user requests deeper coverage)

- **复现要点 / 伪代码 (Reproduction Notes / Pseudo-code)** — 关键超参、数据规模、训练配方，或方法的伪代码
- **相关工作对比 (Related Work Comparison)** — 与最相近基线的并排对比表
- **消融实验解读 (Ablation Analysis)** — 逐项消融的中文解释
- **数据集与评测 (Datasets & Benchmarks)** — 使用的数据/指标清单

控制深度：默认产出上面 1–12 节即可；当用户明确要求「精读 / deep / 复现」或论文方法复杂时，再追加相应可选节。

### Design Requirements

- 语言：全部 **简体中文**；专有名词保留英文并括注中文（如 "RLHF（人类反馈强化学习）"）
- 排版：左右留白舒适、行高 1.7、衬线/无衬线字体搭配（标题 sans-serif，正文 serif 可选）
- 配色：浅色主题、强调色用于章节标题、引用块、公式块
- 公式：MathJax 3 via CDN，支持 `\(...\)` 和 `$$...$$`
- 图片：max-width 100%、居中、下方 `<figcaption>` 含「图 N：原标题 — 中文解释」
- 表格：斑马纹、表头粘性、下方 `<p class="caption">` 含「表 N：原标题 — 中文解释」
- 顶部目录 (TOC) 锚点可跳转
- 单文件：除 MathJax CDN 外不依赖外部资源（图片可外链 arXiv 或内嵌 base64）

## Procedure Summary

1. Normalize input → derive `html_url` / `pdf_url`.
2. Fetch HTML; if fails, fall back to PDF and extract.
3. Read the paper end-to-end; identify sections, figures, tables, key equations.
4. Draft each section in Chinese using the structure above.
5. Copy/refer to figures (URLs) and tables (HTML) verbatim from source; add 中文解释.
6. Render the final HTML from [report_template.html](./assets/report_template.html) and save to `paper_notes/<id>_zh.html`.
7. Report the output path to the user; offer to open it in a browser preview.

## Quality Checklist

- [ ] 所有正文为中文；英文术语首次出现时给出中译
- [ ] 至少保留 2 张原文图，每张都有中文解释（非仅复述图注）
- [ ] 至少保留 1 张原文表，并解读其结论
- [ ] 关键公式 ≥ 3 条，符号含义全部解释
- [ ] HTML 单文件可直接双击打开，公式正确渲染
- [ ] 顶部链接回 arXiv abs / html / pdf 三种入口
- [ ] 创新点与主要论断清晰区分，并标注论文出处

## Anti-patterns

- ❌ 把英文摘要直接机器翻译堆上去 — 必须有提炼与解读
- ❌ 跳过 HTML 直接抓 PDF — 信息损失严重，公式/表格基本不可用
- ❌ 只贴图不解释 — 每张图/表/公式都要配中文 explanation
- ❌ 输出 Markdown 而不是 HTML — 用户明确要求 well-designed HTML
- ❌ 多文件输出 — 必须单文件 HTML，图片外链或 base64 内嵌

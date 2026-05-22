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

> **排版总原则：**
> - **图优先 (figure-first)：** 论文的 "概览图 / 方法总览图"（通常是 Fig. 1 或 Fig. 2）必须放到报告**最顶部**（紧跟 TL;DR 之后），作为「看一眼就懂」的核心入口。其它图按其讲述的故事就近放在相关章节里，**不要把所有图堆到一个 "关键图" 大节末尾**。
> - **以图说事 (explain-via-figure)：** 「核心要点」和「方法」章节要**显式引用图编号**（如「见图 1 ②」「如图 2 中虚线方框所示」），让读者顺着图就能读懂；避免脱离图独立讲故事。
> - **不重复 (no-duplication)：** 「创新点」与「主要论断 / 贡献」**合并为一节**「核心贡献 (Contributions)」。每条只写一次，每条都标注「这是新颖点 / 这是实验论断 / 同时是两者」与论文出处。

1. **论文信息 (Paper Info)** — 标题(原文+中译)、作者、机构、arXiv ID、发表时间、原文链接(abs/html/pdf)
2. **TL;DR 一句话总结** — 1–2 句中文概括
3. **🖼️ 概览图 (Overview Figure)** — **置顶**显示论文中最能代表全局思路的一张图（通常是方法总览图）；配一段「这张图讲了什么 + 论文的核心 idea 在图中的哪个位置」的中文导读
4. **核心要点 (Key Takeaways)** — 3–6 条要点；**每条要点必须挂接到一个具体的图/表/公式编号**，并优先使用「**表格 + 简短解读**」来佐证（例如：「① 提出 X 机制（见图 1 红框） → ② 在 Y 上提升 Z%（见下表 N=100 列）」）。当某条要点适合用数字对比说明时，**直接把对应表格嵌进这条要点下面**，让表服务于要点，不要把表放到独立的「关键表」大节。
5. **核心贡献 (Contributions)** — **合并原"创新点"+"主要论断"**：每条用标签前缀区分性质，避免重复
   - `[创新]` 方法/算法/工程上的新颖之处（与已有工作的差异）
   - `[论断]` 论文用实验证据声明的结论（必附 §/Fig./Table 出处）
   - `[创新+论断]` 既是新方法又被实验验证
6. **方法 (Method)** — 用中文叙述方法流程；**每个方法子步骤都要引用对应图编号**；如方法本身包含必要的公式（如新颖的 loss/reward），就近用 MathJax (`$$ ... $$`) 嵌入并配 1–2 行中文解释。**不要为凑数量而单独开一节列公式**。
7. **实验结果 (Results)** — 中文小结主要数字与对比；如有最关键的主结果表，可在此就近嵌入并配中文表注
8. **局限与未来工作 (Limitations & Future Work)** — 作者承认的局限 + 你的批判性思考
9. **个人评价 (Reviewer Notes)** — 适用场景、是否值得复现、与当前工作的关联

> **关于图/表/公式的放置：**
> - **图、表、公式都不再单独成节**。
> - 图：除「概览图」置顶外，其它图就近放到论证它的章节卡片里。
> - 表：作为「核心要点」与「实验结果」的论据嵌入，每张表配一段「这张表想说明的要点是 …」的中文解读。
> - 公式：仅当对理解方法**确实必要**时才出现，紧贴方法叙述插入；可有可无的公式直接省略。

### Optional Sections (include only if the paper warrants or the user requests deeper coverage)

- **复现要点 / 伪代码 (Reproduction Notes / Pseudo-code)** — 关键超参、数据规模、训练配方，或方法的伪代码
- **相关工作对比 (Related Work Comparison)** — 与最相近基线的并排对比表
- **消融实验解读 (Ablation Analysis)** — 逐项消融的中文解释
- **数据集与评测 (Datasets & Benchmarks)** — 使用的数据/指标清单

控制深度：默认产出上面 1–9 节即可；当用户明确要求「精读 / deep / 复现」或论文方法复杂时，再追加相应可选节。

### Design Requirements — **海报式 (Poster-style) 默认布局**

报告默认采用「**学术海报 (academic poster)**」视觉风格 —— 信息高度密集但视觉清晰，让读者一屏内就能抓到全篇主旨。

**布局骨架（从上到下）：**

1. **顶部 Banner**：深色渐变背景的横幅，左侧显示中文标题（大号）+ 英文原标题（斜体）+ 作者/会议；右侧 badge 区放 arXiv ID、年份、abs/html/pdf 三个链接按钮。
2. **TL;DR 块**：金色/暖色高亮长条，1–2 句把整篇浓缩成「干了啥 + 收益多少」。
3. **多列卡片网格 (CSS Grid)**：用 2–3 列网格把后续所有内容切成 **小卡片 (card)**，每张卡片自带浅色 header bar + 章节标题（如「🎯 核心创新①」「📊 主结果」「🔬 消融」）。**第一行网格必须是 hero figures**（论文最关键的 1–2 张图，跨 2 列大图展示），让读者第一屏看见。
4. **后续行**按主题分组卡片：核心要点 / 核心贡献 / 关键公式 / 方法 / 主表 / 消融 / 实验设置 / 局限 / 个人评价。每张卡片宽度按内容自适应（用 `grid-template-columns: 1fr 1.2fr 1fr` 这类不等宽分配）。

**视觉与紧凑度要求：**

- 字号：正文 16–17px，TL;DR 17–18px，表格 13–14px，图注/表注 13–14px，章节标题 17px —— **以浏览器默认字号偏大一档**，保证投屏/海报场景的可读性
- 行高：1.6–1.7
- 卡片间距：12–14px gap；卡片内 padding 12–16px
- 配色：浅米色背景 + 白色卡片 + 深蓝/酒红/橄榄绿三色 accent（不同卡片用不同 accent 颜色区分主题）
- 章节标题用 emoji 引导（🎯 创新 / 📊 结果 / 🔬 消融 / ⚙️ 方法 / ∑ 公式 / ⚠️ 局限 / 💡 评价 / 📌 要点 / 🧪 设置 / 🏆 贡献），增强扫读
- 「核心贡献」用彩色标签卡片：粉红 `[创新]`、蓝 `[论断]`、金 `[创新+论断]`，每条带左侧色条
- 关键公式：浅色背景框 + 顶部小字 `(N) 公式名` label；★ 标记核心公式并加红色边框
- 表格：表头浅蓝、斑马纹、最佳指标用红色加粗 `class="best"`
- 单文件 HTML：除 MathJax CDN 外零外部依赖；图片可直接 `<img src="https://arxiv.org/html/<id>/x1.png">` 外链或 base64 内嵌
- **全宽自适应**：外层容器使用 `width: 100%; max-width: 100%`（**不要设 1280px 之类的固定上限**），左右 padding 用 `clamp(16px, 2.5vw, 40px)`，让 4K 大屏也能铺满浏览器宽度而不留空白边
- 响应式：≤900px 自动塌缩为单列

**信息正确性与紧凑性的平衡：**

- **绝不为了好看而省略关键数字**：主表、消融表的关键数字必须完整保留（可以用 `WER/BWER` 双栏合并这种紧凑写法）
- **绝不重复**：同一条信息（如「相对 SFT 降低 X%」）只在「核心贡献」或「主结果解读」里出现一次
- 长解释性段落改写成 bullet list；bullet 内用 `<b>` 加粗关键词
- 公式解释控制在 2–3 行；表格 caption 控制在 3 行以内

**实现提示：** 参考 [`assets/report_template.html`](./assets/report_template.html) 中的 poster 骨架（CSS Grid + card + banner + tldr 块）。

### Legacy 长文式布局（仅在用户明确要求时使用）

如果用户说「不要海报、用普通长文式」或「我要打印」，再退回到单列、章节式的传统排版，配色与字号放大。

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
- [ ] **采用海报式多列卡片布局**（Banner + TL;DR + Grid Cards），单屏即可看到全篇要点
- [ ] **概览图置顶**（位于卡片网格第一行 hero 位），并有中文导读说明全局思路
- [ ] **其它图分散到相关章节卡片**就近放置，禁止集中堆到末尾的「关键图」大节
- [ ] **核心要点**每条都引用了具体的图/表/公式编号；适合用数字说明的要点**就近嵌入表格**作为佐证
- [ ] **核心贡献**用 `[创新]/[论断]/[创新+论断]` 标签卡片做单一来源，未与其它章节内容重复
- [ ] 表格作为论据**嵌入到要点/结果**之下，不单独成「关键表」节；关键数字未为美观省略
- [ ] 公式仅在方法叙述需要时就近嵌入，**不为凑数单独成节**
- [ ] HTML 单文件可直接双击打开，公式正确渲染，≤900px 自动单列
- [ ] 顶部 badge 区链接回 arXiv abs / html / pdf 三种入口

## Anti-patterns

- ❌ 把英文摘要直接机器翻译堆上去 — 必须有提炼与解读
- ❌ 跳过 HTML 直接抓 PDF — 信息损失严重，公式/表格基本不可用
- ❌ 只贴图不解释 — 每张图/表/公式都要配中文 explanation
- ❌ 输出 Markdown 而不是 HTML — 用户明确要求 well-designed HTML
- ❌ 多文件输出 — 必须单文件 HTML，图片外链或 base64 内嵌
- ❌ **把所有图集中堆到末尾的「关键图」大节** — 必须就近放在论证它的章节
- ❌ **单独列「关键公式」「关键表」节** — 表服务于要点/结果，公式服务于方法，按需就近嵌入
- ❌ **「创新点」与「主要论断」分两节写、内容互相重复** — 必须合并为单一「核心贡献」节
- ❌ **核心要点写成空泛 bullet（如「方法有效」「实验充分」）** — 每条必须挂接到具体图/表/公式编号

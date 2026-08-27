---
name: code-walkthrough
description: "Generate a self-contained dark-themed single-page HTML doc that explains a code module's workflow with hand-authored inline SVG diagrams (architecture, decision flowcharts, sequence diagrams), explanatory flow-cards, and annotated code snippets. Use when: explain code, code walkthrough, code diagram, explain module, code architecture, code flow, visualize code, 代码讲解, 代码流程图, module walkthrough, class diagram for code."
argument-hint: "File path or module path (e.g. verl/workers/rollout/llm_server.py)"
---

# Code Walkthrough (Dark Single-Page Doc with Hand-Authored SVG Diagrams)

Produce a **self-contained, dark-themed, single-page HTML document** that explains a
code module's architecture, class relationships, and runtime workflow using
**hand-authored inline SVG diagrams**, explanatory **flow-cards**, and **annotated code
snippets**.

The output is **not** a slide deck. It is one vertically scrolling page styled like the
GitHub dark theme — boxes, arrows, decision diamonds, and sequence lifelines are drawn
directly as SVG (no Mermaid, no reveal.js). Reference exemplars:
- `paper_notes/llm_server_loadbalancer_diagram.html` — focused single-component module
  (architecture + decision flowchart + sequence diagram + flow-cards).
- `paper_notes/fully_async_full_architecture_diagram.html` — comprehensive multi-component
  system (9 numbered sections, location pill-badges, stage-cards with pseudo-code, rich
  data-tables, metric stat-cards, footer). See **Optional Advanced Components** below.

## When to Use

- User points to a source file or module and asks "explain this code" / "walkthrough" / "code diagram"
- User wants a visual architecture overview of a module with class/sequence/flowchart diagrams
- User says "draw a diagram for this", "visualize the workflow", "code architecture page"
- Producing a shareable single-file HTML overview of a codebase module

## Inputs

One of:
1. A **file path** relative to the workspace root (e.g., `verl/workers/rollout/llm_server.py`)
2. A **module/package path** (e.g., `verl.workers.rollout`)
3. A **class or function name** with enough context to locate it

If none provided, ask the user for the target file or module.

---

## Procedure

### Step 1 — Read & Analyze the Code

1. Read the target file(s) end-to-end.
2. Identify the **key abstractions**: classes, their public methods, key attributes.
3. Trace the **runtime workflow**: initialization order, data flow, request lifecycle.
4. Identify **external dependencies**: imports, Ray actors, config objects, base classes.
5. Note **design patterns**: factory, proxy, load balancing, observer, etc.

### Step 2 — Plan the Page Sections

Map the code to a top-to-bottom narrative. The page is a single scrolling document with
numbered `<h2>` sections:

| Section | Purpose |
|---------|---------|
| Title + subtitle | Module name (`<h1>`), one-line purpose + key concepts (`.subtitle`) |
| 1. System Architecture | Hand-authored **SVG component diagram** of all classes/actors and how they connect, plus a color **legend** |
| 2. Key Decision / Class Flow | **SVG flowchart** (decision diamonds) for the most important branching logic, or a class-relationship diagram |
| 3. Key Mechanisms | Grid of **flow-cards** — one card per class/mechanism with a short ordered/unordered list |
| 4. Request / Init Sequence | Hand-authored **SVG sequence diagram** (lifelines + numbered arrows) for a typical request or boot-up |
| 5. Key Code | **Annotated code snippets** (≤20 lines each) for the most important methods |
| 6. Summary | Quick-reference **table**: Class / Role / Key Methods |

Adjust the number of sections to the module's complexity (4–7 sections typical). Every
section must contain a diagram, a flow-card grid, a code snippet, or a table — never a
wall of text.

### Step 3 — Author the SVG Diagrams (by hand)

Draw **at least two** diagrams directly as inline `<svg>` — do **not** use Mermaid.

1. **Component / Architecture diagram** — boxes for each class/actor, grouped with
   dashed container rects, connected by curved arrows.
2. **Sequence diagram or decision flowchart** — runtime flow (init, request handling,
   or branching logic).

Additional diagrams as needed (state machine, data pipeline).

**SVG building blocks** (see the CSS/markup section below for the full palette):

- **Canvas**: `<svg viewBox="0 0 W H" xmlns="http://www.w3.org/2000/svg">`. The CSS sets
  `svg { width: 100%; height: auto; }`, so size the diagram via `viewBox`.
- **Boxes**: `<rect x y width height rx fill stroke stroke-width>` with rounded corners
  (`rx="8"`). Add a bold title `<text>` and one or two smaller muted `<text>` sub-lines.
- **Containers/groups**: large dashed rects (`stroke-dasharray="5,3"`) behind related boxes.
- **Decision diamonds**: `<polygon points="cx,top rx,cy cx,bottom lx,cy">`.
- **Lifelines** (sequence diagrams): a header `<rect>`+`<text>` plus a vertical dashed
  `<line stroke-dasharray="3,3">` per participant; horizontal arrows for messages,
  dashed arrows for returns; small rects on lifelines for activation/processing.
- **Arrows**: define reusable markers once in `<defs>`, then reference with `marker-end`:

```html
<defs>
  <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#3fb950"/>
  </marker>
  <marker id="arrowBlue"  markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#58a6ff"/>
  </marker>
  <marker id="arrowRed"   markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#f85149"/>
  </marker>
  <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#d29922"/>
  </marker>
</defs>
```

- **Connectors**: straight `<line>` or curved `<path d="M x1 y1 Q cx cy x2 y2">`. Use
  solid for primary calls, `stroke-dasharray="4,2"` for returns/release/lifecycle.
- **Edge labels**: small `<text>` placed near the midpoint of each arrow.

**Color language** (use consistently across all diagrams + the legend):

| Color | Hex | Use for |
|-------|-----|---------|
| Blue | `#58a6ff` / box `#0d419d` | clients, workers, entry points |
| Green | `#3fb950` / box `#0d2818` | core component, success path, public API |
| Red | `#f85149` / box `#3d1a1a` | servers, replicas, external/remote, error path |
| Orange | `#d29922` / box `#2a1f00` | release, branch selection, trade-offs |
| Purple | `#8957e5` / box `#1a1a2e` | managers, lifecycle, Ray/distributed control |

Mark Ray actors explicitly with a `(Ray Actor)` sub-line in the box.

### Step 4 — Write the HTML Document

Save a single self-contained `.html` file to `paper_notes/` (reuse the project's
existing notes directory):

```
paper_notes/<module-name>_code_walkthrough.html
```

**Naming**: derive from the module — e.g., `llm_server_code_walkthrough.html`.

---

## Page Structure

### Title

- `<h1>` module/system name, centered.
- `.subtitle` paragraph: one line of purpose + the key concepts (e.g.
  "GlobalRequestLoadBalancer · Sticky Sessions · Least-Loaded Routing").

### Section 1 — System Architecture (SVG)

- Hand-authored **SVG component diagram** wrapped in `<div class="diagram">`:
  - One box per class/actor; group related boxes inside dashed container rects.
  - Curved arrows showing the main relationships (creates, calls, routes, returns).
  - Edge labels naming the key methods on each arrow.
- A `.legend` row of colored swatches mapping each color to what it represents.

### Section 2 — Key Decision / Class Flow (SVG)

- **SVG flowchart** of the most important branching logic using decision diamonds
  (`<polygon>`), YES/NO labeled arrows, and terminal rounded boxes — **or** a
  class-relationship diagram if the module is more structural than procedural.

### Section 3 — Key Mechanisms (flow-cards)

- A `<div class="flow-section">` grid of `<div class="flow-card">` cards. A two-column
  grid is expected here (collapses to one column on narrow screens).
- One card per major class/mechanism:
  - `<h3>` card title.
  - A short `<ol>`/`<ul>` (4–6 items) explaining how it works, with inline `<code>` for
    method names, attributes, and config keys.

### Section 4 — Request / Init Sequence (SVG)

- Hand-authored **SVG sequence diagram**:
  - One lifeline per participant (header box + vertical dashed line).
  - Numbered horizontal arrows for calls; dashed arrows for returns.
  - Activation/processing rects on lifelines; optional rotated annotation text for
    state (e.g. counters).
- Use the same arrow-color language as the architecture diagram.

### Section 5 — Key Code (annotated snippets)

- For each of the most important methods, a `<div class="diagram">` containing a
  syntax-highlighted `<pre><code class="language-python">` block (≤20 lines).
  - Trim imports/boilerplate/docstrings; show only essential logic.
  - Add inline `<!-- ← comment -->`-style annotations (as `# ←` comments in the code or
    short `.code-note` lines below) pointing at the key lines.
- Precede or follow each snippet with 2–3 bullets on **why**, not just **what**.

### Section 6 — Summary (table)

- A table with columns: **Class | Role | Key Methods** — one row per class, terse.

---

## Design Requirements — Dark Single-Page HTML

Self-contained: only highlight.js for code is loaded from CDN; everything else
(CSS + SVG) is inline. No reveal.js, no Mermaid.

### Head / CDN

```html
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<!-- Highlight.js for code syntax highlighting (dark theme) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11/languages/python.min.js"></script>
```

```html
<!-- at end of body -->
<script>hljs.highlightAll();</script>
```

### CSS Design System (GitHub dark)

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 40px; }
h1 { text-align: center; margin-bottom: 10px; color: #58a6ff; font-size: 1.8em; }
h2 { color: #79c0ff; margin: 30px 0 15px; font-size: 1.3em;
     border-bottom: 1px solid #21262d; padding-bottom: 8px; }
.subtitle { text-align: center; color: #8b949e; margin-bottom: 40px; }
.container { max-width: 1200px; margin: 0 auto; }
.diagram { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
           padding: 40px; margin: 20px 0; }
svg { width: 100%; height: auto; display: block; }
.flow-section { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }
.flow-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; }
.flow-card h3 { color: #7ee787; margin-bottom: 12px; font-size: 1.1em; }
.flow-card ol, .flow-card ul { padding-left: 20px; line-height: 1.8; }
.flow-card li { color: #c9d1d9; }
.flow-card code, .code-note code { background: #1f2937; padding: 2px 6px;
           border-radius: 4px; color: #f0883e; font-size: 0.9em; }
.legend { display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 8px; color: #8b949e; font-size: 0.9em; }
.legend-color { width: 16px; height: 16px; border-radius: 3px; }
pre { background: #161b22 !important; border: 1px solid #30363d; border-radius: 8px;
      padding: 16px; overflow-x: auto; margin: 12px 0; }
pre code { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85em;
           background: transparent !important; }
.code-note { color: #8b949e; font-size: 0.85em; font-style: italic; margin: 4px 0 12px; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; }
th { background: #161b22; color: #79c0ff; text-align: left; padding: 10px; border: 1px solid #30363d; }
td { padding: 10px; border: 1px solid #30363d; color: #c9d1d9; }
td code { color: #f0883e; }
@media (max-width: 800px) { .flow-section { grid-template-columns: 1fr; } }
```

### Optional Advanced Components

For richer, more comprehensive walkthroughs (large multi-component systems), add these
building blocks — demonstrated in the fuller exemplar
`paper_notes/fully_async_full_architecture_diagram.html`. Use them when they earn their
place; skip them for small modules.

- **Module-path badge in the subtitle** — show where the code lives:
  ```html
  <p class="subtitle">One-line purpose · key concepts<br><code>verl/experimental/fully_async_policy</code></p>
  ```
  `.subtitle code { background:#1f2937; padding:1px 6px; border-radius:4px; color:#f0883e; font-size:0.85em; }`

- **Location / role pill-badges** — tag where a class or step runs (CPU / GPU / Ray / sync).
  Put a `<span class="loc loc-cpu">…</span>` inside a card's `<h3>`:
  ```css
  .loc { display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.7em; margin-left:8px; vertical-align:middle; }
  .loc-cpu  { background:#1a2e4a; color:#79c0ff; border:1px solid #388bfd; }
  .loc-gpu  { background:#2a1f00; color:#e3b341; border:1px solid #9e6a03; }
  .loc-ray  { background:#1a0d2e; color:#bc8cff; border:1px solid #8957e5; }
  .loc-sync { background:#3d1a2e; color:#ff7eb6; border:1px solid #bf4080; }
  ```

- **Stage-cards with a monospace `.fields` box** — a card variant whose body is a
  pseudo-code / data-schema block. `.stage-grid` is the 2-col grid; `.full-width` spans both:
  ```css
  .stage-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0; }
  .stage-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px; position:relative; }
  .stage-card .fields { background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:8px;
                        margin-top:8px; font-family:monospace; font-size:0.75em; line-height:1.5; color:#7ee787; white-space:pre; }
  .full-width { grid-column:1 / -1; }
  @media (max-width: 900px) { .stage-grid { grid-template-columns: 1fr; } }
  ```

- **Rich `.data-table`** — for matrices like operating modes / config keys (hover highlight,
  inline `<code>` cells):
  ```css
  table.data-table { width:100%; border-collapse:collapse; font-size:0.8em; margin:15px 0; }
  table.data-table th, table.data-table td { padding:7px 10px; border:1px solid #30363d; vertical-align:top; }
  table.data-table th { background:#1a2332; color:#79c0ff; text-align:left; }
  table.data-table td { background:#0d1117; }
  table.data-table tr:hover td { background:#11161d; }
  ```

- **Metric / stat cards** — big-number callouts for headline figures (speedup, counts, primitives):
  ```css
  .metric-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:14px; margin:18px 0; }
  .metric { background:#0d1117; border:1px solid #30363d; border-radius:10px; padding:14px 16px; }
  .metric .v { font-size:1.5em; font-weight:700; color:#58a6ff; }
  .metric .l { font-size:0.75em; color:#8b949e; margin-top:2px; }
  ```
  ```html
  <div class="metric"><div class="v">2.35–2.67×</div><div class="l">speedup · 128 GPUs</div></div>
  ```

- **Footer with source citation** — close the page by crediting the source path:
  ```css
  footer { text-align:center; color:#586069; font-size:0.78em; margin-top:40px; }
  footer code { color:#8b949e; }
  ```
  ```html
  <footer>Source: <code>verl/experimental/fully_async_policy</code></footer>
  ```

These cover extra section types a comprehensive walkthrough may add: **Bootstrap /
Initialization**, **Streaming Dataflow (producer/consumer)**, an **asyncio/state-machine**
diagram, a **pipeline with prefetch**, **parameter synchronization**, **operating-modes
matrix**, **key configuration table**, and a **headline-metrics** strip.

### Color Highlighting

Apply the color language on **every diagram** so readers can scan visually:

| Concept | Color (stroke / text) |
|---------|-----------------------|
| Clients, workers, entry points | Blue `#58a6ff` |
| Core component, success, public API | Green `#3fb950` / `#7ee787` |
| Servers, replicas, external/remote, errors | Red `#f85149` / `#ffa198` |
| Release, branch selection, trade-offs | Orange `#d29922` / `#e3b341` |
| Managers, lifecycle, Ray/distributed | Purple `#8957e5` / `#bc8cff` |

### Key Layout Rules

1. **Single scrolling page** — numbered `<h2>` sections inside one `.container`. No slides.
2. **Hand-authored SVG only** — diagrams are inline `<svg>`; never Mermaid.
3. **One topic per section** — one diagram, one card-grid, one snippet cluster, or one table.
4. **Diagrams sit in `.diagram` cards**; size via `viewBox` (CSS scales width to 100%).
5. **Flow-cards use the two-column grid** (`.flow-section`); collapses to one column on
   narrow screens — this 2-column grid is intentional and allowed.
6. **Code snippets ≤ 20 lines** — show only essential logic; trim boilerplate/imports/docstrings.
7. **Reuse arrow markers** via `<defs>` once; reference with `marker-end`.
8. **Section titles** are numbered ("1. System Architecture", "2. …"); an emoji prefix is optional.

---

## Example: `verl/workers/rollout/llm_server.py`

This module's reference output is `paper_notes/llm_server_loadbalancer_diagram.html`.
The skill would produce a page like:

- **Title**: "verl LLM Server Load Balancer Architecture" with subtitle
  "GlobalRequestLoadBalancer · Sticky Sessions · Least-Loaded Routing".
- **1. System Architecture** (SVG): dashed groups for `AgentLoopWorkers` (blue),
  `GlobalRequestLoadBalancer` Ray-actor box (green, with LRU-cache / inflight-counter /
  server-registry internals), and `LLM Server Replicas` (red). Curved green arrows for
  `acquire_server(req_id)`, red arrows for `generate.remote()`, dashed orange for
  `release_server()`, plus a purple `LLMServerManager` box wired with "creates & manages".
  A `.legend` maps each color.
- **2. acquire_server() Decision Flow** (SVG flowchart): start ellipse → "req_id in LRU
  cache?" diamond → YES "server in active pool?" diamond → return cached, NO →
  invalidate, → "Least-Loaded Selection" → update cache + return → Done.
- **3. Key Mechanisms** (flow-cards): Sticky Session (Prefix Caching), Least-Loaded
  Routing, Replica Count Calculation, Lifecycle (LLMServerManager).
- **4. Request Sequence** (SVG): lifelines for `LLMServerClient`, `GlobalRequestLoadBalancer`
  (Ray actor), `Server Replica`; numbered arrows acquire → return handle → generate →
  TokenOutput → release, with an `inflight[sid]=1` activation annotation.
- **5. Key Code**: annotated `acquire_server()` / `release_server()` snippets.
- **6. Summary**: table of classes (LLMServerManager, GlobalRequestLoadBalancer,
  LLMServerClient, RolloutReplica) with roles and key methods.

## Example (comprehensive): `verl/experimental/fully_async_policy`

For a large multi-component system, scale up to ~9 numbered sections using the
**Optional Advanced Components**. Reference output:
`paper_notes/fully_async_full_architecture_diagram.html`.

- **Title + subtitle** with a module-path `<code>` badge and a top `.legend` of resource pools.
- **1. System Overview** (SVG): dashed container rects for each resource pool (Rollouter
  blue, Trainer green), a purple Driver Ray-actor box, a MessageQueue, and colored arrows
  for data vs parameter-sync flows.
- **2. Bootstrap / Initialization** (SVG): component construction order.
- **3. Streaming Dataflow** (SVG): producer/consumer steady state.
- **4. Internals** (SVG): asyncio coroutine **state machine**.
- **5. Pipeline** (stage-cards): `.stage-card`s with `loc-cpu`/`loc-gpu` badges and a
  `.fields` pseudo-code box for the key method.
- **6. Parameter Synchronization** (SVG): NCCL weight-sync sequence.
- **7. Staleness Control** (stage-cards incl. a `.full-width` card with a `.data-table`).
- **8. Operating Modes** + **9. Key Configuration**: rich `.data-table` matrices.
- **Metrics strip** (`.metric-grid`): headline figures (speedup, units, primitives).
- **Footer**: source path citation.

---

## Quality Checklist

- [ ] Single self-contained dark-themed HTML page (only highlight.js loaded from CDN)
- [ ] GitHub dark palette (`#0d1117` background, `#c9d1d9` text) via the CSS design system
- [ ] At least one hand-authored **SVG component/architecture diagram** showing all classes
- [ ] At least one hand-authored **SVG sequence or decision flowchart** showing runtime flow
- [ ] **No Mermaid and no reveal.js** anywhere
- [ ] Reusable arrow markers defined in `<defs>` and referenced via `marker-end`
- [ ] Consistent color language (blue/green/red/orange/purple) across diagrams + a legend
- [ ] Ray actors clearly marked with a `(Ray Actor)` sub-line
- [ ] `.flow-section` grid of `.flow-card`s explaining key mechanisms
- [ ] Code snippets syntax-highlighted (highlight.js github-dark), ≤ 20 lines each
- [ ] Numbered `<h2>` sections; one topic per section
- [ ] Summary table as the final section
- [ ] All key classes and their public methods are covered
- [ ] Design patterns and trade-offs are explicitly called out (why, not just what)

## Anti-patterns

- ❌ **Using Mermaid or reveal.js** — diagrams must be hand-authored inline SVG on a single page
- ❌ **Light theme** — the page must use the GitHub dark palette
- ❌ **Dumping entire file contents** — show only essential 10–20 line snippets per section
- ❌ **Missing diagrams** — every walkthrough needs at least 2 inline SVG diagrams
- ❌ **Text-wall sections** — if a section has no diagram, card-grid, code, or table, restructure it
- ❌ **Inconsistent colors** — reuse one color language and document it in a legend
- ❌ **Implementation details without context** — always explain *why*, not just *what*
- ❌ **Skipping error handling / edge cases** — note gotchas and failure modes
- ❌ **Generic descriptions** — cite specific class names, method signatures, line numbers
- ❌ **External CSS/JS dependencies beyond highlight.js** — keep the page self-contained

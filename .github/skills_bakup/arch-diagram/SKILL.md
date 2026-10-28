---
name: arch-diagram
description: "Generate a self-contained dark-themed single-page HTML system architecture diagram with hand-authored inline SVG (component/pool diagrams, producer-consumer dataflow, asyncio/state machines, parameter-sync sequences), location pill-badges, stage-cards, rich data-tables, and headline metric cards. Use when: architecture diagram, system diagram, dataflow diagram, full architecture, component diagram, system overview, visualize architecture, resource pools, 架构图, 数据流图, 系统架构. For per-class annotated code walkthroughs, use the code-walkthrough skill instead."
argument-hint: "System / module path or a set of files (e.g. verl/experimental/fully_async_policy)"
---

# Architecture Diagram (Dark Single-Page System Overview)

Produce a **self-contained, dark-themed, single-page HTML document** that explains a
*whole system's* architecture and runtime behavior using **hand-authored inline SVG
diagrams**, **location-tagged cards**, **data-table matrices**, and **headline metric
cards**.

This is a **system-level** view — resource pools, data path vs control path, concurrency
model, parameter synchronization, operating modes, and configuration — drawn directly as
SVG (no Mermaid, no reveal.js). Code snippets are optional and secondary; the focus is the
architecture, not a line-by-line code walk.

Reference exemplar (match this look exactly):
`paper_notes/fully_async_full_architecture_diagram.html`.

## When to Use

- User asks to "draw the architecture", "system diagram", "full architecture diagram",
  "dataflow diagram", "component diagram", or "system overview"
- A multi-component system needs a single shareable visual: resource pools, producers/
  consumers, queues, Ray actors, weight sync, operating modes
- User wants concurrency / state-machine / pipeline visualization at the system level

**Use `code-walkthrough` instead** when the focus is a single module explained
**class-by-class with annotated code snippets**. `arch-diagram` is for the bird's-eye
system architecture; `code-walkthrough` is for the code internals. They share the same
dark hand-authored-SVG visual language.

## Inputs

One of:
1. A **system/package path** (e.g., `verl/experimental/fully_async_policy`)
2. A **set of files** that together implement a system
3. A **system name** with enough context to locate its components

If none provided, ask the user for the target system or files.

---

## Procedure

### Step 1 — Analyze the System

Read the relevant files end-to-end and extract:
1. **Components** — the top-level classes/actors and what each owns.
2. **Resource pools / placement** — which components run on CPU drivers, GPU replicas,
   Ray actors; how resources are partitioned (e.g., isolated rollout vs train pools).
3. **Data path** — how data flows (producer → queue → consumer), the minimal transmission
   unit, batching/balancing.
4. **Control path** — supervision, lifecycle (bootstrap/init order), cancellation.
5. **Concurrency model** — asyncio coroutines, threads, prefetch, state machine states.
6. **Synchronization** — parameter/weight sync primitive (NCCL, etc.), when it triggers.
7. **Backpressure / staleness** — budgets, pause conditions, drift bounds.
8. **Operating modes & configuration** — the knobs and the regimes they produce.
9. **Headline metrics** — speedups, counts, key primitives worth a stat card.

### Step 2 — Plan the Numbered Sections

The page is one scrolling document with numbered `<h2>` sections. Pick the subset that
fits the system (4–9 sections typical):

| # | Section | Primary component |
|---|---------|-------------------|
| — | Title + subtitle (module path badge) + top **legend** | header |
| 1 | System Overview — components & resource pools | SVG component diagram |
| 2 | Bootstrap / Initialization | SVG (construction order) |
| 3 | Streaming Dataflow — producer/consumer | SVG (queue + arrows) |
| 4 | Internals — asyncio / state machine | SVG state diagram |
| 5 | Pipeline (per-step) | stage-cards + `.fields` |
| 6 | Parameter Synchronization | SVG sequence |
| 7 | Staleness / Backpressure | stage-cards + data-table |
| 8 | Operating Modes | `.data-table` matrix |
| 9 | Key Configuration | `.data-table` |
| — | Headline metrics strip | `.metric-grid` |
| — | Footer — source citation | `<footer>` |

Every section must contain a diagram, a card grid, a table, or a metric strip — never a
wall of text.

### Step 3 — Author the SVG Diagrams (by hand)

Draw **at least two** diagrams as inline `<svg>` — never Mermaid.

**Building blocks:**

- **Canvas**: `<svg viewBox="0 0 W H" xmlns="http://www.w3.org/2000/svg">` (CSS scales
  width to 100%). Wide systems use a `viewBox` up to ~1200 wide; `.diagram` allows
  horizontal scroll.
- **Resource-pool containers**: large dashed rects (`stroke-dasharray="6,3"`) behind the
  components of one pool, with a bold pool-title `<text>`.
- **Component boxes**: `<rect rx="8" fill stroke>` + a bold title `<text>` + 1–2 muted
  sub-lines (role, file, key method). Mark Ray actors with a `(Ray actor)` sub-line.
- **Queue / store**: a distinct box (orange family) between producer and consumer.
- **Decision diamonds**: `<polygon points="cx,top rx,cy cx,bot lx,cy">` for branching.
- **State machine**: small rounded-rect state nodes connected by labeled curved arrows
  (self-loops via a `<path>` arc).
- **Sequence diagram**: a header box + vertical dashed lifeline per participant; numbered
  horizontal arrows (solid = call, dashed = return); activation rects on lifelines.
- **Arrow markers**: define once in `<defs>`, reference via `marker-end`:

```html
<defs>
  <marker id="aB" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#58a6ff"/></marker>
  <marker id="aG" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#3fb950"/></marker>
  <marker id="aR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#f85149"/></marker>
  <marker id="aO" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#d29922"/></marker>
  <marker id="aP" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#bc8cff"/></marker>
  <marker id="aPk" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill="#ff7eb6"/></marker>
</defs>
```

- **Connectors**: `<line>` or curved `<path d="M x1 y1 Q cx cy x2 y2">`; solid for primary
  flow, `stroke-dasharray="4,2"` for returns/control/param-sync.
- **Edge labels**: small `<text>` near each arrow's midpoint.

**Color language** (use consistently across diagrams + the top legend):

| Color | Stroke / box | Use for |
|-------|--------------|---------|
| Blue | `#388bfd` / `#0a1628`,`#0d419d` | clients, rollout/producer pool, entry points |
| Green | `#238636` / `#0d2818` | core compute, trainer pool, success/public API |
| Red | `#da3633` / `#3d1a1a` | external/remote servers, replicas, error path |
| Orange | `#9e6a03` / `#2a1f00` | queues, branch selection, trade-offs |
| Purple | `#8957e5` / `#1a0d2e` | Ray actors, driver, lifecycle/control |
| Pink | `#bf4080` / `#3d1a2e` | parameter / weight sync (NCCL) |

### Step 4 — Write the HTML Document

Save a single self-contained `.html` file to `paper_notes/`:

```
paper_notes/<system-name>_architecture.html
```

e.g., `fully_async_full_architecture_diagram.html`.

---

## CSS Design System (GitHub dark)

Self-contained: all CSS + SVG inline. Code blocks (optional) may load highlight.js
`github-dark` from CDN; otherwise no external dependencies.

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 40px; }
h1 { text-align: center; margin-bottom: 8px; color: #58a6ff; font-size: 1.8em; }
h2 { color: #79c0ff; margin: 38px 0 15px; font-size: 1.25em;
     border-bottom: 1px solid #21262d; padding-bottom: 8px; }
.subtitle { text-align: center; color: #8b949e; margin-bottom: 30px; }
.subtitle code { background: #1f2937; padding: 1px 6px; border-radius: 4px; color: #f0883e; font-size: 0.85em; }
.container { max-width: 1300px; margin: 0 auto; }
.diagram { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
           padding: 30px; margin: 20px 0; overflow-x: auto; }
svg { width: 100%; height: auto; display: block; }

/* legend */
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 14px 0 4px; font-size: 0.8em; color: #8b949e; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

/* stage cards */
.stage-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
.stage-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; position: relative; }
.stage-card h3 { margin-bottom: 8px; font-size: 0.95em; }
.stage-card ul { padding-left: 16px; line-height: 1.7; font-size: 0.82em; }
.stage-card code { background: #1f2937; padding: 1px 5px; border-radius: 3px; color: #f0883e; font-size: 0.85em; }
.stage-card .fields { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px;
       margin-top: 8px; font-family: monospace; font-size: 0.75em; line-height: 1.5; color: #7ee787; white-space: pre; }
.full-width { grid-column: 1 / -1; }

/* location / role pill-badges */
.loc { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.7em; margin-left: 8px; vertical-align: middle; }
.loc-cpu  { background: #1a2e4a; color: #79c0ff; border: 1px solid #388bfd; }
.loc-gpu  { background: #2a1f00; color: #e3b341; border: 1px solid #9e6a03; }
.loc-ray  { background: #1a0d2e; color: #bc8cff; border: 1px solid #8957e5; }
.loc-sync { background: #3d1a2e; color: #ff7eb6; border: 1px solid #bf4080; }

/* rich data table */
table.data-table { width: 100%; border-collapse: collapse; font-size: 0.8em; margin: 15px 0; }
table.data-table th, table.data-table td { padding: 7px 10px; border: 1px solid #30363d; vertical-align: top; }
table.data-table th { background: #1a2332; color: #79c0ff; text-align: left; }
table.data-table td { background: #0d1117; }
table.data-table tr:hover td { background: #11161d; }
table.data-table code { background: #1f2937; padding: 1px 4px; border-radius: 3px; color: #f0883e; }

/* metric / stat cards */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 14px; margin: 18px 0; }
.metric { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 14px 16px; }
.metric .v { font-size: 1.5em; font-weight: 700; color: #58a6ff; }
.metric .l { font-size: 0.75em; color: #8b949e; margin-top: 2px; }

footer { text-align: center; color: #586069; font-size: 0.78em; margin-top: 40px; }
footer code { color: #8b949e; }
@media (max-width: 900px) { .stage-grid { grid-template-columns: 1fr; } }
```

---

## Section Catalog (markup patterns)

**Title + subtitle + top legend** — orient the reader before any diagram:

```html
<h1>Fully Async Policy — Full Architecture</h1>
<p class="subtitle">Decoupled Rollouter + Trainer · streaming MessageQueue · NCCL weight sync<br>
<code>verl/experimental/fully_async_policy</code></p>
<div class="legend">
  <span><span class="sw" style="background:#0a1628;border:1px solid #388bfd"></span>Rollouter pool</span>
  <span><span class="sw" style="background:#0d2818;border:1px solid #238636"></span>Trainer pool</span>
  <span><span class="sw" style="background:#2a1f00;border:1px solid #9e6a03"></span>MessageQueue</span>
  <span><span class="sw" style="background:#1a0d2e;border:1px solid #8957e5"></span>Ray actor / RPC</span>
  <span><span class="sw" style="background:#3d1a2e;border:1px solid #bf4080"></span>Parameter sync (NCCL)</span>
</div>
```

**Stage-card with a `.fields` pseudo-code box and a location badge:**

```html
<div class="stage-grid">
  <div class="stage-card">
    <h3 style="color:#ff7eb6">Decoupled-PPO / Rollout-IS <span class="loc loc-cpu">bypass_mode=False</span></h3>
    <div class="fields">_compute_old_log_prob(batch):
  if local_trigger_step == 1:
    save_model_to_cpu(1)          # snapshot v1
    old_log_prob = engine(batch)</div>
  </div>
</div>
```

**Operating-modes / config matrix** (rich table, optionally `.full-width`):

```html
<table class="data-table">
  <tr><th>#</th><th>Mode</th><th>Params</th><th>Behavior</th></tr>
  <tr><td>a</td><td><b>On-policy pipeline</b></td><td><code>sync=1</code><br><code>staleness=0</code></td>
      <td>Produce required samples, train once, then sync.</td></tr>
</table>
```

**Headline metrics strip:**

```html
<div class="metric-grid">
  <div class="metric"><div class="v">2.35–2.67×</div><div class="l">speedup · 128 GPUs</div></div>
  <div class="metric"><div class="v">NCCL</div><div class="l">weight sync primitive</div></div>
</div>
```

**Footer (always cite the source path):**

```html
<footer>Source: <code>verl/experimental/fully_async_policy</code></footer>
```

---

## Example: `verl/experimental/fully_async_policy`

Reference output: `paper_notes/fully_async_full_architecture_diagram.html`.

- **Title + subtitle** with the module-path badge and a top `.legend` of resource pools.
- **1. System Overview** (SVG): a purple Driver Ray-actor box supervising two dashed pool
  containers — Rollouter (blue, with `FullyAsyncRollouter` + `AgentLoopManager`) and
  Trainer (green) — connected through an orange `MessageQueue`, with pink NCCL param-sync
  arrows.
- **2. Bootstrap** (SVG): `_initialize_components()` construction order.
- **3. Streaming Dataflow** (SVG): producer → MessageQueue (1 sample unit) → consumer.
- **4. Rollouter Internals** (SVG): asyncio coroutine state machine (feed / processor /
  monitor + staleness control).
- **5. Trainer `fit_step()`** (stage-cards): `loc-cpu`/`loc-gpu` badges, a `.fields`
  pseudo-code box for `_compute_old_log_prob`, async-prefetch invariant card.
- **6. Parameter Synchronization** (SVG sequence): `CheckpointEngineManager.update_weights()`.
- **7. Staleness / Backpressure** (stage-cards incl. a `.full-width` card with a data-table).
- **8. Operating Modes** + **9. Key Configuration**: `.data-table` matrices.
- **Metrics strip** (`.metric-grid`) + **Footer** citing the source path.

---

## Quality Checklist

- [ ] Single self-contained dark-themed HTML page (GitHub dark palette)
- [ ] Title + subtitle with a module-path `<code>` badge
- [ ] A top **legend** of colors/resource pools, reused consistently in every diagram
- [ ] At least one hand-authored **SVG component / resource-pool diagram**
- [ ] At least one hand-authored **SVG dataflow, sequence, or state-machine diagram**
- [ ] **No Mermaid and no reveal.js** anywhere
- [ ] Reusable arrow markers in `<defs>`; consistent color language across diagrams
- [ ] Ray actors marked with a `(Ray actor)` sub-line
- [ ] Location pill-badges (`loc-cpu/-gpu/-ray/-sync`) where placement matters
- [ ] `.data-table` matrices for operating modes / configuration
- [ ] `.metric-grid` headline metrics where meaningful
- [ ] Numbered `<h2>` sections; one topic per section; no text-wall sections
- [ ] Footer citing the source system path
- [ ] Concurrency / sync / backpressure behavior is explicitly explained (why, not just what)

## Anti-patterns

- ❌ **Using Mermaid or reveal.js** — diagrams must be hand-authored inline SVG on one page
- ❌ **Light theme** — must use the GitHub dark palette
- ❌ **Missing the top legend** — readers can't decode colors without it
- ❌ **Class-by-class code dump** — that's `code-walkthrough`; keep this system-level
- ❌ **Text-wall sections** — every section needs a diagram, card grid, table, or metric strip
- ❌ **Inconsistent colors** — one color language, documented in the legend
- ❌ **Ignoring concurrency/sync** — the whole point is the runtime/dataflow/sync behavior
- ❌ **Generic descriptions** — cite specific component names, methods, config keys
- ❌ **External dependencies beyond optional highlight.js** — keep the page self-contained

## Relationship to `code-walkthrough`

Both skills emit the same dark, self-contained, hand-authored-SVG HTML. Choose by focus:

| Focus | Skill |
|-------|-------|
| Bird's-eye **system architecture & dataflow** (pools, queues, sync, modes, metrics) | **arch-diagram** |
| **Per-class code internals** with annotated ≤20-line snippets | **code-walkthrough** |

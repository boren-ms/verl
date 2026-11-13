---
name: code-walkthrough
description: "Generate a self-contained reveal.js HTML slide deck that explains a code module's workflow with Mermaid diagrams, annotated code snippets, and class/sequence diagrams. Use when: explain code, code walkthrough, code slides, code diagram, explain module, code architecture, code flow, visualize code, 代码讲解, 代码流程图, module walkthrough, class diagram for code."
argument-hint: "File path or module path (e.g. verl/workers/rollout/llm_server.py)"
---

# Code Walkthrough (reveal.js Slide Deck with Diagrams)

Produce a **self-contained reveal.js HTML slide deck** that explains a code module's architecture, class relationships, and runtime workflow using **Mermaid diagrams**, annotated code snippets, and concise explanatory bullets.

## When to Use

- User points to a source file or module and asks "explain this code" / "walkthrough" / "code slides"
- User wants a visual architecture overview of a module with class/sequence/flowchart diagrams
- User says "generate code slides" / "code diagram" / "visualize the workflow"
- Producing a shareable single-file HTML presentation of a codebase module

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

### Step 2 — Plan the Slide Deck

Map the code to a narrative arc:

| Slide | Purpose |
|-------|---------|
| Title & Overview | Module name, one-line purpose, file location |
| Architecture Diagram | Mermaid class diagram showing all classes and relationships |
| Class-by-Class Walkthrough | One slide per major class: purpose, key methods, attributes |
| Initialization Flow | Sequence diagram showing how the system boots up |
| Request Lifecycle | Sequence/flowchart showing a typical request from start to finish |
| Key Code Snippets | Annotated code for the most important methods |
| Design Decisions | Patterns used, trade-offs, extension points |
| Summary | Quick-reference table of classes + responsibilities |

Adjust the number of slides to the module's complexity (6–15 slides typical).

### Step 3 — Generate Mermaid Diagrams

Create at least **two** diagrams:

1. **Class/Component Diagram** — shows classes, their key methods, and relationships (composition, inheritance, uses).
2. **Sequence or Flowchart Diagram** — shows the runtime flow (initialization, request handling, or data pipeline).

Additional diagrams as needed:
- State diagram for lifecycle management
- Flowchart for branching logic (e.g., hybrid vs standalone init)

**Mermaid rendering**: Use the Mermaid CDN in the HTML so diagrams render client-side:
```html
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad: false, theme: 'base', themeVariables: {fontSize: '14px'}});</script>
```

Re-render Mermaid on each slide change:
```javascript
Reveal.on('slidechanged', async () => {
  await mermaid.run({querySelector: '.mermaid'});
});
Reveal.on('ready', async () => {
  await mermaid.run({querySelector: '.mermaid'});
});
```

### Step 4 — Write the HTML Slide Deck

Save a single self-contained `.html` file to `paper_notes/` (reuse the project's existing notes directory):

```
paper_notes/<module-name>_code_walkthrough.html
```

**Naming**: derive from the module — e.g., `llm_server_code_walkthrough.html`.

---

## Slide Structure

### Slide 1 — Title & Module Overview

- Module name as title, file path as subtitle
- Mermaid **component overview** (simplified — just boxes and arrows showing major pieces)
- 3–4 compact bullets: what this module does, why it exists, key design goals
- Badge-style links to the source file (relative path)

### Slide 2 — Architecture (Class Diagram)

- Full Mermaid **class diagram** showing:
  - All classes in the module
  - Key methods (public API only — skip internal helpers)
  - Key attributes
  - Relationships: inheritance (`--|>`), composition (`*--`), dependency (`..>`)
- 2–3 bullets summarizing the class hierarchy

### Slides 3–N — Class Walkthrough (one per major class)

Each slide covers one class:
- **Class name** as heading with a one-line description
- **Key attributes** in a small table or bullet list
- **Key methods** with signature + one-line purpose
- **Annotated code snippet** of the most important method (10–20 lines max)
  - Use syntax-highlighted `<pre><code class="language-python">` blocks
  - Add `<!-- ← comment -->` annotations pointing to key lines
- If the class is a Ray actor, note that prominently

### Sequence/Flow Slide — Initialization

- Mermaid **sequence diagram** showing the boot-up order:
  - Which class creates which
  - Async initialization steps
  - Config flow
- Compact bullets explaining each phase

### Sequence/Flow Slide — Request Lifecycle

- Mermaid **sequence diagram** or **flowchart** showing a single request:
  - Entry point → load balancer → server selection → execution → response
  - Error handling / finally blocks
- Compact bullets explaining the happy path and edge cases

### Design Decisions Slide

- 3–5 bullets on key design choices:
  - Why this pattern was chosen (e.g., "sticky sessions for prefix caching")
  - Trade-offs acknowledged
  - Extension points for subclasses

### Summary Slide

- Table with columns: Class | Role | Key Methods
- One row per class, terse descriptions

---

## Design Requirements — reveal.js Slide Deck

Uses **reveal.js 5.2.1** (CDN-loaded) with the **white theme**, Mermaid diagrams, and syntax-highlighted code.

### reveal.js Setup

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.2.1/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.2.1/dist/theme/white.css">
<!-- Highlight.js for code syntax highlighting -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11/styles/github.min.css">
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11/languages/python.min.js"></script>
<!-- Mermaid for diagrams -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
```

```javascript
Reveal.initialize({
  hash: true, slideNumber: 'c/t',
  width: 1280, height: 720, margin: 0.08,
  transition: 'slide', center: false, plugins: []
});
// Re-render Mermaid on slide change
Reveal.on('slidechanged', async () => {
  await mermaid.run({querySelector: '.mermaid'});
});
Reveal.on('ready', async () => {
  await mermaid.run({querySelector: '.mermaid'});
  hljs.highlightAll();
});
mermaid.initialize({startOnLoad: false, theme: 'base',
  themeVariables: {fontSize: '14px', primaryColor: '#dbeafe', lineColor: '#1a6fb5'}});
```

### CSS Design System

```css
/* Base */
.reveal { font-size: 24px; }
.reveal .slides section { overflow: hidden; padding: 20px 30px; }

/* Headings */
.reveal h2 { font-size: 1.15em; border-bottom: 2px solid #1a6fb5; margin-bottom: 18px; }
.reveal h3 { font-size: 0.9em; color: #475569; margin-bottom: 10px; }

/* Content — unified 0.68em */
.reveal ul, .reveal ol { font-size: 0.68em; line-height: 1.45; margin: 10px 0; }
.reveal p { font-size: 0.68em; margin-bottom: 8px; }

/* Code blocks */
.reveal pre { font-size: 0.58em; margin: 10px 0; border-radius: 6px; }
.reveal code { font-family: 'JetBrains Mono', 'Fira Code', monospace; }
.code-annotation { font-size: 0.55em; color: #6b7280; font-style: italic; margin-left: 8px; }

/* Mermaid diagrams — constrained height, centered */
.mermaid { max-height: 420px; margin: 10px auto; }
.mermaid svg { max-height: 420px; }

/* Tables */
.reveal table { font-size: 0.68em; width: 100%; margin: 10px auto 14px; }
.reveal table th { background: #dbeafe; }

/* Color highlights */
.hl-blue { color: #1a6fb5; font-weight: 700; }    /* classes, types */
.hl-green { color: #16a34a; font-weight: 700; }   /* key methods, success paths */
.hl-orange { color: #d97706; font-weight: 700; }  /* patterns, design decisions */
.hl-red { color: #dc2626; font-weight: 700; }     /* warnings, error paths, gotchas */
.hl-purple { color: #7c3aed; font-weight: 700; }  /* Ray/distributed concepts */

/* Title slide */
.title-slide { background: #f8fafc; }
.title-slide h1 { color: #0f172a; border: none; text-align: center; }
.title-slide .meta { color: #475569; text-align: center; font-size: 0.68em; }

/* File path badge */
.file-badge { font-size: 0.55em; background: #dbeafe; color: #1e40af;
              padding: 3px 10px; border-radius: 12px; font-family: monospace; }

/* Ray actor indicator */
.ray-badge { font-size: 0.52em; background: #fce7f3; color: #9d174d;
             padding: 2px 8px; border-radius: 10px; font-weight: 700; }
```

### Color Highlighting

Use on **every slide** to draw attention to key concepts:

| Class | Color | Use for |
|-------|-------|---------|
| `.hl-blue` | Blue (#1a6fb5) | Class names, type annotations |
| `.hl-green` | Green (#16a34a) | Key methods, success paths, public API |
| `.hl-orange` | Orange (#d97706) | Design patterns, config params, trade-offs |
| `.hl-red` | Red (#dc2626) | Error handling, gotchas, warnings |
| `.hl-purple` | Purple (#7c3aed) | Ray actors, distributed concepts, async |

### Key Layout Rules

1. **Vertical stacking only** — no two-column layouts. Content flows top-to-bottom: diagram → bullets, or code → bullets.
2. **One topic per slide** — one class, one flow, or one decision per slide.
3. **Code snippets ≤ 20 lines** — show only the essential logic; trim boilerplate, imports, docstrings.
4. **Mermaid diagrams constrained** — `max-height: 420px` to prevent overflow.
5. **Horizontal navigation** — slides are siblings, not nested.
6. **Section titles with emoji** prefix: 🏗️ 📐 🔄 ⚡ 🎯 🔧 📊

---

## Example: `verl/workers/rollout/llm_server.py`

For this module, the skill would produce slides like:

**Slide 1 — Title**: "LLM Server Manager & Client" — manages LLM server replicas for RL rollout generation. Three main classes orchestrate server lifecycle, load balancing, and request routing.

**Slide 2 — Architecture**: Mermaid class diagram showing `LLMServerManager` creates `RolloutReplica[]` and `GlobalRequestLoadBalancer`, then `LLMServerClient` talks to the load balancer which routes to replicas.

**Slide 3 — GlobalRequestLoadBalancer** (Ray actor): Sticky-session + least-loaded routing. Key methods: `acquire_server()`, `release_server()`, `add_servers()`, `remove_servers()`.

**Slide 4 — LLMServerClient**: Proxy client used by `AgentLoopWorker`. Acquires server via load balancer, calls `server.generate.remote()`, releases on completion.

**Slide 5 — LLMServerManager**: Lifecycle manager. `create()` → `_initialize_llm_servers()` → `_init_global_load_balancer()`. Handles hybrid vs standalone init, disaggregated prefill/decode.

**Slide 6 — Init Flow**: Sequence diagram: `create()` → compute `num_replicas` → instantiate `RolloutReplica` objects → `init_hybrid()` or `init_standalone()` → collect handles → create load balancer.

**Slide 7 — Request Flow**: Sequence diagram: `generate()` → `acquire_server(request_id)` → sticky check → least-loaded fallback → `server.generate.remote()` → `release_server()`.

**Slide 8 — Design Decisions**: Sticky sessions for prefix caching, fire-and-forget release, atomic acquire returning both ID and handle, LRU eviction for session cache.

**Slide 9 — Summary Table**: Quick reference of all classes + roles.

---

## Quality Checklist

- [ ] reveal.js slide deck renders as self-contained HTML (no local dependencies)
- [ ] At least one Mermaid **class/component diagram** showing all classes
- [ ] At least one Mermaid **sequence or flowchart** showing runtime flow
- [ ] Mermaid diagrams render on slide navigation (re-run on `slidechanged`)
- [ ] Code snippets are syntax-highlighted with highlight.js
- [ ] Code snippets ≤ 20 lines, showing only essential logic
- [ ] One topic per slide — no combined concerns
- [ ] Color highlights on every slide (blue for classes, green for methods, etc.)
- [ ] Ray actors/remote calls clearly marked with `.ray-badge`
- [ ] Vertical layout only — no two-column flexbox
- [ ] File path badge on title slide linking to source
- [ ] Summary table as final slide
- [ ] All key classes and their public methods are covered
- [ ] Design patterns and trade-offs are explicitly called out

## Anti-patterns

- ❌ **Dumping entire file contents** — show only essential 10–20 line snippets per slide
- ❌ **Missing diagrams** — every code walkthrough must have at least 2 Mermaid diagrams
- ❌ **Text-wall slides** — if a slide has no diagram, code, or table, it needs restructuring
- ❌ **Implementation details without context** — always explain *why*, not just *what*
- ❌ **Skipping error handling / edge cases** — note gotchas and failure modes
- ❌ **Generic descriptions** — cite specific class names, method signatures, line numbers
- ❌ **Unrendered Mermaid** — must include re-render on `slidechanged` event
- ❌ **Two-column layouts** — stack vertically
- ❌ Outputting Markdown — must be well-designed HTML

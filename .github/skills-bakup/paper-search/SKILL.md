---
name: paper-search
description: "Search academic papers for an input topic, research area, author, lab, company, university, or organization. Use when: find papers, literature search, survey recent work, arXiv search, Semantic Scholar search, Google Scholar-style search, papers from OpenAI/Meta/Google/Microsoft/Stanford, papers by organization, related work discovery, build a reading list, compare papers on a topic."
argument-hint: "Topic, research question, author, lab, company, university, or organization name"
---

# Paper Search

Find and rank academic papers for a user-provided topic, research question, author, lab, company, university, or other organization. Produce a concise reading list with source links, relevance rationale, and next-step recommendations.

## When to Use

- User asks to search papers for a topic, method, benchmark, dataset, model family, or research question
- User asks for papers from or by an organization, company, lab, university, team, or author group
- User wants recent work, seminal work, related work, a literature survey seed list, or papers to read next
- User wants to compare paper directions across organizations or identify who is working on a topic

## Inputs

The user should provide at least one of:

1. **Topic or research question**: e.g. `speech recognition RLHF`, `long-context ASR`, `agentic search evaluation`
2. **Organization or lab**: e.g. `Microsoft Research`, `OpenAI`, `Meta FAIR`, `Stanford NLP`, `Google DeepMind`
3. **Author or team**: e.g. `papers by <author> on ASR`, `work from the Qwen audio team`
4. **Constraints**: optional date range, venue tier, minimum citation count, paper count, language, must-include or must-exclude keywords

If the input is ambiguous, ask one concise clarification: whether the user wants a **topic search**, **organization search**, or **combined topic + organization search**.

## Output

Default output is a Markdown report in the chat. If the search is large, also save a structured report under:

```text
paper_notes/search_<slug>.md
```

The report should include:

- Search scope and normalized query terms
- Ranked table of papers with title, year, authors, venue/source, organization signal, links, and relevance reason
- Short grouping by theme or chronology
- Top 3-5 papers to read first and why
- Gaps, caveats, and suggested follow-up searches

## Source Strategy

Use multiple sources when available; no single index is complete.

### Primary Sources

- **Semantic Scholar**: broad paper metadata, citation counts, abstracts, related papers, author pages
- **arXiv**: newest preprints, exact arXiv links, author-submitted versions
- **OpenReview**: ICLR/NeurIPS workshop submissions and reviews where relevant
- **ACL Anthology / IEEE / ACM / PubMed / DBLP**: domain-specific sources when the topic calls for them
- **Organization publication pages**: lab, company, or university publication lists for affiliation-grounded searches

### Search Query Patterns

For topic searches, combine:

```text
"<core topic>" paper
"<core topic>" arXiv
"<core topic>" site:semanticscholar.org
"<core topic>" "benchmark" OR "dataset" OR "survey"
```

For organization searches, combine:

```text
"<organization>" "<topic>" paper
"<organization>" "<topic>" arXiv
"<organization>" publications "<topic>"
"<organization domain>" "<topic>" "paper"
```

For author searches, combine:

```text
"<author name>" "<topic>"
"<author name>" site:semanticscholar.org
"<author name>" site:arxiv.org
```

## Procedure

### Step 1 - Normalize the Request

Extract:

- Core topic terms and synonyms
- Organization, lab, university, or company aliases
- Author names, if any
- Time range, if stated; otherwise default to recent 5 years plus seminal older works
- Desired result count; otherwise default to 10-15 papers

For organizations, include common aliases and parent/sub-lab names. Examples: `Google DeepMind` may also appear as `DeepMind`, `Google Research`, or `Google`; `Meta FAIR` may appear as `FAIR`, `Meta AI`, or `Facebook AI Research`.

### Step 2 - Plan the Search Branch

Choose one branch:

1. **Topic branch**: prioritize topical relevance, then citations/recency
2. **Organization branch**: prioritize reliable affiliation evidence, then relevance
3. **Combined branch**: require both topical relevance and organization signal, unless the user asks for a broad landscape

If the topic is very new, lower the citation threshold and rely more on recency, venue, author track record, and abstract fit.

### Step 3 - Search and Collect Candidates

Collect 20-50 candidate papers before ranking when possible.

For each candidate, capture:

- Title
- Year
- Authors
- Venue or source
- Abstract or summary snippet
- Links: arXiv, DOI, project page, PDF, code if present
- Citation count when available
- Organization evidence: author affiliation, publication page, institutional domain, lab page, or explicit paper metadata

Deduplicate aggressively by normalized title. Merge arXiv, conference, and project links into one record.

### Step 4 - Score and Rank

Rank using these criteria:

- **Relevance**: direct match to the user's topic or research question
- **Authority**: venue, citations, author history, known benchmark impact
- **Recency**: newer papers matter more for fast-moving areas; older highly cited papers may be marked as seminal
- **Organization confidence**: high only when affiliation or official publication listing is verified
- **Usefulness**: whether the paper provides methods, experiments, datasets, code, or a survey useful for the user's likely next step

Use labels rather than false precision:

- `Must read`: central paper for the query
- `Strong fit`: relevant and useful, but narrower or less foundational
- `Adjacent`: useful context, baseline, or related direction
- `Organization match uncertain`: organization connection is plausible but not verified

### Step 5 - Read Enough to Summarize Correctly

For the top papers, read abstracts and, when needed, introductions, figures, tables, or conclusions. Do not infer contributions from the title alone.

Summaries should answer:

- What problem does the paper address?
- What is the main idea or contribution?
- Why is it relevant to the user's query?
- What should the user read it for?

### Step 6 - Produce the Report

Use this structure by default:

```markdown
# Paper Search: <query>

## Search Scope
<topic terms, organization aliases, date range, sources checked>

## Best Starting Points
| Priority | Paper | Why read it first |
|---|---|---|

## Ranked Papers
| Rank | Paper | Year | Source | Org Signal | Why It Matters |
|---|---:|---|---|---|---|

## Themes
<3-5 grouped observations>

## Caveats and Follow-ups
<missing sources, uncertain affiliations, suggested next searches>
```

When there are many good papers, group by theme first and rank inside each theme.

## Quality Checks

- [ ] Query terms and organization aliases are explicit
- [ ] At least two independent sources were checked when feasible
- [ ] Paper links are included and deduplicated
- [ ] Organization matches include evidence or are labeled uncertain
- [ ] Top recommendations are based on abstract/content inspection, not title-only matches
- [ ] Recent papers and seminal papers are both considered when relevant
- [ ] Caveats mention index gaps, paywalls, affiliation ambiguity, or weak evidence

## Anti-patterns

- Do not present search results as exhaustive unless the search really was systematic across named indexes
- Do not claim a paper is from an organization based only on a coauthor's current employer; verify affiliation for that paper when possible
- Do not rank solely by citation count; new papers and niche benchmark papers may be more relevant
- Do not include unrelated papers just because they share a buzzword with the query
- Do not fabricate venue, citation, affiliation, PDF, or code links
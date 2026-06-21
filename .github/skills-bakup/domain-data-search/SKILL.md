---
name: domain-data-search
description: "Search HuggingFace for open-source text datasets matching target domains. Use when: finding domain-specific training data, analyzing dataset coverage for entity recognition, building domain text corpora, downloading HF dataset samples, generating HTML analysis reports."
argument-hint: "Path to a YAML/JSON config listing target domains, or a comma-separated list of domain names"
---

# Domain Data Search

Search HuggingFace for open-source text datasets that match a set of target domains, download samples, classify them by domain, and produce a statistical analysis with an HTML report.

## When to Use

- You have a list of target domains (e.g. from an eval config YAML) and need to find matching open-source text data
- You want to assess how well publicly available HF datasets cover specific industry/topic domains
- You need domain-specific text corpora for entity recognition, ASR, or NLP fine-tuning
- You want a visual HTML report of domain coverage, vocabulary, text-length distributions, and co-occurrence

## Inputs

The user provides **one** of:
1. A path to a YAML or JSON config file that contains domain names (the skill will extract them)
2. A comma-separated list of domain names (e.g. `Gaming, Insurance, Banking, Energy`)

If not provided, ask the user.

## Procedure

### Step 1 — Extract Target Domains

Parse the input to get a list of domain names. For YAML configs like eval audio sets, look for `test_name` fields and extract the domain component (the last path segment, e.g. `en-US-entity-v3/Gaming` → `Gaming`).

### Step 2 — Define Domain Keywords

For each domain, define a set of 15-25 keyword phrases that characterize the domain. See [keyword reference](./references/domain-keywords.md) for the baseline keyword lists. Adapt or extend keywords based on the specific domains requested.

### Step 3 — Search HuggingFace

Use the HuggingFace Datasets API to find candidate datasets:

```
GET https://huggingface.co/api/datasets?search=<terms>&sort=downloads&direction=-1&limit=10
```

**Search strategy** (run these in parallel):
- General multi-domain corpora: `SlimPajama`, `fineweb`, `pile`, `c4`
- Domain-specific: search for each domain's top 2-3 keywords
- Known high-quality datasets per category — see [dataset catalog](./references/dataset-catalog.md)

Select 2-5 datasets that together maximize domain coverage.

### Step 4 — Download & Classify

Run [search_and_analyze.py](./scripts/search_and_analyze.py) with the target domains and selected datasets:

```bash
python3 .github/skills/domain-data-search/scripts/search_and_analyze.py \
  --domains "Gaming,Insurance,Banking,..." \
  --datasets "DKYoon/SlimPajama-6B:train:text:50000" \
             "timaeus/pile-pubmed_abstracts:train:text:20000" \
  --output-dir <output_directory>
```

The script will:
1. Stream samples from each HF dataset
2. Classify each sample into domains via keyword matching (≥2 keyword hits required)
3. Compute per-domain statistics (sample count, word counts, char lengths, percentiles)
4. Build a domain co-occurrence matrix
5. Generate an HTML report and save JSON/CSV data

If the script is not available or needs modification, follow the logic in the script to implement the analysis inline.

### Step 5 — Generate HTML Report

The script produces three output files in `--output-dir`:
- `domain_analysis_report.html` — Interactive dashboard with charts, heatmaps, box plots
- `domain_analysis_results.json` — Raw per-domain statistics and sample texts
- `domain_analysis_summary.csv` — Tabular summary for spreadsheet use

### Step 6 — Recommend Datasets

Based on the coverage analysis, recommend additional HF datasets for under-represented domains (< 500 samples). Consult [dataset-catalog.md](./references/dataset-catalog.md) for known good datasets per domain.

## Output

- HTML report with: summary stats, domain coverage bar chart, text-length box plots, co-occurrence heatmap, per-domain detail cards with recommendations
- JSON file with full statistics
- CSV summary table
- Console summary of coverage and recommendations

## Quality Checks

- [ ] All target domains have at least some samples (even if few)
- [ ] HTML report renders correctly (valid structure, no broken tags)
- [ ] Under-represented domains have concrete dataset recommendations
- [ ] Co-occurrence matrix is included to show domain overlap

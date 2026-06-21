# HuggingFace Dataset Catalog

Curated list of open-source HF datasets organized by domain coverage.
Datasets are sorted by relevance; ★ marks best-fit datasets.

## General Multi-Domain (covers many domains via filtering)

| Dataset | Size | Notes |
|---------|------|-------|
| ★ DKYoon/SlimPajama-6B | 6B tokens | 7 sources: Wikipedia, C4, CC, ArXiv, GitHub, StackExchange, Book. Best general-purpose starting point. |
| HuggingFaceFW/fineweb-edu | 1.3T tokens | Educational web content. Excellent for K12/Education and ScienceTech. |
| HuggingFaceFW/fineweb-edu-score-2 | 5.4T tokens | Broader version with lower quality threshold. |
| monology/pile-uncopyrighted | ~200M rows | The Pile without copyrighted content. Multiple sub-domains. |

## Finance & Banking

| Dataset | Size | Notes |
|---------|------|-------|
| ★ FinanceMTEB/financial_phrasebank | ~5k sentences | Financial news sentiment. Covers Banking, CapitalMarket. |
| pile-of-law/pile-of-law | 256GB | Legal/regulatory text. Insurance, Banking compliance. |
| SEC EDGAR filings | varies | Via sec-api or manual download. CapitalMarket focus. |

## Health & Medical

| Dataset | Size | Notes |
|---------|------|-------|
| ★ timaeus/pile-pubmed_abstracts | 100k+ abstracts | Biomedical research. LifeHealth, DoctorPatient, PatientHistory. |
| casinca/PUBMED_title_abstracts_2019_baseline | 14M+ | Full PubMed baseline. Large-scale medical text. |
| MIMIC-III (PhysioNet) | varies | Clinical notes. Requires credentialed access. |

## Education

| Dataset | Size | Notes |
|---------|------|-------|
| ★ HuggingFaceFW/fineweb-edu | 1.3T tokens | Best fit for K12HigherEdu. |
| allenai/peS2o | 40M papers | Academic paper abstracts. |

## Science & Technology

| Dataset | Size | Notes |
|---------|------|-------|
| ★ allenai/s2orc | 81M papers | Semantic Scholar. Requires data agreement. |
| ArXiv subset from The Pile | varies | Via timaeus/ or monology/ subsets. |
| bigcode/starcoderdata | 783GB | Code and tech content. |

## Media & News

| Dataset | Size | Notes |
|---------|------|-------|
| ★ cc_news | large | CommonCrawl news articles. |
| multi_news | 56k articles | Multi-document news summarization. |
| cnn_dailymail | 300k articles | CNN and DailyMail news. |

## Energy & Sustainability

| Dataset | Size | Notes |
|---------|------|-------|
| Filter from SlimPajama/FineWeb | varies | Keyword-filter for energy/sustainability terms. |
| Wikipedia energy category dumps | varies | Targeted extraction. |

## Adding Datasets

When discovering new datasets via HF API search:
1. Prefer datasets with `format:parquet` tag (easier streaming)
2. Prefer `language:en` for English domain text
3. Check `downloads` count as quality signal
4. Verify the dataset loads with `load_dataset(name, split=split, streaming=True)`
5. Note the text field name (commonly `text`, `sentence`, `content`)

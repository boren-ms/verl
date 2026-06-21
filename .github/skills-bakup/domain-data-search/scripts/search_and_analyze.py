#!/usr/bin/env python3
"""
Domain Data Search & Analysis Script

Searches HuggingFace for open-source text datasets matching target domains,
downloads samples, classifies them, computes statistics, and generates an
HTML report.

Usage:
    python3 search_and_analyze.py \
        --domains "Gaming,Insurance,Banking,Energy,..." \
        --datasets "DKYoon/SlimPajama-6B:train:text:50000" \
                   "timaeus/pile-pubmed_abstracts:train:text:20000" \
        --output-dir ./analysis_output

    # Or from a YAML config:
    python3 search_and_analyze.py \
        --config path/to/entity_raw.yaml \
        --output-dir ./analysis_output

Each --datasets entry is "name:split:text_field:sample_size".
"""

import argparse
import html as html_mod
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

# ─── Default domain keyword definitions ───────────────────────────────────────

DEFAULT_DOMAIN_KEYWORDS = {
    "Gaming": [
        "game", "gaming", "gamer", "video game", "esports", "playstation", "xbox",
        "nintendo", "steam", "twitch", "fps", "rpg", "mmorpg", "multiplayer",
        "console", "gameplay", "fortnite", "minecraft", "league of legends",
        "overwatch", "valorant", "call of duty", "battlefield", "apex legends",
        "level up", "quest", "boss fight", "dungeon", "loot", "spawn",
    ],
    "Insurance": [
        "insurance", "policyholder", "premium", "deductible", "underwriting",
        "claim", "coverage", "actuary", "actuarial", "liability", "indemnity",
        "reinsurance", "insurer", "insured", "beneficiary", "annuity",
        "property insurance", "casualty", "workers compensation", "health plan",
        "risk assessment", "loss ratio", "co-pay", "copay", "coinsurance",
    ],
    "K12HigherEdu": [
        "education", "school", "university", "college", "student", "teacher",
        "classroom", "curriculum", "syllabus", "lecture", "professor",
        "kindergarten", "elementary", "middle school", "high school",
        "undergraduate", "graduate", "phd", "dissertation", "thesis",
        "tuition", "scholarship", "campus", "semester", "academic",
        "enrollment", "gpa", "homework", "exam", "degree",
    ],
    "Retail": [
        "retail", "shopping", "store", "purchase", "consumer", "product",
        "e-commerce", "ecommerce", "marketplace", "inventory", "merchandise",
        "discount", "coupon", "checkout", "cart", "order", "shipping",
        "customer service", "return policy", "brand", "wholesale",
        "supply chain", "point of sale", "barcode", "sku",
    ],
    "ScienceTech": [
        "science", "technology", "research", "experiment", "hypothesis",
        "laboratory", "scientific", "innovation", "engineering", "physics",
        "chemistry", "biology", "computer science", "artificial intelligence",
        "machine learning", "algorithm", "software", "hardware", "robotics",
        "quantum", "nanotechnology", "biotechnology", "genome", "semiconductor",
    ],
    "Manufactory": [
        "manufacturing", "factory", "production", "assembly", "industrial",
        "automation", "quality control", "supply chain", "lean manufacturing",
        "six sigma", "cnc", "machining", "welding", "fabrication",
        "warehouse", "logistics", "iso 9001", "defect", "tooling",
        "injection molding", "stamping", "forging", "casting",
    ],
    "PatientHistoryDictation": [
        "patient history", "medical history", "chief complaint", "diagnosis",
        "symptoms", "medication", "prescription", "allergy", "vital signs",
        "blood pressure", "heart rate", "temperature", "physical examination",
        "medical record", "clinical notes", "dictation", "hpi",
        "history of present illness", "past medical history", "family history",
        "surgical history", "review of systems", "assessment and plan",
    ],
    "Energy": [
        "energy", "electricity", "power plant", "renewable", "solar",
        "wind energy", "hydroelectric", "nuclear energy", "fossil fuel",
        "oil and gas", "petroleum", "natural gas", "coal", "biomass",
        "geothermal", "grid", "transmission", "kilowatt", "megawatt",
        "energy efficiency", "carbon emission", "utility", "turbine",
        "photovoltaic", "battery storage",
    ],
    "Sustain": [
        "sustainability", "sustainable", "climate change", "carbon footprint",
        "greenhouse gas", "recycling", "renewable energy", "biodiversity",
        "conservation", "ecosystem", "environmental", "green energy",
        "circular economy", "zero waste", "esg", "carbon neutral",
        "net zero", "deforestation", "pollution", "clean energy",
        "sustainable development", "paris agreement",
    ],
    "Media": [
        "media", "journalism", "news", "broadcast", "television", "radio",
        "podcast", "streaming", "social media", "content creator",
        "advertising", "marketing", "public relations", "entertainment",
        "film", "movie", "documentary", "newspaper", "magazine",
        "digital media", "influencer", "viral", "audience", "ratings",
    ],
    "CapitalMarket": [
        "capital market", "stock market", "equity", "bond", "securities",
        "trading", "investment", "portfolio", "hedge fund", "mutual fund",
        "ipo", "dividend", "market cap", "bull market", "bear market",
        "derivative", "futures", "options", "commodities", "forex",
        "financial analyst", "wall street", "nasdaq", "dow jones", "s&p 500",
        "asset management", "venture capital", "private equity",
    ],
    "LifeHealth": [
        "health", "wellness", "nutrition", "diet", "exercise", "fitness",
        "mental health", "therapy", "chronic disease", "diabetes",
        "cardiovascular", "cancer", "obesity", "prevention", "public health",
        "epidemiology", "vaccine", "immunization", "healthcare",
        "life expectancy", "mortality", "morbidity", "clinical trial",
    ],
    "DoctorPatientConsultation": [
        "doctor", "physician", "consultation", "patient", "diagnosis",
        "treatment", "prognosis", "referral", "follow-up", "appointment",
        "medical advice", "clinical", "outpatient", "inpatient", "hospital",
        "emergency room", "primary care", "specialist", "surgeon",
        "telemedicine", "telehealth", "medical examination",
    ],
    "Banking": [
        "bank", "banking", "deposit", "withdrawal", "savings account",
        "checking account", "loan", "mortgage", "credit", "debit",
        "interest rate", "atm", "wire transfer", "online banking",
        "branch", "fdic", "treasury", "central bank", "federal reserve",
        "commercial bank", "investment bank", "fintech", "cryptocurrency",
    ],
}

DOMAIN_COLORS = {
    "Banking": "#4e79a7", "CapitalMarket": "#f28e2b",
    "DoctorPatientConsultation": "#e15759", "Energy": "#76b7b2",
    "Gaming": "#59a14f", "Insurance": "#edc948", "K12HigherEdu": "#b07aa1",
    "LifeHealth": "#ff9da7", "Manufactory": "#9c755f", "Media": "#bab0ac",
    "PatientHistoryDictation": "#86bcb6", "Retail": "#8cd17d",
    "ScienceTech": "#499894", "Sustain": "#f1ce63",
}


# ─── Core functions ───────────────────────────────────────────────────────────

def parse_yaml_domains(config_path):
    """Extract domain names from a YAML config file."""
    domains = []
    with open(config_path) as f:
        for line in f:
            m = re.search(r'test_name:\s*\S+/(\S+)', line)
            if m:
                domains.append(m.group(1))
    return list(dict.fromkeys(domains))  # dedupe, preserve order


def classify_text(text, domain_keywords):
    """Classify a text into zero or more domains based on keyword matching."""
    text_lower = text.lower()
    matched = []
    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score >= 2:
            matched.append(domain)
    return matched


def compute_text_stats(texts):
    """Compute statistics for a list of texts."""
    if not texts:
        return {"count": 0}
    lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    n = len(texts)
    wc_sorted = sorted(word_counts)
    return {
        "count": n,
        "avg_char_length": sum(lengths) / n,
        "median_char_length": sorted(lengths)[n // 2],
        "min_char_length": min(lengths),
        "max_char_length": max(lengths),
        "avg_word_count": sum(word_counts) / n,
        "median_word_count": wc_sorted[n // 2],
        "total_words": sum(word_counts),
        "total_chars": sum(lengths),
        "p10_words": wc_sorted[int(n * 0.1)],
        "p25_words": wc_sorted[int(n * 0.25)],
        "p50_words": wc_sorted[int(n * 0.5)],
        "p75_words": wc_sorted[int(n * 0.75)],
        "p90_words": wc_sorted[int(n * 0.9)],
    }


def download_and_sample(name, split, text_field, sample_size):
    """Download a sample from a HF dataset using streaming."""
    from datasets import load_dataset

    print(f"\n  Downloading: {name} (split={split}, sample={sample_size})")
    try:
        ds = load_dataset(name, split=split, streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            text = item.get(text_field, "")
            if text and len(text.strip()) > 50:
                samples.append({"text": text, "source": name})
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} rows, kept {len(samples)} samples...")
        print(f"    Collected {len(samples)} valid samples")
        return samples
    except Exception as e:
        print(f"    Error loading {name}: {e}")
        return []


# ─── HTML report generator ────────────────────────────────────────────────────

def pct_color(v):
    if v == 0:
        return "#f8f9fa"
    r = int(255 - v * 1.8)
    g = int(255 - v * 0.8)
    return f"rgb({max(r, 60)},{max(g, 100)},255)"


def generate_html(domains, domain_keywords, domain_texts, all_samples, output_dir):
    """Generate the full HTML analysis report."""
    total = len(all_samples)
    unclassified = sum(1 for s in all_samples if not classify_text(s["text"], domain_keywords))

    # Compute all stats
    domain_stats = {}
    for d in domains:
        texts = domain_texts.get(d, [])
        domain_stats[d] = compute_text_stats(texts)

    # Co-occurrence matrix
    text_domain_sets = []
    for s in all_samples:
        ds = set(classify_text(s["text"], domain_keywords))
        if ds:
            text_domain_sets.append(ds)

    co_occ = {}
    for d1 in domains:
        co_occ[d1] = {}
        count_d1 = sum(1 for ds in text_domain_sets if d1 in ds)
        for d2 in domains:
            if d1 == d2 or count_d1 == 0:
                co_occ[d1][d2] = 0
            else:
                co_occ[d1][d2] = int(sum(1 for ds in text_domain_sets if d1 in ds and d2 in ds) / count_d1 * 100)

    # Source distribution
    source_counts = Counter(s["source"] for s in all_samples)

    max_count = max((st["count"] for st in domain_stats.values()), default=1)
    max_p90 = max((st.get("p90_words", 0) for st in domain_stats.values()), default=1) * 1.1

    short = {d: d[:5] for d in domains}
    classified = total - unclassified

    # Build HTML
    p = []
    p.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Domain Text Data Analysis</title>
<style>
:root{{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--text:#212529;--muted:#6c757d;--accent:#4e79a7}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:0 0 60px}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:40px 0;text-align:center}}
.header h1{{font-size:2rem;font-weight:700;margin-bottom:8px}}
.header p{{opacity:.8;font-size:1.05rem}}
.container{{max-width:1280px;margin:0 auto;padding:0 24px}}
.stats-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:32px 0}}
.stat-card{{background:var(--card);border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}}
.stat-card .num{{font-size:2rem;font-weight:700;color:var(--accent)}}
.stat-card .label{{font-size:.85rem;color:var(--muted);margin-top:4px}}
h2{{font-size:1.4rem;margin:40px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--accent)}}
.card{{background:var(--card);border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border)}}
th{{background:#f1f3f5;font-weight:600;position:sticky;top:0}}
tr:hover{{background:#f8f9fa}}
.bar-inner{{height:22px;border-radius:4px;display:inline-block;vertical-align:middle;min-width:2px}}
.bar-label{{margin-left:8px;font-size:.82rem;color:var(--muted)}}
.heatmap table{{font-size:.78rem;text-align:center}}
.heatmap th,.heatmap td{{padding:6px 4px;min-width:48px}}
.heatmap td{{font-weight:600}}
.chip{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:600;margin-right:4px}}
.chip-good{{background:#d3f9d8;color:#2b8a3e}}
.chip-warn{{background:#fff3bf;color:#e67700}}
.chip-crit{{background:#ffe3e3;color:#c92a2a}}
.domain-tag{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:middle}}
.box-row{{display:flex;align-items:center;margin-bottom:6px;font-size:.82rem}}
.box-lbl{{width:170px;text-align:right;padding-right:12px;flex-shrink:0}}
.box-track{{flex:1;height:18px;position:relative;background:#f1f3f5;border-radius:4px}}
.box-whisker{{position:absolute;top:6px;height:6px;background:#adb5bd;border-radius:2px}}
.box-box{{position:absolute;top:2px;height:14px;border-radius:3px;opacity:.85}}
.box-med{{position:absolute;top:0;width:2px;height:18px;background:#fff;border-radius:1px}}
.box-vals{{width:120px;padding-left:8px;color:var(--muted);font-size:.75rem;flex-shrink:0}}
.sources-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.source-card{{background:var(--card);border-radius:10px;padding:18px;border-left:4px solid var(--accent);box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.source-card h4{{margin-bottom:4px;font-size:.95rem}}
.source-card p{{font-size:.82rem;color:var(--muted);margin:2px 0}}
footer{{text-align:center;padding:40px 0 20px;color:var(--muted);font-size:.82rem}}
</style></head><body>
<div class="header"><h1>Domain Text Data Analysis Report</h1>
<p>HuggingFace Open-Source Datasets &middot; {len(domains)} Target Domains</p></div>
<div class="container">
<div class="stats-row">
  <div class="stat-card"><div class="num">{total:,}</div><div class="label">Total Samples</div></div>
  <div class="stat-card"><div class="num">{len(domains)}</div><div class="label">Target Domains</div></div>
  <div class="stat-card"><div class="num">{len(source_counts)}</div><div class="label">HF Datasets</div></div>
  <div class="stat-card"><div class="num">{classified:,}</div><div class="label">Domain-Classified</div></div>
  <div class="stat-card"><div class="num">{classified/total*100:.1f}%</div><div class="label">Classification Rate</div></div>
</div>""")

    # Source datasets
    p.append('<h2>Source Datasets</h2><div class="sources-grid">')
    for src, cnt in source_counts.most_common():
        p.append(f'<div class="source-card"><h4>{html_mod.escape(src)}</h4><p><strong>{cnt:,}</strong> samples ({cnt/total*100:.1f}%)</p></div>')
    p.append('</div>')

    # Domain coverage table
    p.append('<h2>Domain Coverage</h2><div class="card"><table><thead><tr><th>Domain</th><th>Samples</th><th style="width:40%">Distribution</th><th>Avg Words</th><th>Total Words</th></tr></thead><tbody>')
    for d in domains:
        st = domain_stats[d]
        cnt = st["count"]
        pct = cnt / total * 100 if total else 0
        bar_w = cnt / max_count * 100
        col = DOMAIN_COLORS.get(d, "#4e79a7")
        avg_w = st.get("avg_word_count", 0)
        tot_w = st.get("total_words", 0)
        p.append(f'<tr><td><span class="domain-tag" style="background:{col}"></span>{d}</td><td>{cnt:,}</td>'
                 f'<td><span class="bar-inner" style="width:{bar_w:.1f}%;background:{col}"></span><span class="bar-label">{pct:.1f}%</span></td>'
                 f'<td>{avg_w:,.0f}</td><td>{tot_w:,}</td></tr>')
    p.append('</tbody></table></div>')

    # Box plot distribution
    p.append('<h2>Text Length Distribution (words)</h2><div class="card">')
    for d in domains:
        st = domain_stats[d]
        if st["count"] == 0:
            continue
        col = DOMAIN_COLORS.get(d, "#4e79a7")
        def pos(v):
            return v / max_p90 * 100 if max_p90 else 0
        p.append(f'<div class="box-row"><div class="box-lbl"><span class="domain-tag" style="background:{col}"></span>{d}</div>'
                 f'<div class="box-track">'
                 f'<div class="box-whisker" style="left:{pos(st["p10_words"]):.1f}%;width:{pos(st["p90_words"]-st["p10_words"]):.1f}%"></div>'
                 f'<div class="box-box" style="left:{pos(st["p25_words"]):.1f}%;width:{pos(st["p75_words"]-st["p25_words"]):.1f}%;background:{col}"></div>'
                 f'<div class="box-med" style="left:{pos(st["p50_words"]):.1f}%"></div>'
                 f'</div><div class="box-vals">P50={st["p50_words"]:,} &mu;={int(st["avg_word_count"]):,}</div></div>')
    p.append('<p style="margin-top:12px;font-size:.8rem;color:var(--muted)">Whiskers=P10-P90 | Box=P25-P75 | White=Median</p></div>')

    # Heatmap
    p.append('<h2>Domain Co-occurrence</h2><div class="card heatmap" style="overflow-x:auto">'
             '<p style="font-size:.85rem;color:var(--muted);margin-bottom:12px">% of rows in domain A also matching domain B</p>'
             '<table><thead><tr><th></th>')
    for d in domains:
        p.append(f'<th>{short[d]}</th>')
    p.append('</tr></thead><tbody>')
    for d1 in domains:
        p.append(f'<tr><th>{short[d1]}</th>')
        for d2 in domains:
            if d1 == d2:
                p.append('<td style="background:#e9ecef;color:#adb5bd">&mdash;</td>')
            else:
                v = co_occ[d1][d2]
                p.append(f'<td style="background:{pct_color(v)}">{v}%</td>')
        p.append('</tr>')
    p.append('</tbody></table></div>')

    # Per-domain cards
    p.append('<h2>Per-Domain Details</h2>')
    for d in domains:
        st = domain_stats[d]
        cnt = st["count"]
        col = DOMAIN_COLORS.get(d, "#4e79a7")
        cls = "chip-good" if cnt >= 1000 else ("chip-warn" if cnt >= 200 else "chip-crit")
        txt = "Good" if cnt >= 1000 else ("Low" if cnt >= 200 else "Critical")
        p.append(f'<div class="card" style="border-left:4px solid {col}">'
                 f'<h3 style="margin:0 0 8px"><span class="domain-tag" style="background:{col}"></span>{d} '
                 f'<span class="chip {cls}">{txt} &mdash; {cnt:,} samples</span></h3>'
                 f'<table style="width:auto"><tr>')
        for label, val in [("Avg chars", st.get("avg_char_length", 0)),
                           ("Median chars", st.get("median_char_length", 0)),
                           ("Avg words", st.get("avg_word_count", 0)),
                           ("Total words", st.get("total_words", 0))]:
            p.append(f'<td style="padding:4px 16px 4px 0"><span style="color:var(--muted);font-size:.78rem">{label}</span><br><strong>{val:,.0f}</strong></td>')
        p.append('</tr></table></div>')

    p.append('<footer>Generated by domain-data-search skill</footer></div></body></html>')

    html_path = os.path.join(output_dir, "domain_analysis_report.html")
    with open(html_path, "w") as f:
        f.write("".join(p))
    print(f"  HTML report: {html_path}")
    return domain_stats


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Domain data search & analysis")
    parser.add_argument("--domains", help="Comma-separated domain names")
    parser.add_argument("--config", help="YAML config file to extract domains from")
    parser.add_argument("--datasets", nargs="+",
                        help="HF datasets as 'name:split:text_field:sample_size'",
                        default=["DKYoon/SlimPajama-6B:train:text:50000",
                                 "timaeus/pile-pubmed_abstracts:train:text:20000",
                                 "FinanceMTEB/financial_phrasebank:train:sentence:5000"])
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse domains
    if args.config:
        domains = parse_yaml_domains(args.config)
        print(f"Extracted {len(domains)} domains from {args.config}: {domains}")
    elif args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    else:
        domains = sorted(DEFAULT_DOMAIN_KEYWORDS.keys())
        print(f"Using default {len(domains)} domains")

    # Build keyword dict (use defaults, add empty for unknown domains)
    domain_keywords = {}
    for d in domains:
        if d in DEFAULT_DOMAIN_KEYWORDS:
            domain_keywords[d] = DEFAULT_DOMAIN_KEYWORDS[d]
        else:
            print(f"  Warning: No keywords defined for domain '{d}', using domain name as keyword")
            domain_keywords[d] = [d.lower(), d.lower().replace(" ", "")]

    # Download samples
    print(f"\nDownloading from {len(args.datasets)} datasets...")
    all_samples = []
    for ds_spec in args.datasets:
        parts = ds_spec.split(":")
        if len(parts) != 4:
            print(f"  Skipping invalid spec: {ds_spec} (expected name:split:field:size)")
            continue
        name, split, field, size = parts[0], parts[1], parts[2], int(parts[3])
        samples = download_and_sample(name, split, field, size)
        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples):,}")

    # Classify
    domain_texts = defaultdict(list)
    for s in all_samples:
        for d in classify_text(s["text"], domain_keywords):
            domain_texts[d].append(s["text"])

    # Generate outputs
    print("\nGenerating report...")
    domain_stats = generate_html(domains, domain_keywords, domain_texts, all_samples, args.output_dir)

    # Save JSON
    summary = {}
    for d in domains:
        texts = domain_texts.get(d, [])
        st = domain_stats.get(d, compute_text_stats(texts))
        summary[d] = {
            "sample_count": st["count"],
            "stats": st,
            "sample_texts": [t[:200] for t in texts[:3]],
        }
    json_path = os.path.join(args.output_dir, "domain_analysis_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON data:   {json_path}")

    # Save CSV
    rows = []
    for d in domains:
        st = domain_stats.get(d, {"count": 0})
        rows.append({
            "domain": d,
            "sample_count": st.get("count", 0),
            "avg_char_length": st.get("avg_char_length", 0),
            "avg_word_count": st.get("avg_word_count", 0),
            "total_words": st.get("total_words", 0),
        })
    csv_path = os.path.join(args.output_dir, "domain_analysis_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV summary: {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Domain':<30} {'Samples':>10} {'Status'}")
    print(f"{'='*60}")
    for d in domains:
        cnt = domain_stats.get(d, {}).get("count", 0)
        status = "Good" if cnt >= 1000 else ("Low" if cnt >= 200 else "CRITICAL")
        sym = "OK" if cnt >= 1000 else ("!!" if cnt >= 200 else "XX")
        print(f"{d:<30} {cnt:>10,}  [{sym}] {status}")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()

---
name: asr-entity-error-analysis
description: "Analyze entity error rate (EER) and entity word error rate (EWER) from ASR result_details JSONL files by comparing hypothesis text against entity-annotated Transcription. EER = entities_with_errors / total_entities. EWER = entity_word_errors / total_entity_words. Use when inspecting entity recognition accuracy, diagnosing entity-level substitution/deletion/insertion patterns, computing per-entity-type error rates, or generating HTML reports that highlight errors only on entity spans. Entities are marked with <NE> or <NE:type> tags in the Transcription field."
---

# ASR Entity Error Analysis

Analyze entity recognition accuracy in ASR output: extract named entities from `<NE>` / `<NE:type>` tags in the `Transcription` field, align hypothesis words against reference, and compute error rates exclusively on entity spans. Produces a standalone HTML report optimized for very long transcripts with entity-only error highlighting.

For *general word-level error analysis* (WER on all words), use the **asr-word-error-analysis** skill instead.

## When to Use

- You have `result_details*.jsonl` from an ASR eval where `Transcription` contains `<NE>` or `<NE:type>` entity tags.
- You need Entity Error Rate (EER = entities_with_errors / total_entities) or Entity Word Error Rate (EWER = entity_word_errors / total_entity_words).
- You want to see which entities the model gets wrong most often.
- You need an HTML report that highlights entity errors in context of long transcripts.
- You want per-entity-type breakdown (e.g., `<NE:name>` vs generic `<NE>`).

## Workflow

1. Identify the `result_details*.jsonl` file by explicit path or by `--model` + `--dataset` auto-discovery.
2. Run [analyze_entity_errors.py](./scripts/analyze_entity_errors.py).
3. Review artifacts: start with `summary.json` for EER/EWER overview, then `entity_errors.csv` for per-entity error details, then `report.html` for visual inspection with entity highlighting.

## Script

```bash
python .github/skills/asr-entity-error-analysis/scripts/analyze_entity_errors.py \
  --input-path path/to/result_details.jsonl \
  --output-dir tmp/asr-entity-error-analysis \
  --write-html
```

With model auto-discovery:

```bash
python .github/skills/asr-entity-error-analysis/scripts/analyze_entity_errors.py \
  --model baseline \
  --dataset en-US-entity-v3/Insurance \
  --output-dir tmp/asr-entity-error-analysis/baseline/en-US-entity-v3/Insurance \
  --write-html
```

See [options reference](./references/options.md) for the full flag table.

## Output Artifacts

| File | Content |
|------|---------|
| `summary.json` | Dataset-level EER, EWER, entity counts, error totals, per-type breakdown |
| `entity_errors.csv` | Every entity instance ranked by error severity with ref/hyp spans and error type |
| `entity_substitutions.csv` | Ranked entity-level substitution pairs (ref entity → hyp entity) |
| `entity_type_breakdown.csv` | Error rates grouped by entity type (`NE`, `NE:name`, etc.) |
| `report.html` | Standalone visual report with EER/EWER KPI cards, per-utterance view showing full transcript with entity errors highlighted, and scrollable long-text layout |

## Interpreting Results

- **High entity substitution rate**: model produces wrong words for entity spans. Check `entity_substitutions.csv` for patterns.
- **High entity deletion rate**: model drops entity words entirely. Common for proper nouns and technical terms.
- **High entity insertion rate**: model adds extra words within entity boundaries.
- **Per-type breakdown**: `entity_type_breakdown.csv` shows whether typed entities (`NE:name`) are harder than generic `NE`.
- **HTML report**: each utterance shows the full transcript with entity spans highlighted — correct entities in green, errors in red/yellow, with inline diff on the entity part only.

See [format reference](./references/format.md) for entity tag notation and input schema.

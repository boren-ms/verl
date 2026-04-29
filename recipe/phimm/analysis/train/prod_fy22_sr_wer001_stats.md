# prod_fy22_sr_wer001 Dataset Statistics

**Source:** `az://orngwus2cresco/data/boren/data/verl/gen_filtered/prod_fy22_sr_wer001`

**Generation config:** [gen_prod_fy22_sr.yaml](../../config/gen_prod_fy22_sr.yaml) — model `Phi4-7b-STT-2603-SR2`, WER filter `[0.01, 2000]`, `temperature=0.0`

**Source data:** `hcv2_fy22_info_tag` (FY22 human caption v2 with adjusted boundary + bias LM)

> **Note:** Statistics below are computed on **part-004** (583,839 rows). Distribution percentages are representative of the full dataset. Total row count is from all 5 parts.

---

## **Summary**

| Metric | Value |
|--------|-------|
| Total utterances | 4,585,165 |
| Parts | 5 (part-000: 1,000,364 · part-001: 1,000,411 · part-002: 1,000,402 · part-003: 1,000,149 · part-004: 583,839) |
| Total size | 3.4 GiB |
| Avg words per utterance | 66.0 |
| Median words per utterance | 68.0 |
| Min / Max words | 1 / 113 |

## **WER Statistics**

| Metric | Value |
|--------|-------|
| Mean WER | 27.43% |
| Median WER | 8.33% |
| P25 / P75 | 3.70% / 21.21% |
| P90 | 94.83% |
| P95 | 105.56% |
| P99 | 215.91% |
| Min / Max WER | 1.00% / 17800.00% |
| Mean Edge WER | 18.09% |
| Median Edge WER | 1.54% |
| Zero Edge WER | 222,875 (38.2%) |

## **Formatted Flag**

| Formatted | Count | % | Mean WER | Median WER |
|-----------|-------|---|----------|------------|
| False | 464,778 | 79.6% | 23.18% | 7.79% |
| True | 119,061 | 20.4% | 44.03% | 11.76% |

## **Language Distribution**

Only languages with ≥10 utterances shown. "(not set)" = model did not output a language tag.

| Language | Count | % | Mean WER | Median WER | Avg Words |
|----------|-------|---|----------|------------|-----------|
| (not set) | 458,220 | 78.5% | 21.54% | 7.50% | 67.3 |
| English | 93,591 | 16.0% | 30.55% | 8.64% | 60.6 |
| Russian | 18,415 | 3.2% | 99.75% | 100.00% | 67.6 |
| German | 8,000 | 1.4% | 100.44% | 97.87% | 58.6 |
| Japanese | 1,360 | 0.2% | 95.67% | 100.00% | 59.2 |
| Spanish | 1,072 | 0.2% | 106.16% | 100.00% | 51.3 |
| Mandarin | 718 | 0.1% | 100.72% | 100.00% | 17.7 |
| Hindi | 662 | 0.1% | 196.72% | 179.80% | 40.6 |
| French | 471 | 0.1% | 136.47% | 110.34% | 65.2 |
| ASR | 385 | 0.1% | 26.63% | 20.41% | 73.3 |
| Korean | 199 | 0.0% | 116.06% | 100.00% | 47.1 |
| Python | 97 | 0.0% | 19.95% | 8.45% | 67.7 |
| Thai | 94 | 0.0% | 94.59% | 100.00% | 54.1 |
| Arabic | 76 | 0.0% | 132.06% | 100.00% | 46.3 |
| Italian | 72 | 0.0% | 146.25% | 103.21% | 38.9 |
| Portuguese | 68 | 0.0% | 311.56% | 120.00% | 46.4 |
| TXT | 61 | 0.0% | 41.87% | 29.23% | 73.3 |
| JSON | 46 | 0.0% | 16.74% | 10.42% | 74.6 |
| CSS | 36 | 0.0% | 16.60% | 7.94% | 71.0 |
| Turkish | 32 | 0.0% | 177.22% | 100.00% | 28.2 |
| Polish | 22 | 0.0% | 140.82% | 116.39% | 50.5 |
| Czech | 21 | 0.0% | 372.97% | 100.00% | 24.3 |
| Dutch | 21 | 0.0% | 142.25% | 108.77% | 41.3 |
| Hebrew | 18 | 0.0% | 150.96% | 100.00% | 41.9 |
| Txt | 14 | 0.0% | 9.61% | 6.08% | 68.4 |
| Language | 13 | 0.0% | 21.55% | 21.05% | 89.3 |
| Norwegian | 11 | 0.0% | 1766.99% | 139.53% | 33.1 |

**Observation:** Non-English languages (Russian, German, Japanese, etc.) have ~100% WER because the model transcribes English text for non-English audio, resulting in near-total mismatches. The "(not set)" rows are cases where the model did not emit a structured language tag (unformatted responses).

## **WER Distribution (5% buckets)**

| Bucket | Count | % |
|--------|-------|---|
| 0–5% | 195,853 | 33.5% |
| 5–10% | 127,474 | 21.8% |
| 10–15% | 68,738 | 11.8% |
| 15–20% | 38,084 | 6.5% |
| 20–25% | 23,331 | 4.0% |
| 25–30% | 15,717 | 2.7% |
| 30–35% | 10,469 | 1.8% |
| 35–40% | 7,384 | 1.3% |
| 40–45% | 6,081 | 1.0% |
| 45–50% | 4,446 | 0.8% |
| 50–55% | 4,649 | 0.8% |
| 55–60% | 3,457 | 0.6% |
| 60–65% | 3,288 | 0.6% |
| 65–70% | 2,800 | 0.5% |
| 70–75% | 2,542 | 0.4% |
| 75–80% | 2,489 | 0.4% |
| 80–85% | 2,432 | 0.4% |
| 85–90% | 2,565 | 0.4% |
| 90–95% | 3,812 | 0.7% |
| 95–100% | 8,330 | 1.4% |
| 100%+ | 49,898 | 8.5% |

**55.4%** of utterances have WER ≤ 10%. **8.5%** have WER > 100% (insertions exceed reference length).

## **Word Count Distribution (5-word buckets)**

| Bucket (words) | Count | % |
|-----------------|-------|---|
| 0–5 | 1,383 | 0.2% |
| 5–10 | 2,255 | 0.4% |
| 10–15 | 2,625 | 0.4% |
| 15–20 | 3,586 | 0.6% |
| 20–25 | 4,618 | 0.8% |
| 25–30 | 6,377 | 1.1% |
| 30–35 | 9,086 | 1.6% |
| 35–40 | 12,853 | 2.2% |
| 40–45 | 18,905 | 3.2% |
| 45–50 | 27,595 | 4.7% |
| 50–55 | 38,627 | 6.6% |
| 55–60 | 50,575 | 8.7% |
| 60–65 | 62,096 | 10.6% |
| 65–70 | 70,011 | 12.0% |
| 70–75 | 73,697 | 12.6% |
| 75–80 | 71,184 | 12.2% |
| 80–85 | 58,489 | 10.0% |
| 85–90 | 40,030 | 6.9% |
| 90–95 | 22,220 | 3.8% |
| 95–100 | 7,016 | 1.2% |
| 100–105 | 523 | 0.1% |
| 105–110 | 79 | 0.0% |
| 110–115 | 9 | 0.0% |

The distribution is approximately normal, centered around 65–75 words per utterance, consistent with the FY22 human caption chunking strategy.

## **Language × Formatted Cross-Tab** (top languages)

| Language | Unformatted | Formatted |
|----------|-------------|-----------|
| (not set) | 449,737 | 8,483 |
| English | 2,335 | 91,256 |
| Russian | 9,148 | 9,267 |
| German | 2,394 | 5,606 |
| Japanese | 496 | 864 |
| Spanish | 31 | 1,041 |
| Mandarin | 8 | 710 |
| Hindi | 13 | 649 |
| French | 210 | 261 |

**Observation:** English is overwhelmingly formatted (97.5%), while "(not set)" is overwhelmingly unformatted (98.1%). Non-English languages skew formatted because the structured response format includes a language tag.

## **Schema**

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | string | User prompt (e.g., `<\|audio_1\|>Transcribe the audio clip into text.`) |
| `text` | string | Ground-truth transcription |
| `id` | null | Always null |
| `keywords` | list\<string\> | Named entities / keywords (empty in part-004) |
| `audio_path` | null | Always null (audio stored as chunks) |
| `audio_chunk` | string | Audio chunk reference: `path.audio:count:index` |
| `lang` | string | Detected language from model response (may be null) |
| `formatted` | bool | Whether model produced structured `<ASR>` response |
| `response` | string | Parsed model transcription output |
| `raw_response` | string | Full raw model output |
| `wer` | double | Word Error Rate vs ground truth |
| `edge_wer` | double | Edge WER (ignoring boundary effects) |

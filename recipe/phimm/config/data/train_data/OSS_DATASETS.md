# Open-Source ASR Training Datasets

This document describes the open-source speech datasets used in `oss.yaml` for multilingual ASR training.

## Summary

| Dataset | Languages | Hours (used) | Examples | Avg Duration | Source |
|---|---|---|---|---|-****--|
| Multilingual LibriSpeech (MLS) | en, de, es, fr, it, pt, pl, nl | ~8,039 h | ~1.93M | ~15s | Read audiobooks (LibriVox) |
| Open Chinese ASR (WenetSpeech) | zh | 3,000 h | 4.29M | 2.52s | YouTube & Podcast |
| GigaSpeech | en | 2,499 h | 2.12M | 4.24s | Audiobooks, podcasts, YouTube |
| Common Voice 15 | en, de, es, fr, it, ja, pt, zh + tier2 | ~7,151 h | ~4.85M | ~5.3s | Crowdsourced read speech |
| VoxPopuli | en, de, es, fr, it + tier2 | ~2,995 h | ~1.08M | ~10s | European Parliament recordings |
| FLEURS | multilingual + tier2 | ~372 h | ~0.12M | ~11s | Read sentences (102 languages) |
| OpenASR – AMI | en | ~100 h | 0.11M × 2 | varies | Meeting recordings |
| OpenASR – Earnings | en | ~119 h | 0.05M × 2 | varies | Earnings calls |
| WavLLM chunk ASR | en | varies | varies | varies | Mixed ASR data |

**Total: ~24,000+ hours across 15+ languages**

---

## 1. Multilingual LibriSpeech (MLS)

- **Paper:** Pratap et al., "MLS: A Large-Scale Multilingual Dataset for Speech Research" (2020). [arXiv:2012.03411](https://arxiv.org/abs/2012.03411)
- **HuggingFace:** [facebook/multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech)
- **License:** CC-BY-4.0

MLS is a large multilingual corpus derived from read audiobooks from LibriVox, consisting of 8 languages: English, German, Dutch, Spanish, French, Italian, Portuguese, and Polish. The full corpus includes ~44.5K hours of English and ~6K hours for other languages.

**Subsets used:**

| Config | Language | Hours | Examples | Avg Duration |
|---|---|---|---|---|
| `open_mls_en_asr_2k` | English | 1,999.67 h | 0.484M | 14.87s |
| `open_mls_de_asr` | German | 1,966.50 h | 0.470M | 15.06s |
| `open_mls_es_asr` | Spanish | 929.73 h | 0.224M | 14.97s |
| `open_mls_fr_asr` | French | 1,076.58 h | 0.258M | 15.01s |
| `open_mls_it_asr` | Italian | 247.38 h | 0.060M | 14.94s |
| `open_mls_pt_asr` | Portuguese | 160.96 h | 0.038M | 15.44s |
| `open_mls_pl_asr` | Polish | 103.65 h | 0.025M | 14.90s |
| `open_mls_nl_asr` | Dutch | 1,554.24 h | 0.374M | 14.95s |

**Key characteristics:**
- Read speech from audiobooks – clean recording conditions
- Consistent average utterance duration (~15s)
- Suitable for large-scale multilingual ASR training

---

## 2. Open Chinese ASR (WenetSpeech)

- **Paper:** Zhang et al., "WenetSpeech: A 10000+ Hours Multi-domain Mandarin Corpus for Speech Recognition" (2022). [arXiv:2110.03370](https://arxiv.org/abs/2110.03370)
- **HuggingFace:** [wenet-e2e/wenetspeech](https://huggingface.co/datasets/wenet-e2e/wenetspeech)
- **License:** CC-BY-4.0 (non-commercial)

WenetSpeech is a 10,000+ hour multi-domain Mandarin speech corpus collected from YouTube and Podcast, covering a variety of speaking styles, scenarios, domains, topics, and noisy conditions. An OCR-based method generates audio/text segmentation for YouTube data, while a high-quality ASR system generates candidates for Podcast data.

**Subset used:**

| Config | Language | Hours | Examples | Avg Duration |
|---|---|---|---|---|
| `open_zh_asr_3k` | Chinese | 3,000.23 h | 4.290M | 2.52s |

**Key characteristics:**
- Multi-domain: spontaneous and read speech
- Short average utterance duration (2.52s) – typical for Chinese ASR segmentation
- Large number of examples (4.29M) for fine-grained Chinese speech coverage

---

## 3. GigaSpeech

- **Paper:** Chen et al., "GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio" (Interspeech 2021). [arXiv:2106.06909](https://arxiv.org/abs/2106.06909)
- **HuggingFace:** [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)
- **License:** Apache-2.0 (research use)

GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high-quality labeled audio. Audio is collected from audiobooks (~2,655 h), podcasts (~3,498 h), and YouTube (~3,845 h), covering both read and spontaneous speaking styles across diverse topics (arts, science, sports, etc.).

**Subset used:**

| Config | Language | Hours | Examples | Avg Duration |
|---|---|---|---|---|
| `asr_chunk_gigaspeech_2k5` | English | 2,498.74 h | 2.124M | 4.24s |

**Key characteristics:**
- Multi-domain: audiobooks, podcasts, and YouTube
- Mix of read and spontaneous speech styles
- Diverse topics and acoustic conditions

---

## 4. Common Voice 15 (CV15)

- **Paper:** Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus" (LREC 2020). [arXiv:1912.06670](https://arxiv.org/abs/1912.06670)
- **HuggingFace:** [mozilla-foundation/common_voice_15_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0)
- **License:** CC0 (public domain)

Common Voice is Mozilla's crowdsourcing project to create a free, open, massively multilingual speech dataset. Volunteers worldwide record and validate sentences via a web platform. As of version 15, the dataset covers 100+ languages with thousands of hours of validated speech. It is the largest open-source crowdsourced speech corpus.

**Subsets used:**

| Config | Language | Hours | Examples | Avg Duration |
|---|---|---|---|---|
| `asr_chunk_cv15_en` | English | 1,688.26 h | 1.070M | 5.68s |
| `asr_chunk_cv15_de` | German | 1,801.81 h | 1.136M | 5.71s |
| `asr_chunk_cv15_es` | Spanish | 901.72 h | 0.623M | 5.21s |
| `asr_chunk_cv15_fr` | French | 1,514.72 h | 1.055M | 5.17s |
| `asr_chunk_cv15_it` | Italian | 489.32 h | 0.333M | 5.29s |
| `asr_chunk_cv15_ja` | Japanese | 18.82 h | 0.014M | 4.97s |
| `asr_chunk_cv15_pt` | Portuguese | 48.12 h | 0.042M | 4.11s |
| `asr_chunk_cv15_zh` | Chinese | 84.63 h | 0.059M | 5.18s |
| `asr_chunk_cv15_tier2` | Other langs | 603.61 h | 0.486M | 4.47s |

**Key characteristics:**
- Crowdsourced read speech – diverse speaker demographics
- Variable recording quality (consumer microphones, varying environments)
- Broad accent and dialect coverage per language
- Some language subsets are repeated (upsampled) in the config for data balancing

---

## 5. VoxPopuli

- **Paper:** Wang et al., "VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation" (ACL 2021). [arXiv:2101.00390](https://arxiv.org/abs/2101.00390)
- **HuggingFace:** [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)
- **License:** CC0

VoxPopuli is a large-scale multilingual speech corpus collected from 2009–2020 European Parliament event recordings. It provides transcribed speech data for 18 languages plus 29 hours of accented English (15 L2 accents). The corpus contains ~1,791 hours of transcribed speech across all languages. Utterances are segmented to a maximum of 20 seconds using ASR force-alignment.

**Subsets used:**

| Config | Language | Hours | Examples | Avg Duration |
|---|---|---|---|---|
| `asr_chunk_voxpopuli_en` | English | 1,045.21 h | 0.365M | 10.31s |
| `asr_chunk_voxpopuli_de` | German | 529.29 h | 0.217M | 8.78s |
| `asr_chunk_voxpopuli_es` | Spanish | 303.85 h | 0.102M | 10.74s |
| `asr_chunk_voxpopuli_fr` | French | 411.40 h | 0.147M | 10.07s |
| `asr_chunk_voxpopuli_it` | Italian | 156.13 h | 0.045M | 12.45s |
| `asr_chunk_voxpopuli_tier2` | Other langs | 549.45 h | 0.201M | 9.84s |

**Key characteristics:**
- Formal parliamentary speech – mostly non-native English speakers
- Longer average utterance duration (~10s)
- High-quality recordings from institutional setting
- Some language subsets are repeated (upsampled) for data balancing

---

## 6. FLEURS

- **Paper:** Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech" (2022). [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)
- **HuggingFace:** [google/fleurs](https://huggingface.co/datasets/google/fleurs)
- **License:** CC-BY-4.0

FLEURS is the speech version of the FLoRes machine translation benchmark. It uses 2,009 n-way parallel sentences from FLoRes in 102 languages. Training sets have around 10 hours of supervision per language. Speakers of the train sets are different from dev/test speakers. Languages are grouped into seven geographical areas: Western Europe, Eastern Europe, Central-Asia/Middle-East/North-Africa, Sub-Saharan Africa, South-Asia, South-East Asia, and CJK languages.

**Subsets used:**

| Config | Languages | Hours | Examples | Avg Duration |
|---|---|---|---|---|
| `asr_chunk_fleurs` | Major langs | 159.75 h | 0.050M | 11.39s |
| `asr_chunk_fleurs_tier2` | Other langs | 212.54 h | 0.069M | 11.13s |

**Key characteristics:**
- N-way parallel sentences across 102 languages
- Read speech with diverse speakers
- Small per-language size (~10h) but broad language coverage
- Useful for multilingual representation learning

---

## 7. OpenASR Benchmarks

### 7a. AMI Meeting Corpus

- **Homepage:** [Edinburgh CSTR AMI](https://groups.inf.ed.ac.uk/ami/corpus/)
- **HuggingFace:** [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami)
- **License:** CC-BY-4.0

The AMI Meeting Corpus consists of 100 hours of meeting recordings using close-talking and far-field microphones, video cameras, and other synchronized signals. Meetings were recorded in English across three different rooms with different acoustic properties and include mostly non-native speakers. This is a standard evaluation benchmark for meeting transcription ASR.

**Subset used:**

| Config | Hours | Examples |
|---|---|---|
| `asr_chunk_openasr_ami` | ~100 h | 0.11M × 2 (upsampled) |

### 7b. Earnings-22

- **Paper:** Del Rio et al., "Earnings-22: A Practical Benchmark for Accents in the Wild" (Interspeech 2022). [arXiv:2203.15591](https://arxiv.org/abs/2203.15591)
- **HuggingFace:** [revdotcom/earnings22](https://huggingface.co/datasets/revdotcom/earnings22)
- **License:** CC-BY-SA-4.0

Earnings-22 is a 125-file, 119-hour corpus of English-language earnings calls gathered from global companies. It serves as a benchmark for real-world, accented speech ASR, featuring speakers with diverse accents from various countries of origin.

**Subset used:**

| Config | Hours | Examples |
|---|---|---|
| `asr_chunk_openasr_earnings` | ~119 h | 0.05M × 2 (upsampled) |

**Key characteristics (both):**
- Real-world speech: meeting and financial domain
- Challenging acoustic conditions (far-field, cross-talk, accented speech)
- Standard Open ASR Leaderboard evaluation benchmarks
- Upsampled (×2) in training config

---

## 8. WavLLM Chunk ASR

- **Paper:** Hu et al., "WavLLM: Towards Robust and Adaptive Speech Large Language Model" (EMNLP 2024 Findings). [arXiv:2404.00656](https://arxiv.org/abs/2404.00656)
- **GitHub:** [microsoft/SpeechT5/WavLLM](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)

WavLLM is a speech large language model with dual encoders (Whisper for semantic content + WavLM for speaker characteristics). The `wavllm_chunk_asr` data spec contains mixed ASR training data used for WavLLM's foundational ASR training stage, covering diverse speech recognition scenarios.

**Subset used:**

| Config | Description |
|---|---|
| `wavllm_chunk_asr` | Mixed ASR training data from WavLLM pipeline |

---

## Language Coverage Summary

| Language | Datasets Contributing | Approx. Total Hours |
|---|---|---|
| English (en) | MLS, GigaSpeech, CV15, VoxPopuli, AMI, Earnings, FLEURS | ~9,500 h |
| German (de) | MLS, CV15, VoxPopuli, FLEURS | ~4,300 h |
| French (fr) | MLS, CV15, VoxPopuli, FLEURS | ~3,000 h |
| Chinese (zh) | WenetSpeech, CV15, FLEURS | ~3,100 h |
| Spanish (es) | MLS, CV15, VoxPopuli, FLEURS | ~2,100 h |
| Dutch (nl) | MLS, FLEURS | ~1,600 h |
| Italian (it) | MLS, CV15, VoxPopuli, FLEURS | ~900 h |
| Portuguese (pt) | MLS, CV15, FLEURS | ~210 h |
| Polish (pl) | MLS, FLEURS | ~104 h |
| Japanese (ja) | CV15, FLEURS | ~19 h |
| Tier-2 languages | CV15, VoxPopuli, FLEURS | ~1,400 h |

## Notes

- Several dataset entries are **duplicated** in the YAML config (e.g., `open_mls_it_asr`, `open_mls_pt_asr`, `open_mls_pl_asr`, `asr_chunk_cv15_es`, `asr_chunk_cv15_ja`, `asr_chunk_cv15_pt`, `asr_chunk_cv15_zh`, `asr_chunk_voxpopuli_es`, `asr_chunk_voxpopuli_it`). This is intentional **upsampling** to increase the weight of lower-resource languages during training.
- OpenASR subsets (AMI, Earnings) are also upsampled (×2) to boost domain-specific meeting and financial speech representation.
- The `tier2` configs aggregate multiple smaller languages into a single data spec for efficiency.

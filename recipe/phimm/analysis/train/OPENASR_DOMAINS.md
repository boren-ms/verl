# OpenASR Datasets — Domain Information

The [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (formerly ESB benchmark) evaluates ASR systems across **8 diverse English speech datasets** spanning different domains, speaking styles, and acoustic conditions. This document describes the domain characteristics of each dataset.

## Summary Table

| Dataset | Domain | Speaking Style | Acoustic Setting | Total Hours (test) | Utterances (test) | License |
|---|---|---|---|---|---|---|
| **AMI** | Business meetings | Spontaneous, conversational | Indoor, close-talk + far-field mics | 8.7 h | 12,643 | CC-BY-4.0 |
| **Common Voice** | General / crowdsourced | Read speech (scripted sentences) | Consumer microphones, varied environments | 29.9 h | 16,334 | CC0 |
| **Earnings-22** | Corporate finance (earnings calls) | Spontaneous + narrated | Telephony / conferencing | 4.0 h | 2,741 | CC-BY-SA-4.0 |
| **GigaSpeech** | Multi-domain (audiobooks, podcasts, YouTube) | Read + spontaneous | Clean to noisy, indoor/outdoor | 35.4 h | 19,931 | Apache-2.0 |
| **LibriSpeech** | Audiobooks (literature) | Read speech | Clean studio recordings | 10.7 h | 5,559 | CC-BY-4.0 |
| **SPGISpeech** | Corporate finance (earnings calls) | Spontaneous + narrated | Telephony / conferencing | 100.0 h | 39,341 | Non-commercial |
| **TED-LIUM** | TED conference talks | Prepared presentations | Auditorium / stage | 2.6 h | 1,155 | CC-BY-NC-ND-3.0 |
| **VoxPopuli** | European Parliament proceedings | Formal parliamentary speech | Institutional recording | 4.9 h | 1,842 | CC0 |

---

## 1. AMI Meeting Corpus

- **Domain:** Business meetings
- **Paper:** Carletta et al., "The AMI Meeting Corpus: A Pre-announcement" (2005)
- **HuggingFace:** [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami)
- **License:** CC-BY-4.0

**Description:** 100 hours of meeting recordings captured in three instrumented meeting rooms with different acoustic properties. Uses both close-talking headset microphones (IHM) and far-field microphone arrays (SDM). Participants conducted scenario-driven design meetings; most speakers are non-native English speakers.

**Domain characteristics:**
- Spontaneous conversational speech with overlapping speakers, cross-talk, and interruptions
- Technical and business vocabulary (product design discussions)
- Short utterances (avg 2.5s, 7.1 words) due to conversational turn-taking
- Background noise from room acoustics and equipment
- Non-native English accents (predominantly European)

---

## 2. Common Voice

- **Domain:** General-purpose crowdsourced
- **Paper:** Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus" (LREC 2020). [arXiv:1912.06670](https://arxiv.org/abs/1912.06670)
- **HuggingFace:** [mozilla-foundation/common_voice_15_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0)
- **License:** CC0

**Description:** Mozilla's crowdsourcing project where volunteers worldwide record and validate sentences through a web platform. Covers 100+ languages. The English test set in OpenASR has 16,334 utterances (29.9 hours).

**Domain characteristics:**
- Read speech from curated sentences (Wikipedia, Europarl, user submissions)
- Wide variety of topics — no single domain focus
- Extremely diverse speaker demographics: ages, accents, dialects, genders
- Variable recording quality — consumer microphones, laptops, phones
- Varying background noise levels (home environments)
- Moderate utterance length (avg 6.6s, 9.4 words)

---

## 3. Earnings-22

- **Domain:** Corporate finance — earnings calls
- **Paper:** Del Rio et al., "Earnings-22: A Practical Benchmark for Accents in the Wild" (Interspeech 2022). [arXiv:2203.15591](https://arxiv.org/abs/2203.15591)
- **HuggingFace:** [revdotcom/earnings22](https://huggingface.co/datasets/revdotcom/earnings22)
- **License:** CC-BY-SA-4.0

**Description:** 125 files, 119 hours of English-language earnings calls gathered from global companies. Designed as a benchmark for real-world accented speech, featuring speakers from diverse countries of origin.

**Domain characteristics:**
- Financial/business domain vocabulary (revenue, margins, guidance, fiscal quarters)
- Mix of spontaneous (Q&A with analysts) and narrated (prepared remarks) speech
- Strong international accent diversity — speakers from global companies
- Telephony/conferencing audio quality — varying channel conditions
- Moderate utterance length (avg 5.3s, 18.3 words)
- Long-tail duration distribution with some very long segments (up to 45.9s)

---

## 4. GigaSpeech

- **Domain:** Multi-domain (audiobooks, podcasts, YouTube)
- **Paper:** Chen et al., "GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio" (Interspeech 2021). [arXiv:2106.06909](https://arxiv.org/abs/2106.06909)
- **HuggingFace:** [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)
- **License:** Apache-2.0 (research use)

**Description:** 10,000 hours of transcribed English audio from three sources: audiobooks (~2,655h), podcasts (~3,498h), and YouTube (~3,845h). Covers diverse topics including arts, science, sports, news, comedy, education, gaming, and more.

**Domain characteristics:**
- **Audiobook subset:** Read speech, clean recording, literary/narrative content
- **Podcast subset:** Spontaneous conversational speech, interview format, indoor near-field, sometimes with background music
- **YouTube subset:** Highly varied — clean to noisy, indoor and outdoor, near- and far-field, reading and spontaneous, various ages and accents
- 29 content categories (People & Blogs, Business, News, Science & Tech, Sports, Comedy, Education, etc.)
- Moderate utterance length (avg 6.4s, 19.7 words)
- One of the most domain-diverse datasets in the benchmark

---

## 5. LibriSpeech

- **Domain:** Audiobooks (public domain literature)
- **Paper:** Panayotov et al., "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books" (ICASSP 2015)
- **HuggingFace:** [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)
- **License:** CC-BY-4.0

**Description:** ~1,000 hours of 16kHz read English speech derived from LibriVox audiobook recordings of public domain books (Project Gutenberg). Carefully segmented and aligned. Split into "clean" (lower WER speakers) and "other" (higher WER speakers).

**Domain characteristics:**
- Read speech from literary texts — novels, non-fiction, essays
- Formal, literary vocabulary and sentence structures
- Clean recording conditions (volunteer home studios)
- US English accents predominantly (speakers self-selected via WER ranking)
- Moderate utterance length (avg 7.0s, 18.9 words)
- The most widely used ASR benchmark — strong baseline for read speech performance
- No spontaneous speech, disfluencies, or background noise

---

## 6. SPGISpeech

- **Domain:** Corporate finance — S&P Global earnings calls
- **Paper:** O'Neill et al., "SPGISpeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition" (Interspeech 2021). [arXiv:2104.02014](https://arxiv.org/abs/2104.02014)
- **HuggingFace:** [kensho/spgispeech](https://huggingface.co/datasets/kensho/spgispeech)
- **License:** Non-commercial research only (Kensho/S&P Global)

**Description:** 5,000 hours of professionally transcribed company earnings calls (2007–2020) from S&P Global. ~50,000 speakers — one of the largest speaker counts of any corpus. Transcripts include full formatting: capitalization, punctuation, and denormalized non-standard words.

**Domain characteristics:**
- Financial/corporate domain — earnings reports, guidance, revenue figures, market analysis
- Mix of narrated speech (prepared CEO/CFO remarks) and spontaneous (analyst Q&A)
- Diverse L1 and L2 English accents from global corporations
- Telephony/conferencing audio — consistent but not studio quality
- Tightly segmented: all utterances fall within 5–15 seconds (forced alignment)
- ~90% male speakers (reflecting earnings call participant demographics)
- Fully formatted transcriptions (unique among ASR datasets)
- Largest subset in OpenASR test: 39,341 utterances, 100 hours

---

## 7. TED-LIUM 3

- **Domain:** TED conference talks — science, technology, culture, education
- **Paper:** Hernandez et al., "TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation" (SPECOM 2018). [arXiv:1805.04699](https://arxiv.org/abs/1805.04699)
- **OpenSLR:** [SLR51](https://www.openslr.org/51/)
- **License:** CC-BY-NC-ND-3.0

**Description:** 452 hours of English speech from TED conference talk recordings. The third release of the TED-LIUM corpus, doubling the data from TED-LIUM 2 (207 hours). Audio is extracted from TED talk videos with aligned transcriptions.

**Domain characteristics:**
- Prepared/rehearsed presentations on diverse intellectual topics
- Wide topic range: science, technology, culture, psychology, education, global issues
- Professional presentation style — clear, articulate, audience-facing speech
- Auditorium/stage acoustics with audience present
- Diverse speakers — international TED speakers from many countries
- Longer utterances (avg 8.2s, 23.8 words) — structured sentences from prepared talks
- Occasional audience laughter, applause as background noise
- Smallest subset in OpenASR test: 1,155 utterances, 2.6 hours

---

## 8. VoxPopuli

- **Domain:** European Parliament proceedings — politics, legislation, policy
- **Paper:** Wang et al., "VoxPopuli: A Large-Scale Multilingual Speech Corpus" (ACL 2021). [arXiv:2101.00390](https://arxiv.org/abs/2101.00390)
- **HuggingFace:** [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)
- **License:** CC0

**Description:** Large-scale multilingual corpus from 2009–2020 European Parliament event recordings. Contains transcribed speech in 18 languages plus 29 hours of accented English. The OpenASR benchmark uses the English subset.

**Domain characteristics:**
- Formal parliamentary speech — legislative debates, policy statements, committee discussions
- Political and legal vocabulary — regulations, amendments, directives, EU-specific terminology
- Predominantly non-native English speakers (EU officials from various member states)
- 15 different L2 English accents represented
- Institutional recording quality — consistent, professional audio infrastructure
- Longer utterances (avg 9.6s, 24.1 words) — formal, structured sentences
- Gender imbalance — female speakers below 50% across most languages
- Long-tail duration distribution (some segments up to 69.2s)

---

## Domain Coverage Matrix

| Characteristic | AMI | CV | Earn22 | Giga | Libri | SPGI | TED | VoxP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Read speech** | | ✓ | | ✓ | ✓ | | | |
| **Spontaneous speech** | ✓ | | ✓ | ✓ | | ✓ | | |
| **Prepared/formal speech** | | | ✓ | | | ✓ | ✓ | ✓ |
| **Conversational** | ✓ | | | ✓ | | | | |
| **Financial domain** | | | ✓ | | | ✓ | | |
| **Literary domain** | | | | ✓ | ✓ | | | |
| **Political/legal domain** | | | | | | | | ✓ |
| **Science/education** | | | | ✓ | | | ✓ | |
| **General/multi-topic** | | ✓ | | ✓ | | | ✓ | |
| **Non-native accents** | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| **Noisy/far-field** | ✓ | ✓ | | ✓ | | | | |
| **Telephony/conferencing** | | | ✓ | | | ✓ | | |
| **Clean/studio** | | | | ✓ | ✓ | | | ✓ |

---

## Key Takeaways

1. **Financial domain is heavily represented** — both Earnings-22 and SPGISpeech cover corporate earnings calls, comprising 42% of total OpenASR test utterances (42,082 of 99,546).

2. **Read vs. spontaneous speech** — The benchmark balances read speech (LibriSpeech, Common Voice) with spontaneous/conversational speech (AMI, Earnings-22, parts of GigaSpeech).

3. **Accent diversity** — Most datasets feature non-native English speakers. Only LibriSpeech is predominantly native US English.

4. **Acoustic conditions vary widely** — from clean studio recordings (LibriSpeech) to noisy far-field meeting rooms (AMI) to telephony channels (SPGISpeech, Earnings-22).

5. **Vocabulary specialization** — Financial (SPGISpeech, Earnings-22), political/legal (VoxPopuli), literary (LibriSpeech), and scientific/educational (TED-LIUM) vocabularies test domain generalization.

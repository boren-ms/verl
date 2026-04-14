# English Open-Source ASR Training Data

Extracted from `recipe/phimm/fast-llm-s2-data-main-run-oss.yaml`

## Dataset Summary

| Block | Dataset | Source Path | Format | Examples | Hours |
|-------|---------|-------------|--------|----------|-------|
| 1 | MLS-en | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/open_mls_en_asr_2k.json` | asr-lid-lexical | 0.484M | 1,999.67h |
| 1 | GigaSpeech | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/asr_chunk_gigaspeech_2k5.json` | asr-lid-lexical | 2.124M | 2,498.74h |
| 2 | CV15-en | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/asr_chunk_cv15_en.json` | asr-lid | 1.070M | 1,688.26h |
| 2 | VoxPopuli-en | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/asr_chunk_voxpopuli_en.json` | asr-lid | 0.365M | 1,045.21h |
| 2 | AMI (×2) | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/asr_chunk_openasr_ami.json` | asr-lid | 0.22M×2 | — |
| 2 | Earnings (×1) | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/asr_chunk_openasr_earnings.json` | asr-lid | 0.05M×2 | — |
| 3 | LibriSpeech/LibriHeavy | `/datablob1/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/wavllm_chunk_asr.json` | asr-lid-lexical | 1.030M | 3,772.18h |
| **Total** | | | | **~5.61M** | **~11,004h+** |

## Dataset Descriptions

### MLS (Multilingual LibriSpeech) — English
- **Source:** Read audiobooks from LibriVox
- **Total English:** ~44,660 hours (we use a 2k-hour subset)
- **Style:** Read speech, clean audio
- **License:** CC BY 4.0
- **Reference:** Pratap et al., "MLS: A Large-Scale Multilingual Dataset for Speech Research", 2020 ([arXiv:2012.03411](https://arxiv.org/abs/2012.03411))

### GigaSpeech
- **Source:** Audiobooks, podcasts, and YouTube
- **Total:** 10,000 hours labeled English speech (we use a 2.5k-hour subset)
- **Style:** Mixed — read and spontaneous speech across diverse topics (arts, science, sports, etc.)
- **License:** Non-commercial research use only
- **Reference:** Chen et al., "GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio", Interspeech 2021 ([arXiv:2106.06909](https://arxiv.org/abs/2106.06909))

### Common Voice 15.0 — English
- **Source:** Mozilla crowdsourced platform; volunteers read prompted sentences
- **Total English:** ~1,688 hours (version 15.0)
- **Style:** Read speech, diverse accents and recording conditions (contributed by 50k+ volunteers worldwide)
- **License:** CC-0
- **Reference:** Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus", LREC 2020 ([arXiv:1912.06670](https://arxiv.org/abs/1912.06670))

### VoxPopuli — English
- **Source:** European Parliament event recordings (2009–2020)
- **Total English:** ~543 hours transcribed (we use 1,045h chunk version)
- **Style:** Spontaneous parliamentary speech; includes native and non-native speakers
- **License:** CC-0
- **Reference:** Wang et al., "VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation", ACL 2021 ([arXiv:2101.00390](https://arxiv.org/abs/2101.00390))

### AMI Meeting Corpus
- **Source:** 100 hours of meeting recordings; ~2/3 scenario-based design meetings, ~1/3 natural meetings
- **Style:** Spontaneous multi-party conversational speech; close-talk and far-field microphones
- **License:** CC BY 4.0
- **Reference:** Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", 2005 ([link](https://groups.inf.ed.ac.uk/ami/corpus/))

### Earnings-21
- **Source:** 39 hours of real-world earnings call recordings from 9 financial sectors
- **Style:** Spontaneous, entity-dense business speech (company names, financial terms, ticker symbols)
- **License:** CC BY-SA 4.0
- **Reference:** Del Rio et al., "Earnings-21: A Practical Benchmark for ASR in the Wild", Interspeech 2021 ([arXiv:2104.11348](https://arxiv.org/abs/2104.11348))

### LibriSpeech + LibriHeavy
- **LibriSpeech:** ~1,000 hours of read English speech from LibriVox audiobooks, carefully segmented and aligned. A standard ASR benchmark with clean/other splits.
- **LibriHeavy:** ~50,000 hours of read English speech derived from LibriLight/LibriVox with richer supervision including punctuation, casing, and text context.
- **Style:** Read speech, clean audio
- **License:** Public domain / CC BY 4.0
- **References:**
  - Panayotov et al., "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015 ([link](https://www.openslr.org/12))
  - Kang et al., "LibriHeavy: a 50,000 hours ASR corpus with punctuation casing and context", ICASSP 2024 ([arXiv:2309.08105](https://arxiv.org/abs/2309.08105))

---

## Open ASR Leaderboard Test Sets (ESB Benchmark)

8 datasets used by the [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). Overall score = average WER across all 8 test sets.

| # | Dataset | Test Hours | Style | License | Description |
|---|---------|-----------|-------|---------|-------------|
| 1 | LibriSpeech | 11h (clean + other) | Narrated | CC BY 4.0 | ~1,000h of read English audiobooks from LibriVox. Standard ASR benchmark with "clean" and "other" (harder) splits. |
| 2 | Common Voice | 27h | Narrated | CC0-1.0 | Mozilla crowdsourced corpus — volunteers read Wikipedia sentences. Diverse accents, nationalities, and recording conditions (v9.0). |
| 3 | VoxPopuli | 5h | Oratory | CC0 | European Parliament recordings (2009–2020). Largely non-native English speakers in parliamentary settings. |
| 4 | TED-LIUM | 3h | Oratory | CC BY-NC-ND 3.0 | ~450h of English TED Talk recordings covering cultural, political, and academic topics. |
| 5 | GigaSpeech | 40h | Narrated + spontaneous | Apache 2.0 | 10,000h multi-domain corpus from audiobooks, podcasts, and YouTube. Diverse topics and speaking styles. |
| 6 | SPGISpeech | 100h | Oratory + spontaneous | Kensho User Agreement | 5,000h of professionally-transcribed S&P Global earnings calls. ~50k speakers, L1+L2 accents. Fully formatted transcripts. |
| 7 | Earnings-22 | 5h | Oratory + spontaneous | CC BY-SA 4.0 | 119h corpus of English earnings calls from global companies. Focuses on accented speech from many countries. |
| 8 | AMI | 9h | Spontaneous | CC BY 4.0 | 100h of multi-party meeting recordings with close-talk and far-field microphones. Scenario-based + natural meetings. |

### References
- LibriSpeech: Panayotov et al., ICASSP 2015 ([openslr.org/12](https://www.openslr.org/12))
- Common Voice: Ardila et al., LREC 2020 ([arXiv:1912.06670](https://arxiv.org/abs/1912.06670))
- VoxPopuli: Wang et al., ACL 2021 ([arXiv:2101.00390](https://arxiv.org/abs/2101.00390))
- TED-LIUM: Hernandez et al., 2018 ([openslr.org/51](https://www.openslr.org/51/))
- GigaSpeech: Chen et al., Interspeech 2021 ([arXiv:2106.06909](https://arxiv.org/abs/2106.06909))
- SPGISpeech: O'Neill et al., Interspeech 2021 ([arXiv:2104.02014](https://arxiv.org/abs/2104.02014))
- Earnings-22: Del Rio et al., Interspeech 2022 ([arXiv:2203.15591](https://arxiv.org/abs/2203.15591))
- AMI: Carletta et al., 2005 ([ami/corpus](https://groups.inf.ed.ac.uk/ami/corpus/))
- Open ASR Leaderboard: Srivastav et al., 2025 ([arXiv:2510.06961](https://arxiv.org/abs/2510.06961))

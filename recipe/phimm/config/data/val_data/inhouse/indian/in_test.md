# India-locale audio duration statistics

Audit date: 2026-10-14.

## Scope and method

- All 147,356 manifest-referenced audio files across the 35 newly configured
  datasets in 11 locales were measured; no files failed.
- Existing Hindi (`hi-IN`) and English (`en-IN`) datasets are outside this audit.
- Durations are original-file lengths before SVAD splitting, calculated from
  audio headers as `frames / sample_rate`.
- Duration statistics below are in seconds, rounded to three decimal places.
- The `>40s` count uses unrounded durations and a strict threshold: exactly
  40 seconds does not count.
- Each dataset is counted separately; this report does not deduplicate audio
  content across datasets.
- All counts and statistics were independently recomputed from the per-audio
  audit results.

## Summary

- Audited datasets: **35**
- Audited audio files: **147,356**
- Failed measurements: **0**
- Audio files longer than 40 seconds: **10**
- Only Bengali `Dictation_DisplayGPT` and `TestCorpus_Mobil8` contain files longer
  than 40 seconds: **5 each**. Every other dataset has zero.

## Per-dataset statistics

| Locale | Dataset | Audios | Min (s) | Max (s) | Mean (s) | Median (s) | >40s |
|---|---|---:|---:|---:|---:|---:|---:|
| as-IN | Dictation_DisplayGPT | 8,154 | 1.939 | 18.780 | 8.233 | 7.900 | 0 |
| as-IN | SpeechOcean-as-IN-Test-a82e508a | 8,154 | 1.939 | 18.780 | 8.233 | 7.900 | 0 |
| bn-IN | DeviceCommands_DisplayGPT | 4,785 | 2.072 | 16.114 | 5.532 | 5.207 | 0 |
| bn-IN | Dictation_DisplayGPT | 21,666 | 1.236 | 59.496 | 9.278 | 8.496 | 5 |
| bn-IN | TestCorpus_Mobil8 | 21,666 | 1.236 | 59.496 | 9.278 | 8.496 | 5 |
| bn-IN | TestCorpus_SpeechOcean | 4,785 | 2.072 | 16.114 | 5.532 | 5.207 | 0 |
| gu-IN | BS21_DeviceCommands | 2,047 | 0.720 | 10.860 | 3.049 | 2.880 | 0 |
| gu-IN | BS21_Dictation | 2,010 | 1.754 | 18.030 | 5.900 | 5.540 | 0 |
| gu-IN | DeviceCommands_DisplayGPT | 2,047 | 0.720 | 10.860 | 3.049 | 2.880 | 0 |
| gu-IN | Dictation_DisplayGPT | 2,010 | 1.754 | 18.030 | 5.900 | 5.540 | 0 |
| kn-IN | DeviceCommands | 1,616 | 2.561 | 35.858 | 5.698 | 5.633 | 0 |
| kn-IN | DeviceCommands_DisplayGPT | 1,616 | 2.561 | 35.858 | 5.698 | 5.633 | 0 |
| kn-IN | Dictation-Mobil8-knin | 4,946 | 1.320 | 29.496 | 8.894 | 8.496 | 0 |
| kn-IN | Dictation_DisplayGPT | 4,946 | 1.320 | 29.496 | 8.894 | 8.496 | 0 |
| ml-IN | Aether_Migration_ml-IN_Karya-VoiceCmd-B1_DTEST | 5,000 | 1.291 | 10.546 | 3.912 | 3.749 | 0 |
| ml-IN | DeviceCommands_DisplayGPT | 5,000 | 1.291 | 10.546 | 3.912 | 3.749 | 0 |
| ml-IN | Dictation_DisplayGPT | 1,767 | 4.319 | 23.638 | 10.237 | 9.892 | 0 |
| ml-IN | Kathbath-test-clean-known | 1,767 | 4.319 | 23.638 | 10.237 | 9.892 | 0 |
| mr-IN | DeviceCommands_DisplayGPT | 1,135 | 0.720 | 8.520 | 2.710 | 2.580 | 0 |
| mr-IN | Device_Commands | 1,135 | 0.720 | 8.520 | 2.710 | 2.580 | 0 |
| mr-IN | Dictation_DisplayGPT | 4,099 | 0.840 | 29.496 | 6.264 | 4.424 | 0 |
| mr-IN | OfficeDictation | 4,099 | 0.840 | 29.496 | 6.264 | 4.424 | 0 |
| or-IN | Aether_Migration_or-in-Kathbath-test-clean-known | 2,954 | 2.647 | 13.700 | 6.113 | 5.898 | 0 |
| or-IN | Dictation_DisplayGPT | 2,954 | 2.647 | 13.700 | 6.113 | 5.898 | 0 |
| pa-IN | Dictation_DisplayGPT | 3,207 | 2.415 | 13.026 | 5.623 | 5.410 | 0 |
| pa-IN | Kathbath-test-clean-known | 3,207 | 2.415 | 13.026 | 5.623 | 5.410 | 0 |
| ta-IN | DeviceCommands | 2,047 | 0.240 | 9.420 | 2.956 | 2.760 | 0 |
| ta-IN | DeviceCommands_DisplayGPT | 2,047 | 0.240 | 9.420 | 2.956 | 2.760 | 0 |
| ta-IN | Dictation_DisplayGPT | 2,145 | 1.400 | 19.484 | 6.185 | 5.840 | 0 |
| ta-IN | OfficialDictationSet | 2,145 | 1.400 | 19.484 | 6.185 | 5.840 | 0 |
| te-IN | DeviceCommandsDisplayGPT_Clean | 2,048 | 0.840 | 15.720 | 2.957 | 2.700 | 0 |
| te-IN | DictationDisplayGPT_Clean | 1,906 | 1.410 | 19.420 | 6.013 | 5.660 | 0 |
| te-IN | TestCorpus_OfficialDictationSet | 1,906 | 1.410 | 19.420 | 6.013 | 5.660 | 0 |
| ur-IN | Dictation_DisplayGPT | 3,170 | 2.670 | 13.793 | 5.702 | 5.457 | 0 |
| ur-IN | Kathbath-test-clean-known | 3,170 | 2.670 | 13.793 | 5.702 | 5.457 | 0 |

# Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input-path` | — | Explicit path to JSONL (local or `az://`). Mutually exclusive with `--model`. |
| `--model` | — | Model directory under `--results-root`. Auto-discovers the most recently modified `result_details_*.jsonl` file. |
| `--dataset` | `""` | Dataset label (used in discovery and output). |
| `--results-root` | `az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1` | Root for auto-discovery. |
| `--ref-column` | `ref` | Column name for reference text. |
| `--hyp-column` | `hyp` | Column name for hypothesis text. |
| `--id-column` | auto | Column for utterance ID. Auto-detected from `audio_file_stem`, `id`, etc. |
| `--top-n` | `50` | How many worst utterances to show in alignment samples. |
| `--top-confusions` | `100` | How many substitution/deletion/insertion entries to output. |
| `--length-bucket-size` | `5` | Ref-word-count bucket width for `error_patterns.csv`. |
| `--case-sensitive` | off | Do not lowercase ref/hyp before alignment. |
| `--write-html` | off | Generate a standalone HTML report and download playable local audio for the ranked worst utterances when an audio path is available. |
| `--output-dir` | `tmp/asr-word-error-analysis` | Output directory. |

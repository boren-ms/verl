# Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input-path` | — | Explicit path to JSONL (local or `az://`). Mutually exclusive with `--model`. |
| `--model` | — | Model directory under `--results-root`. Auto-discovers the most recently modified `result_details_*.jsonl` file. |
| `--dataset` | `""` | Dataset label (used in discovery and output). |
| `--results-root` | `az://orngwus2cresco/data/boren/data/results/gpt-4o-mini-asr-v1` | Root for auto-discovery. |
| `--transcription-column` | `Transcription` | Column containing reference text with `<NE>` entity tags. |
| `--hyp-column` | `hyp` | Column name for hypothesis text. |
| `--id-column` | auto | Column for utterance ID. Auto-detected from `id`, `audio_file_stem`, etc. |
| `--top-n` | `50` | How many worst utterances shown in the HTML report. |
| `--top-entities` | `100` | How many top entity error entries to output. |
| `--case-sensitive` | off | Do not lowercase ref/hyp before alignment. |
| `--write-html` | off | Generate a standalone HTML report with entity-highlighted long-text view. |
| `--output-dir` | `tmp/asr-entity-error-analysis` | Output directory. |

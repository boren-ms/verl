#!/bin/bash
# Trim silence from LibriSpeech training data and save as parquet

python -m recipe.phimm.trim_dataset \
  --tsv_path az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe.tsv \
  --output_dir az://orngwus2cresco/data/boren/data/LibriSpeechTrim03 \
  --jobs 64 \
  --n_examples 3

# ZIP-backed ASR datasets

The design follows [Bitbank](bitbank.txt): a standard, uncompressed ZIP contains
all physical audio files for one dataset. Each JSONL audio path stores
`archive.zip!data_offset:byte_length`, pointing directly to a member's bytes.
The loader needs neither the ZIP central directory nor a Bitbank dependency.
ZIP64 supports large archives; WAV/FLAC/other audio bytes are not re-encoded.

## Package one dataset

Run on a CPU-capable remote node with Blob credentials and enough scratch disk:

```bash
cd /root/code/verl
python -m scripts.pack_audio_zip \
  az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/data.jsonl \
  --zip-path az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/audio.zip \
  --output-jsonl az://orngwus2cresco/data/boren/data/openasr_jsonl/ls-clean/data_zip.jsonl \
  --workers 16 --work-dir /tmp
```

Repeat with different input/output paths for each dataset. Relative audio paths
are resolved against the input manifest's directory, or `--audio-root` when
provided. `--audio-field` supports another field such as `url`. Downloads are
bounded to one batch of `--workers` files in memory. Duplicate physical files
are packed once. Plain and gzip input JSONL are supported.

The new JSONL preserves row order, text, IDs, and other metadata. It adds
`original_audio_path` (or `original_<audio-field>`) and `audio_zip_member` for
traceability. Time selectors such as `#0.1:0.5` or `#0%:50%` are retained.
Chunk containers (`file:count:index`) and already packed input are rejected:
this script packages standalone audio files, not chunk-container records.
Existing outputs are never overwritten. The complete archive is published
before the new manifest; original audio and JSONL are untouched. All member CRCs
are checked before publishing.

For Parquet with embedded `audio.bytes`, use `--input-format parquet` and pass
a quoted Parquet glob as the input. Original audio bytes are staged locally and
packed without re-encoding. All non-audio columns are preserved; `audio` becomes
`audio_path`, and `original_audio_path` retains the original embedded filename.
This requires scratch space for the extracted bytes and the ZIP simultaneously.
Parquet containing audio paths without embedded bytes is rejected explicitly.

## Load and cache

Use the new JSONL path in a dataset YAML, remove the old audio `path_map`, and
change `cache_name` to avoid reusing the original processed dataset.
`recipe.phimm.utils.audio.load_raw_audio` supports ZIP references alongside
existing standalone files, chunk containers, embedded bytes, and time ranges.

Without a local archive, the loader uses a streaming byte-range read of only
the selected audio member. If the Orange archive exists under the normal
`~/data` cache, it is read locally. Background dataset caching and
`python -m recipe.phimm.cache_eval_audio DATASET_YAML` deduplicate references
to the same ZIP and cache the archive once, preserving byte/time selectors.

Offsets apply to the exact published ZIP. Do not recompress or edit it without
regenerating the JSONL. Standard ZIP tools can inspect/extract named members.

Opt in with `eval_2609_openall_mix_zip.yaml` for the original base model, or
`eval_2609r2_openall_mix_zip.yaml` for the R2 model. Their 26 datasets
use ZIP-backed manifests with fresh cache names through three new configurations:
`mixlang_fy26q2_zh_seg_zip.yaml`, `openasr_verb_langhint_zip.yaml`, and
`openasr_ml_verb_langhint_zip.yaml`. The original evaluation and dataset YAMLs
remain unchanged and continue using the original sources. The ZIP evaluation
uses the same checkpoint, reward settings, and batch sizes as the original,
changing only its dataset groups. The selected AMI/GigaSpeech >=1s subsets, prompts,
language hints, Hindi lattices, and parent/segment grouping fields are unchanged.
Monsoon English is externalized from Parquet to
`openasr_jsonl/monsoon-en-in/data_zip.jsonl`. Each dataset directory contains one
`audio.zip`; manifests retain their original basename with `_zip` appended.

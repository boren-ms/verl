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

### Quartr reviewed native BitBank data

The training configuration
`recipe/phimm/config/data/train_data/quartr_20261016.yaml`
uses `lang_asr_verb` with an English language hint, like the URGENT ZIP config.
Its JSONL is published at
`az://orngwus2cresco/data/speech/users/xiaoych/quartr_earliest2_reviewed_bitbank_20261016/data_zip.jsonl.gz`.
The YAML uses lossless gzip compression (8.16 MB instead of 63.00 MB) for faster
remote loading; the uncompressed `data_zip.jsonl` is also available alongside it.
It contains all 19,728 records from the 32 Parquet shards (213 events,
182.39 decoded hours). Every Parquet field is retained, including `text` as the
training reference, the comparison transcript/WER, keywords, company/speaker
metadata, timestamps, and the original `audio_ref`. Added fields are `id`,
`language`, `audio_path`, `audio_zip_member`, and `source_parquet`.

The audio paths reuse the existing 32 ZIPs under the same dataset's `bitbank/`
directory: `audio_ref.blob_path` selects the ZIP, and `start_byte`/`size` form
the `!offset:length` selector. No audio is extracted, repacked, or uploaded.
Original `start`/`end` values describe positions in the parent event; they are
metadata only, not an additional crop of the already segmented FLAC.

Use `recipe/phimm/config/data/train_data/quartr_20261016_wer05.yaml` to retain
only records with `wer_maiears <= 0.05` (inclusive 5% WER). This variant maps
`wer_maiears` to the existing filter's `wer` field before filtering and keeps
9,893 records (approximately 82.02 hours), including the 81 records at exactly
5%. It reuses the same JSONL and ZIPs, prompts, and `quartr_20261016` data source,
with a separate processed-dataset cache name. The unfiltered YAML is unchanged.

The earnings-only ReMax recipe is
`recipe/phimm/config/v2611_dev/earning/remax_earning_base0a_r0.yaml`, with its
local base dependency in the same folder. It uses `quartr_20261016_wer05` for
training and only `earnings22_verb_langhint` (Earnings22 AA chunks) for validation.
It preserves the dev `remax_dev_base0a_r0` model, reward weights, and training
settings: two epochs, batch size 128, eight rollouts, LoRA rank 320, constant
learning rate `3e-6`, and two eight-GPU nodes. Each epoch uses the complete
filtered Quartr dataset once, with shuffling rather than mixed-source quotas.

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

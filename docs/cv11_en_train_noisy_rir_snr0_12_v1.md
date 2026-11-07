# Common Voice 11.0 English train: speed, RIR, noise at 0–12 dB

This dataset uses the **full English training split**, not the test split and
not the 0.48-hour Common Voice subset inside URGENT24. It preserves all 948,736
training IDs and original transcripts, with one augmented variant per source
record. No digit, model-WER, text-length, or speaker filtering is applied.

Generation config:
`recipe/phimm/config/data/cache/gen_cv11_en_train_speed_rir_noise_snr0_12.yaml`.
Training data config:
`recipe/phimm/config/data/train_data/noisy/cv11_en_train_rir_snr0_12_verb_langhint.yaml`.

Destination:
`az://orngwus2cresco/data/boren/data/verl/gen_cached/cv11_en_train_speed_rir_noise_snr0_12_but_v1/`.
The presence of audio shards does **not** mean the dataset is ready. `data.jsonl`
is published only after every source record succeeds and all audio uploads finish.

## Completed remote build: 2026-11-06

The full build and publication finished successfully on `verl-new-n2-i4-0`.
All **948,736** source rows produced one WAV each across **24** uploaded shards.
There are **0 missing or filtered rows**. Output audio totals **1,042.252720 hours**
from **1,503.282829 source hours**, occupying **120,109,257,748 bytes** (120.109 GB).
All 2,325 RIRs and 100 noises were used at least once.

Observed speed min/mean/max: **1.000000058 / 1.499648132 / 1.999999444**.
Requested SNR min/mean/max: **0.000000023 / 6.004009763 / 11.999999675 dB**.
The maximum SNR error was **0.000000657 dB before PCM16** and
**0.003925532 dB after PCM16**.

Published files beneath the destination above:

- `audio/en_train_0/` through `audio/en_train_23/`: 16 kHz mono PCM16 WAVs.
- `data.jsonl`: the complete manifest, with Blob audio paths and augmentation metadata.
- `training.yaml`: the training-loader config linked above.
- `generation.yaml`: the reproducible build configuration.
- `summary.json`, `provenance.json`, `source_train.tsv`, and `resources/`.
- `validation.json`: all remote WAV paths and sizes verified; 192 remote WAVs
  read back for SHA256/header/signal checks; complete remote JSONL SHA256 verified.
- `validation_loader.json`: actual `create_datasets` training-loader checks on
  192 samples across all 24 shards, preserving audio paths, transcripts, IDs,
  English language, empty prefix, prompts, and formatted reward references.
- `README.md`: a compact dataset handoff.

The loader check used repository revision
`f32ae5aa74e825026819c9dc4c7c16da50fbddfa`. The independent dataset repository
was synchronized at `74697a038b2d52680c8a5060c0fa0e0d49b843c6` for generation
and `0969cfb99259011e855b357902cebca86305ea78` to add the loader audit.
The remote targeted test suite passed **40 tests** before generation.

Full JSONL SHA256:
`9243d564a8b1f97c52a22858fa95d6709a32b5cbb18c0c7d679ab2bb0a16767f`.
Build fingerprint:
`7797dbbceebdf3fe1b74a188fd2cfdbddd2cf3c85af4f04924e7a9437bc1ce1b`.

Small published evidence was downloaded into
`/home/boren/data/cv11_en_train_rir_snr0_12_build_20261106/remote_evidence/`.
The training-loader audit samples the full shard set; it does not materialize
and transform all 948,736 rows through `create_datasets`.

## Source identity

- Original release: `cv-corpus-11.0-2022-09-21`, English, **train**.
- Original Hub ID: `mozilla-foundation/common_voice_11_0` (currently unavailable
  through the Hub API in this environment).
- Available mirror: <https://huggingface.co/datasets/akahana/common-voice-11-eng-sample>.
- Pinned mirror revision: `4354744379973dd44a1b2273d7beb893810912f5`.
- Despite the mirror's name, its `train.tsv` has all **948,736** unique training
  paths, matching the CV11 English release statistics, with 24 `en_train_*.tar`
  files. Every archive's SHA256 is checked against its pinned Hub LFS metadata;
  every audio member is joined to the original TSV. The final union must match
  all TSV paths exactly.
- Release-statistics cross-check:
  <https://huggingface.co/datasets/echodict/common_voice_11_0/blob/3e0760c0935fec38e1c32ff776f61f47296dfb5e/release_stats.py>.
- Speech license: CC0-1.0. No demographic or client identifiers are copied into
  the training JSONL; the original source TSV is retained separately for provenance.

The older OpenASR `common_voice` export describes its source as CV9, so it is
not substituted for CV11 here.

## Acoustic simulation

1. Decode original MP3, average channels if needed, and resample to 16 kHz with
   polyphase resampling.
2. Draw speed uniformly from **[1, 2]**, then change waveform length using the
   same `speed_perturb` implementation as the existing LibriSpeech cache.
   This changes both speed and pitch.
3. Convolve with a uniformly selected **BUT Speech@FIT ReverbDB** real RIR,
   randomly selecting its channel. Normalize RIR energy, align its maximum
   impulse, retain the speed-perturbed length, and restore speech RMS.
   The tail outside that length is trimmed. No extra RT60 selection is applied.
4. Draw a BUT ReverbDB environmental-noise recording and add a random segment
   at SNR uniformly distributed over **[0, 12] dB**. SNR is the whole-utterance
   power ratio of **reverberant speech to additive noise**, not dry speech.
   Noise itself is not reverberated again.
5. Apply a common gain to keep the mixture below a 0.99 peak and, for very quiet
   clips, raise RMS toward 0.01 (subject to that peak limit). This protects SNR
   against PCM16 quantization; normal-level clips retain their level. Encode
   **16 kHz mono PCM16 WAV**. The exact gain is recorded as `peak_gain`.

Every example receives all three augmentations. Resource pools reuse the verified
2,325 RIRs and 100 noises from `ls_raw_rp_edge_nodigits_aug_audio_rir_but_v1`.
Both carry CC BY 4.0; their manifests, checksums, room labels, original archive
members, notices, and attribution are retained. The source resource audio remains
at the existing Blob URIs recorded in those manifests.

Randomness is deterministic per original TSV index. Main seed: **20261106**;
RIR seed: **20261027**. The manifest records speed, requested/measured SNR,
resource SHA256, noise-crop seed/offset, RIR alignment/channel/gain, original
MP3 SHA256, exact source archive/revision, output WAV SHA256/size, and duration.

## Build, resume, and checks

The requested full build runs on **`verl-new-n2-i4-0`**, cluster
`prod-westus2-cw-6`, in the independent repository
`/root/code/cv11-augmentation-20261106`. It uses 32 CPU workers with lowered CPU
and I/O priority. The existing `/root/code/verl` training checkout is not used
for data generation. No GPU is used by the builder.

The local build was stopped when remote execution was requested. Its incomplete
outputs are retained under `/home/boren/data/cv11_en_train_rir_snr0_12_build_20261106`;
they are not the completed dataset. The remote build regenerates all source rows.

Remote work directory:
`/root/data/cv11_en_train_rir_snr0_12_build_20261106`.
`build.pid`, `build.log`, `validation.pid`, and `validation.log` track only this
dataset build. Code is synchronized using `bpush` from the independent local
repository `/home/boren/code/cv11-augmentation-20261106`.

Because public Hugging Face HTTPS is unavailable on the remote node, Azure
fetches the pinned public Hub CDN archive URLs server-side into the
Green-accessible `orngtransfer` bridge. Azcopy then copies those Azure blobs
server-side into Orange. The local helper handles only HTTP headers and Azure
control requests, and does not log or store the short-lived CDN URLs. The remote
builder verifies every complete archive against pinned Hub LFS SHA256 before
using it. Both a small file and a complete 1.85 GB archive were checked through
this route. The source TSV and verified BUT resource bundle are staged separately.
The source cache is:
`az://orngwus2cresco/data/boren/data/verl/source_cache/cv11_en_train_4354744379973dd44a1b2273d7beb893810912f5/`.

```bash
python -u scripts/stage_cv11_sources.py \
  --source-dir /home/boren/data/cv11_en_train_rir_snr0_12_build_20261106/source \
  --blob-dir az://orngwus2cresco/data/boren/data/verl/source_cache/cv11_en_train_4354744379973dd44a1b2273d7beb893810912f5 \
  --bridge-url https://orngtransfer.blob.core.windows.net/data/speech/boren/data/verl/source_cache/cv11_en_train_4354744379973dd44a1b2273d7beb893810912f5
```

The faster server-side archive-ingestion route actually used for the full build:

```bash
python -u scripts/ingest_cv11_sources_from_url.py \
  --metadata /home/boren/data/cv11_en_train_rir_snr0_12_build_20261106/source/hub_files.json \
  --bridge-dir az://orngtransfer/data/speech/boren/data/verl/source_cache/cv11_en_train_4354744379973dd44a1b2273d7beb893810912f5/url_ingest \
  --output-dir az://orngwus2cresco/data/boren/data/verl/source_cache/cv11_en_train_4354744379973dd44a1b2273d7beb893810912f5 \
  --log-dir /home/boren/data/cv11_en_train_rir_snr0_12_build_20261106/server_ingest
```

On the remote node:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
ionice -c 2 -n 7 nice -n 15 \
python -u scripts/build_cv11_augmented_dataset.py \
  --work-dir /root/data/cv11_en_train_rir_snr0_12_build_20261106 \
  --resource-dir /root/data/cv11_en_train_rir_snr0_12_build_20261106/resources \
  --source-blob-dir az://orngwus2cresco/data/boren/data/verl/source_cache/cv11_en_train_4354744379973dd44a1b2273d7beb893810912f5 \
  --source-files-json /root/data/cv11_en_train_rir_snr0_12_build_20261106/source/hub_files.json \
  --publish
```

For direct local reproduction when public Hub access is available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -u scripts/build_cv11_augmented_dataset.py \
  --work-dir /home/boren/data/cv11_en_train_rir_snr0_12_build_20261106 \
  --resource-dir /home/boren/data/ls_rir_build_20261027/but_reverbdb \
  --publish
```

The builder hashes its config, source TSV, archive metadata, resource manifests,
and DSP/builder source. Changing inputs requires a new work directory. Completed
shards and successful upload checkpoints are reused. Existing final remote
manifests are never overwritten. `--max-shards 1` is a smoke run that does not
publish a final manifest.

For every WAV, the builder verifies decoding, finite positive signal power,
sample rate, expected sample count, SNR error before PCM16 below 0.001 dB,
and SNR error after PCM16 below 0.02 dB. It fails on unexpected/duplicate/missing
source IDs, changed transcripts, resource checksum/license mismatches, or invalid
audio; it does not silently drop bad rows.

`summary.json` records actual source/output hours, resource usage, observed
speed/SNR ranges, maximum SNR errors, row count, and manifest/build fingerprints.
`provenance.json` records exact archive hashes. Audio upload checkpoints are local
build state, not substitutes for independent remote verification.

`scripts/validate_cv11_augmented_dataset.py` independently lists every uploaded
shard and checks **all WAV paths and byte sizes**, reads **8 random WAVs per
shard (192 in total)** back from Blob for SHA256/header/signal checks, and verifies
the complete remote final manifest's SHA256. It publishes `validation.json` only
when all 24 shards and the final manifest pass.

```bash
python -m pytest tests/scripts/test_build_cv11_augmented_dataset.py \
  tests/scripts/test_augment_audio.py -q
```

The training YAML rebuilds `lang_asr_verb` references with English hints and no
prefix, as in the LibriSpeech YAML. Its `data_source` is `noisy_cv11_en_train`;
an experiment must map that name to its intended ASR reward function. This task
creates data and its loader YAML; it does not launch training.

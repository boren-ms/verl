# Repository data formats and previous examples

Read this when preparing MLS/FLEURS chunk datasets in `/home/boren/code/verl`.
These paths are implementation examples, not current job status or fixed skill
parameters. Verify sources and effective configs before use.

## Training source discovery

- `recipe/phimm/config/data/train_data/oss/no_en_v0.yaml`: includes language-specific
  MLS DataSpecs such as `open_mls_es_asr.json`.
- `recipe/phimm/config/data/train_data/oss/no_en_v1.yaml`: includes
  `asr_chunk_fleurs.json` and `asr_chunk_fleurs_tier2.json`, alongside other data.
  Despite the config name, the first FLEURS spec includes English. Choose locales
  from the actual source metadata and requested evidence.
- A DataSpec's `data_sources` points to manifests and chunk roots. Hash the spec,
  manifest and transcription-bearing chunks; retain their URLs and indices.
  Multiple sources can share a language: key caches by source/path hash as well
  as locale to avoid reusing one manifest for another source.

## Chunk formats

Use `recipe/phimm/data/chunk.py` as the authoritative parser:

- `load_examples`, `load_specs`, `load_chunk_info` resolve source paths and fields.
- MLS commonly uses `.transcription` UTF-8 records and `.audio` containers.
- The inspected FLEURS source uses `.sft` JSON records and `.audios` containers;
  each inspected SFT record is a list of one object, with reference at
  `sft.0.messages.1.content`. Validate the assistant role and content structure,
  not just the field index. Do not assume MLS and FLEURS schemas are identical.
- The reference `audio_chunk` is `<container>:<record_count>:<record_index>`.
  `load_chunk_sample` returns `(waveform, sample_rate)` for `.audio`, and a list
  of those pairs for `.audios`. Verify audio count against the intended task.
- Binary containers start with the type, version and indexed payload records.
  Use the native parser where practical. Any custom metadata scanner must check
  headers, record counts, sequential indices, payload lengths and trailing bytes.
- Source `/datablob1?/...` paths are resolved by `resolve_path`; do not invent a
  new storage mapping. Native caches may localize entire multi-example audio
  containers, so estimate total unique-container bytes and disk capacity before
  launch. Range reads for sample verification do not prove a cache is populated.

Validate with `recipe.phimm.data.dataset.jsonl_dataset` and `add_task_info` using
actual language names from the repository. Preserve model/task language-hint
semantics from the chosen donor.

## Example implementations to inspect, not blindly execute

- `scripts/build_mls_bad_val.py`: collect runnable original-source bad cases and
  separate regression analysis from archived paired scores.
- `scripts/build_mls_pattern_subset.py`: MLS train metadata scanning, per-row
  selection provenance, normalization exclusions and deterministic ranking.
- `scripts/build_fleurs_pattern_subset.py`: FLEURS train SFT scanning and six-locale
  diagnostic extraction. It contains fixed base2@560 paths, donor names, source
  lists and publication destinations; adapt into a new version for another run.
  `--publish` writes blob data and the script writes recipe files even without
  that flag. Its literal donor replacements are historical convenience, not a
  generic config-construction API. Compose and audit generated output explicitly.

Historical ready recipes:

- `recipe/phimm/config/v2611_dev/base2_mls_v1.yaml`
- `recipe/phimm/config/v2611_dev/dev/remax_dev_base2_mls_v1.yaml`
- `recipe/phimm/config/v2611_dev/base2_fleurs_v1.yaml`
- `recipe/phimm/config/v2611_dev/dev/remax_dev_base2_fleurs_v1.yaml`

The FLEURS v1 example used fresh v3.5.4-r2/54000 initialization and inherited the
base2 ReMax hyperparameters. Its error evidence was base2@560 vs the previously
accepted v3.5.3-r2 baseline, with the model-version confound explicitly labeled.
Those choices apply to that example, not every future experiment.

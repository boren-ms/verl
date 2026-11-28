# Offline prompt biasing

Prepare candidates once, then reuse the same JSONL for `null` (no biasing), `before_audio`,
and `after_audio`. `add_task_info` only formats the prompt; it does not sample
words. References remain unchanged, and `keywords` are passed to reward and
evaluation through `extra_info`.

## Prepare the RLBR data

Run in the repository's Python environment with the phimm plugin installed, or
set `PYTHONPATH=plugins/phimm/src`:

```bash
PYTHONPATH=plugins/phimm/src python -m recipe.phimm.data.simulate_biasing \
  --rlbr-data-root /home/boren/data \
  --output-dir /home/boren/data/librispeech_biasing/offline_rlbr_100pct_seed42 \
  --keyword-coverage 0.1 1.0 \
  --include-internal-validation --seed 42
```

The source training TSV is `LibriSpeech/asr_train_transcribe.tsv` (all three
LibriSpeech 960h training splits). Training samples from the existing
`librispeech_biasing/words/all_words.count.txt` vocabulary, removing the first
3,000 common words and deduplicating normalized spellings. There is no buffer
or warm-up. Override the vocabulary with `--candidate-word-file`.

Every retained training row has 1–1,000 unique candidates and at least one
actual hit. Each training row independently draws a coverage ratio between
10% and 100% and selects that fraction of its unique usable transcript words,
rounding the positive count up (`--keyword-coverage 0.1 1.0`). Each row also
independently draws its total candidate budget, capped at 1,000 and raised
when necessary to fit its selected positives. This measures transcript keyword
coverage, rather than the fraction of candidates that are hits. Negative
candidates exclude actual transcript hits; `keywords` contains exactly the
selected actual hits. Rows with no usable keywords are saved separately as
`train/rejected_no_keywords.jsonl` for subsequent offline merging.

Add `--negative-only` to generate a separate full negative
training split. Every input row is retained, including transcripts with no
usable hotwords. Each row independently draws 1–1,000 unique candidates from
the same filtered vocabulary, excluding every actual transcript hit;
`keywords` is always empty. Use a separate output directory to retain both
positive and negative datasets for offline merging. Validation lists remain
fixed in both suites.

The generated directory contains:

- `train/data.jsonl`, with `text`, `audio_path`, `biasing_words`, `keywords`,
  and `gt_output`. JSONL contains no `prompt` field; `add_task_info` constructs
  the prompt from the saved candidates when the dataset loads.
- Six LibriSpeech validation conditions: clean/other with 0/100/500 candidates.
  Existing RLBR lists and annotated ground-truth keywords are retained, with
  normalized duplicate candidates removed. Zero-bias conditions keep annotation
  keywords for the same evaluation denominator.
  The published 100/500 labels describe distractor budgets; total candidates
  can exceed those labels because the source lists also include ground-truth
  words. They are preserved rather than truncated.
- With `--include-internal-validation`, fifteen additional conditions:
  Gaming, Insurance, ScienceTech, K12HigherEdu, and Retail, each with 0/100/500
  candidates. Source manifests and vocabulary are saved under `sources/`.
- `datasets_none.yaml`, `datasets_before_audio.yaml`, and
  `datasets_after_audio.yaml`, containing `data.train_data`/`data.val_data` for
  the same manifests, a full-epoch training mix, and source-to-scorer mappings.
  Merge the selected YAML into the trainer's global configuration, or load it
  after the base config through Hydra defaults. Validation uses the existing
  English WER/keyword scorer across all RLBR cohorts.
- `preparation.json`, recording source paths, settings, counts, and SHA-256 hashes.

Audio is not decoded during preparation. Historical LibriSpeech paths are
mapped to `--audio-root`, which defaults to the existing Azure corpus. Internal
validation requires access to `--internal-source-root`. Existing output
manifests are never overwritten; use a fresh output directory for another seed.

The generated configurations raise the base's 1,024-token prompt limit to
8,192 and align actor/reference token budgets with `data.max_model_length`.
Before launching training, measure the expanded prompt lengths with the actual
audio processor and tokenizer, and size batches for the target GPUs. Offline
preparation does not load a model or establish a performance improvement.

## Dataset block

```yaml
dataset_name: jsonl
jsonl_paths: /path/to/train/data.jsonl
add_task_info:
  style: verb
  lang_hint: true
  biasing: before_audio  # null / before_audio / after_audio
post_process:
  add_field:
    fields:
      data_source: asr
  verl_format:
    prompt_key: prompt
    extra_keys: [biasing_words]
```

`before_audio` places `Pay extra attention to the following phrases: *word*.`
immediately before `<audio>`. `after_audio` keeps the MoE ordering: task,
audio, language hint, then bias instruction. Candidates, their order,
references, and keywords are identical across the three configurations.
An empty candidate list produces the normal prompt. Enabling biasing without
the `biasing_words` column raises an error.

## ReMax with the Base3a GSPO template

The `config/v2611_dev/biasing` directory provides `remax_bias_before`
and `remax_bias_after`. They share
`bias_base.yaml`, which directly inherits `config/v2611_dev/base.yaml` and
explicitly sets the 2611r2 checkpoint, ReMax advantages, GSPO loss, and clipping
to match the Base3a GSPO template. Saving and validation run every 100 steps. Data imports and
validation selection live in the two training entry YAMLs.

These configurations use Orange JSONL paths under
`az://orngwus2cresco/data/speech/users/boren/data/librispeech_biasing/`.
Training uses all 268,760 positive rows and samples 26,876 of the 280,393
pure-negative rows, giving 10:1 positive/negative (295,636 rows).
Training validation contains only LibriSpeech test-other at the fixed
100/500 conditions: two datasets and 5,878 rows. Both use the selected
before/after prompt position. Training validates and saves every 100 steps.
The template's fixed 868-step budget is cleared. The shared bias base sets
one epoch: 2,309 steps at batch size 128 with drop_last.

Training sources are defined separately in
`config/data/train_data/biasing/ls_pos.yaml` and `ls_neg.yaml`, with datasource
names `ls_pos` and `ls_neg`. Each ReMax entry defines `data.mix` sample budgets:
268,760 positive and 26,876 negative rows by default (10:1). Adjust the two `sample`
values to change the mix; the expected training-row count follows their sum.
Validation manifests live in `config/data/val_data/biasing`, grouped into separate `ls_before`,
`ls_after`, and `ls_none` YAMLs. Before/after
contain only test-clean/test-other at 100/500; none contains only the 0 datasets.
The single `eval_ls_bias` configuration evaluates the unchanged
2611r2 checkpoint without LoRA. It imports the separate `ls_none.yaml`,
`ls_before.yaml`, and `ls_after.yaml`: LS0 none
and LS100/500 before/after, ten datasets and 27,795 rows. Each source name
includes its prompt mode to keep evaluation results separate.

For arbitrary JSONL, use `--input-manifest file1.jsonl file2.jsonl` instead of
`--rlbr-data-root`, and supply `--candidate-word-file`. Inputs need `text` and
should include `language` and `audio_path`. `--fixed-candidates` converts already prepared
`biasing_words`/`distractors` and `ground_truth`/`keywords` without resampling.

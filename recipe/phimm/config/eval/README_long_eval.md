# Long Eval ASR

Use `long_eval_*` configs to generate hypotheses for long recordings and score them in one pass. The entrypoint is `recipe.phimm.main_long_eval_asr`; it splits long audio into segments, runs rollout generation on each segment, regroups segments by parent recording, concatenates the segment hypotheses, and scores the full recording with DisfluencyTolerant TER plus entity EER.

## Files

- `recipe/phimm/config/eval/long_eval_test.yaml`: small eval config. Set `model.path`, choose the val-data default, and set `data.output_path`.
- `recipe/phimm/config/base/long_eval_asr.yaml`: shared rollout and data knobs, including `max_audio_dur`, prompt and response lengths, batch size, and `measure_kwargs`.
- `recipe/phimm/config/data/val_data/long_eval_test.yaml`: sample long-audio dataset. It uses `svad_explode` to split a long WAV before generation.
- `recipe/phimm/config/data/val_data/long_eval_test_seg.yaml`: pre-segmented variant for JSONL rows that already contain `WavPath#start:end` plus segment metadata.
- `recipe/phimm/main_long_eval_asr.py`: implementation that generates segment hypotheses, groups them by parent recording, writes per-recording details, and writes aggregate measures.

## Configure

Start from `long_eval_test.yaml`:

```yaml
defaults:
  - long_eval_asr
  - data/val_data/long_eval_test
  - _self_

model:
  path: az://orngwus2cresco/path/to/checkpoint/qwen_hf/

data:
  output_path: az://orngwus2cresco/data/boren/data/verl/eval/my_run/long_audio_test
```

`model.path` must point at an HF-format model export directory, for example a checkpoint `qwen_hf/` directory. `data.output_path` should be a unique directory for the run; the evaluator creates per-data-source subdirectories below it.

To evaluate another long-audio JSONL, copy `recipe/phimm/config/data/val_data/long_eval_test.yaml` and update `jsonl_paths`. The raw JSONL should provide fields compatible with the configured `rename_fields` mapping, typically `WavPath`, `DisplayTranscription`, and `CorpusName`.

## Run With Config Values

From the repository root, use the wrapper when `long_eval_test.yaml` already contains the desired `model.path` and `data.output_path`:

```bash
bash quick_run.sh recipe/phimm/config/eval/long_eval_test.yaml
```

`quick_run.sh` detects the `long_eval_*` config name, prepares the Ray environment, and submits:

```bash
python3 -m recipe.phimm.main_long_eval_asr --config-name long_eval_test
```

## Run With Overrides

Use a direct Ray submission when you want to keep the YAML unchanged and set the model or output directory at launch time:

```bash
python3 ray_tool.py prepare_env

ray job submit --working-dir="$PWD" --no-wait -- \
  python3 -m recipe.phimm.main_long_eval_asr \
  --config-name long_eval_test \
  trainer.experiment_name=long_eval_test \
  model.path=az://orngwus2cresco/path/to/checkpoint/qwen_hf/ \
  data.output_path=az://orngwus2cresco/data/boren/data/verl/eval/my_run/long_audio_test
```

For a pre-segmented dataset config, switch the eval config default to `data/val_data/long_eval_test_seg` or launch a config that already uses that default.

## Outputs

For each `data_source`, the evaluator writes:

```text
{data.output_path}/{data_source_slug}/details.jsonl
{data.output_path}/{data_source_slug}/measures.json
```

`details.jsonl` has one row per parent recording. Important fields:

- `ref`: full reference text for the parent recording.
- `hyp`: concatenated hypothesis generated from all segments for that parent recording.
- `response`: list of per-segment hypotheses before concatenation.
- `dter`, `dter_n_err`, `dter_n_ref`: per-recording DisfluencyTolerant TER and counts.
- `eer`, `eer_n_err`, `eer_n_ref`: per-recording entity error rate and counts.
- `dter_detail`: detailed TER alignment output from the scorer.

`measures.json` contains the micro-averaged score for the data source:

```json
{
  "dter": 0.0,
  "dter_n_err": 0,
  "dter_n_ref": 0,
  "eer": 0.0,
  "eer_n_err": 0,
  "eer_n_ref": 0,
  "n_recordings": 0
}
```

Display `dter` and `eer` as percentages by multiplying by 100.

## Inspect Results

For Azure Blob outputs, list the generated source directories and inspect the score:

```bash
bbb ls az://orngwus2cresco/data/boren/data/verl/eval/my_run/long_audio_test
bbb cat az://orngwus2cresco/data/boren/data/verl/eval/my_run/long_audio_test/<data_source_slug>/measures.json
bbb cat az://orngwus2cresco/data/boren/data/verl/eval/my_run/long_audio_test/<data_source_slug>/details.jsonl | head -n 1
```

For local outputs, use `ls`, `cat`, or `jq` on the same directory structure.

## Common Knobs

- `data.max_audio_dur`: maximum segment duration accepted by the model pipeline. The default is `50`; `svad_explode.max_len_sec` is usually `40` to leave headroom.
- `data.batch_size`: number of exploded segments per generation batch.
- `rollout.response_length`: maximum generated token length per segment.
- `rollout.no_repeat_ngram_size`: repetition control for long ASR outputs.
- `data.measure_kwargs`: keyword arguments passed to `recipe.phimm.reward.asr_inhouse_measure.eval_score`.
- `data.log_first_n_samples`: number of parent recordings to log per data source. Each preview includes the aggregate source measures and ordered per-segment prompt, reference, parsed response, and raw model response. Defaults to `3`; set to `0` to disable it.

## Completion Signal

A successful job prints a line like this for each data source:

```text
[<data_source>] DTER: 12.34% [123/997]  EER: 5.67% [7/123]  on 10 recordings
```

It then prints the `details.jsonl` and `measures.json` locations followed by `All Done`.

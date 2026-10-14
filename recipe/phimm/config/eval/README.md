# ASR evaluation

Standard `eval_*` jobs use `recipe.phimm.main_asr_eval`. ASR evaluation
configs enable `trainer.validation_resume=true` by default, including
[eval_2609_openall_mix](../v2609_asr/eval/eval_2609_openall_mix.yaml).

The 2609 ASR [evaluation base](../v2609_asr/eval/eval_base.yaml) sets
`data.val_batch_size` to `256 * trainer.nnodes`: 256 for one node, 512 for two,
and 1024 for four. Dataset-specific batch settings, including full-dataset
long-audio scoring, still take precedence. Training's default validation batch
size remains 256.

[eval_2609r2_openall_mix](../v2609_asr/eval/eval_2609r2_openall_mix.yaml) uses the
`v3.5.4-r2/54000/qwen_hf` baseline checkpoint. Its datasets, scoring, and
evaluation settings match `eval_2609_openall_mix`; only the model path differs.

[eval_2609r2b_in](../v2609_asr/eval/eval_2609r2b_in.yaml) evaluates the same
step-400 checkpoint as `eval_2609r2b_openall_zip` on the 23 India-locale
datasets selected from the benchmark table:

| Locale | Selected corpora |
|---|---|
| en-IN | Conversation FY23Q3, OnlineMeetings FY24Q2, Dictation FY23Q2 |
| hi-IN | Conversation FY23Q3, OnlineMeetings FY24Q2, Dictation FY23Q2 |
| as-IN | Dictation_DisplayGPT |
| bn-IN | DeviceCommands_DisplayGPT, Dictation_DisplayGPT |
| gu-IN | DeviceCommands_DisplayGPT, Dictation_DisplayGPT |
| kn-IN | DeviceCommands_DisplayGPT, Dictation_DisplayGPT |
| ml-IN | DeviceCommands_DisplayGPT, Dictation_DisplayGPT |
| mr-IN | DeviceCommands_DisplayGPT |
| or-IN | Dictation_DisplayGPT |
| pa-IN | Dictation_DisplayGPT |
| ta-IN | DeviceCommands_DisplayGPT, Dictation_DisplayGPT |
| te-IN | DeviceCommandsDisplayGPT_Clean, DictationDisplayGPT_Clean |
| ur-IN | Dictation_DisplayGPT |

Each locale has a separate benchmark-only `<locale>_ter.yaml` config in
[inhouse/indian](../data/val_data/inhouse/indian), using the existing compact
locale tags: [enin_ter](../data/val_data/inhouse/indian/enin_ter.yaml),
[hiin_ter](../data/val_data/inhouse/indian/hiin_ter.yaml), `asin_ter`, `bnin_ter`,
`guin_ter`, `knin_ter`, `mlin_ter`, `mrin_ter`, `orin_ter`, `pain_ter`,
`tain_ter`, `tein_ter`, and `urin_ter`. The general locale configs are unchanged.
These TER configs are imported as 13 locale groups in `data.val_data`.
The evaluation base reuses that list for `data.train_data`; the loader flattens
the 23 entries. Banking, medical, and legacy corpus variants are excluded.
English and Hindi use their existing pre-segmented manifests; other locales use on-the-fly
SVAD splitting with a 40-second limit. Locale-prefixed source IDs keep results
separate, and explicit TER locale registrations prevent the new languages
and en-IN from falling back to en-US scoring.
English, Hindi, and Bengali dictation use complete-dataset validation batches
(`val_batch_size: -1`) so long-audio parent segments are scored together rather
than split across minibatches.
The job still uses one node and experiment name `v2609r2b_in`.

This dataset selection is incompatible with older Hindi-only resume checkpoints.
Use a fresh `trainer.validation_data_dir` or set `trainer.validation_resume=false`
for the first run after this update.

[in1](../v2609_asr/eval/in1.yaml) is the smaller India-locale evaluation:
it selects the first TER dataset from each of the 13 locale configs, retaining
the full job's checkpoint, scorer registrations, and segmented English/Hindi
inputs. Its experiment name is `v2609r2b_in1`, keeping output and resume state
separate from the full 23-dataset evaluation.

Validation sample previews include the reference ground truth alongside the
input, generated output, and score. W&B's `val/generations` table has a
`ground_truth_<n>` column for each selected sample. Selection is controlled by
`trainer.log_val_generations` (zero disables previews), and references stay
aligned with their samples through sorting and deterministic shuffling.

## Prefetch evaluation audio

For JSONL-backed dataset YAMLs, warm the local audio cache before evaluation:

```bash
python -m recipe.phimm.cache_eval_audio \
  recipe/phimm/config/data/val_data/earnings_aa_chunked_langhint.yaml \
  --workers 16
```

Run from the repository root, on **each evaluation node**, as the same user
that runs evaluation. Downloads use `bbb cp` with the existing cache's retries,
timeouts, file locks, and atomic writes. Configure `bbb` authentication first.
Orange files under `az://orngwus2cresco/data/` are mirrored beneath `~/data/`;
the evaluator automatically uses these copies, so keep using the original YAML.
No transcripts, prompts, manifests, or audio bytes are changed.

The script reads JSONL (including `.jsonl.gz`, lists, directories, and globs),
deduplicates manifests and physical audio files across dataset entries, and
preserves chunk/time selector behavior by caching the underlying file.
It scans `audio_path`, `audio_file`, `audio_chunk`, and `url` by default.
Use `--audio-fields audio_path` to select specific source fields; metadata such
as `parent_audio_path` is not downloaded by default. Local audio must already
exist. Other dataset types and remote storage prefixes fail explicitly rather
than claiming to have been cached.

Reruns skip existing files and retry missing files. Failed downloads are logged
and cause a nonzero exit status. A successful run prints counts of unique,
already-local, and downloaded files. As with evaluation resume, keep remote
objects immutable: existing cached files are reused, not checked for updates.
The full referenced manifests are prefetched; dataset filters and transforms
are not applied, so ensure sufficient local disk space.

## Dataset-level resume

Rerun the same config with the same model and `trainer.validation_data_dir`
(normally derived from the project and experiment name). After each complete
dataset, the evaluator saves predictions, scores, sample IDs, and metric details
to `validation_data_dir/_resume/<step>.json`, supporting local and Azure paths.
Completed datasets are skipped without loading their audio or regenerating
predictions. Set `trainer.validation_resume_save_freq` to a positive number to
also checkpoint deterministic intra-dataset progress every N batches; zero keeps
dataset-level-only resume. Full-dataset long-audio scoring batches are preserved.

Final per-source `<step>.jsonl` files and metrics include both restored and newly
evaluated datasets. If all datasets finished, rerunning regenerates these final
artifacts and metrics without decoding again. Model/worker startup still occurs.

Use `trainer.validation_resume=false` to force a fresh run. This invalidates the
old progress checkpoint and disables checkpoint writes for that run. Training
validation is unaffected, regardless of this option. Evaluation resume is
independent of `trainer.resume_mode`, which controls loading model checkpoints.

Resume requires `data.validation_shuffle=false` and compatible model, data,
rollout, scorer, dataset order, and batch settings. Incompatible checkpoints fail
explicitly; use a new output directory or force a fresh run. Keep model and data
contents immutable at their configured paths, and do not run concurrent jobs
against the same output directory. Results created before dataset checkpoints
were supported cannot be resumed.

The separate `long_eval_*` entrypoint is described in
[README_long_eval.md](README_long_eval.md); this option applies to the standard
trainer-based `eval_*` path.

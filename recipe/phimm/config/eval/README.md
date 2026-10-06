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

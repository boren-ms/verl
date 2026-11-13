---
name: eval-mix-openml-aa-report
description: "Evaluate an ASR checkpoint on the complete 2607 MixLang, in-house, OpenASR-ML, and Artificial Analysis suite with eval_mix_openml_aa_ter30_2607v1a.yaml, run the matching reference baseline, retain detailed artifacts, and build a baseline-aware Excel report. Use when: evaluate mixed OpenML AA, run the 2607v1a mixed benchmark, report earnings22_cleaned_aa or voxpopuli_cleaned_aa, compare a checkpoint with the 2607v1a reference, or create a mixed ASR evaluation report."
argument-hint: '<model-label> --checkpoint <candidate-hf-path> [--node <verl-node> ...] [--artifact-root <az-path>] [--out <xlsx>]'
---

# Mixed OpenML + Artificial Analysis Evaluation Report

Evaluate one candidate checkpoint and the matching embedded reference with exactly
`recipe/phimm/config/v2607_new/eval_mix_openml_aa_ter30_2607v1a.yaml`. Each job runs all 34 configured datasets. Preserve raw outputs under durable blob storage and produce one baseline-aware workbook with the Artificial Analysis datasets in their own group.

This workflow is self-contained. Use the commands and report builder below directly; do not invoke or depend on another evaluation/report skill.

## Inputs

| Input | Required | Default |
|---|---:|---|
| `model-label` | Yes | Workbook and experiment label. |
| `--checkpoint` | Yes | Candidate HF model directory, normally `az://.../qwen_hf/`. |
| `--node` | No | Repeated allowlist of Ready `verl-n<N>-*` pools. Otherwise use eligible free pools. |
| `--artifact-root` | No | `az://orngwus2cresco/data/boren/outputs/v2607_new_eval/<model-label>/`. |
| `--out` | No | `tmp/eval_mix_openml_aa_reports/<model-label>.xlsx`. |

Fixed contract:

- Config: `recipe/phimm/config/v2607_new/eval_mix_openml_aa_ter30_2607v1a.yaml`
- Config name: `eval_mix_openml_aa_ter30_2607v1a`
- Entrypoint: `recipe.phimm.main_asr_eval`
- Reference model: `az://orngwus2cresco/data/boren/outputs/ver_2607/remax_2607v1_openml_verb_s100_bs256_lid/global_step_100/qwen_hf/`
- Candidate and reference details: `<run-root>/val_data_gen/<data_source>/0.jsonl`

Never substitute another config, dataset list, reference model, scoring function, or segment setting. A prior reference run may be reused only when its manifest records this exact config, config revision, reference path, and all 34 datasets.

## Dataset And Metric Contract

The report has four groups in this order:

| Group | Datasets | Final metric |
|---|---|---|
| `mixlang` | `mixlang_fy26q2` | `val-core/<dataset>/dter_p_err/mean@1` |
| `openasr_ml` | `de/fr/it/es/pt_fleurs`, `de/fr/it/es_mcv`, `fr/it/es/pt_mls` | `val-core/<dataset>/p_err/mean@1` |
| `AA` | `earnings22_cleaned_aa`, `voxpopuli_cleaned_aa` | `val-core/<dataset>/p_err/mean@1` |
| `inhouse` | all 18 configured `enus`, `dadk`, `huhu`, `nbno`, `nlnl`, and `cscz` corpora | `val-core/<dataset>/dter_p_err/mean@1` |

Artificial Analysis is always the separate `AA` report group. Do not merge its two datasets into `openasr_ml`, even though both use the `openasr_ml` reward function.

Display all metric values as percentages. Compute group averages as arithmetic means over displayed dataset values. Compute overall average as the arithmetic mean of the four displayed group averages. Error reduction is

$$
1 - \frac{\mathrm{candidate\ error}}{\mathrm{reference\ error}}.
$$

Positive values mean the candidate improved.

## Procedure

1. Resolve and validate inputs.
   - Confirm the config and its inherited `recipe/phimm/config/data/val_data/mix_openml_aa_ter30.yaml` exist.
   - Read the reference path from `actor_rollout_ref.model.path`; fail if it differs from the fixed contract above unless the user explicitly requests a config update.
   - Confirm the candidate path is an HF-loadable directory and differs from the reference path.
    - Record the current Git revision. Every distinct checkpoint path must receive a distinct experiment name; never reuse the config name or bare model label as `trainer.experiment_name`.
   - Generate each name with the bundled helper. It produces a readable hyphenated name containing the short suite label, sanitized model label, extracted checkpoint step/name, role, and a six-character hash of the complete checkpoint path:

       ```bash
       EXPERIMENT_NAME=$(python .github/skills/eval-mix-openml-aa-report/scripts/make_experiment_name.py \
          --model-label '<model-label>' \
          --checkpoint '<candidate-or-reference-hf-path>' \
          --role '<candidate-or-reference>')
       ```

      Example: `mix-openml-aa-remax-2607-step100-candidate-ffb575`. Two paths ending in `global_step_100/qwen_hf/` and `global_step_200/qwen_hf/` produce names containing `step100` and `step200`, with different hashes. On retry, pass `--attempt 2` (incrementing as needed) to append `-try2` and avoid reusing the failed run name.

2. Find up to two free Ready pools, respecting a supplied `--node` allowlist. Check both signals on each pool:

   ```bash
   brix pools 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^verl-'
   brix ssh <node> -- 'bash -l -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
   brix ssh <node> -- 'bash -l -c "python /root/code/verl/ray_job.py list 2>/dev/null || true"'
   ```

   A pool is free only when no Ray job is running and every GPU has utilization at most 5% and memory use at most 5000 MiB. Never stop, pause, replace, or colocate with unrelated work. Keep pending rows queued if fewer than two pools are free.

3. For each selected `verl-n<N>-*` pool, set `EVAL_NNODES=N` and verify that count against healthy nodes in `ray status`. A mismatch makes the pool ineligible until resolved. Sync once before submission:

   ```bash
   rcall-brix sync <node>
   ```

4. Build a two-row work matrix: one candidate and one reference. They may run concurrently on separate free pools or sequentially on one pool. Use these durable roots:
   - `<artifact-root>/candidate/`
   - `<artifact-root>/reference/`

5. Submit each row directly so all required Hydra overrides are explicit. Run from `/root/code/verl` on the assigned pool:

   ```bash
   python3 ray_tool.py prepare_env
   ray job submit --working-dir=/root/code/verl --no-wait -- \
     python3 -m recipe.phimm.main_asr_eval \
     --config-dir recipe/phimm/config/v2607_new \
     --config-name eval_mix_openml_aa_ter30_2607v1a \
     'actor_rollout_ref.model.path=<candidate-or-reference-hf-path>' \
     'trainer.default_hdfs_dir=<candidate-or-reference-run-root>' \
   "trainer.experiment_name=${EXPERIMENT_NAME}" \
     'trainer.nnodes=<EVAL_NNODES>'
   ```

   Generate `EXPERIMENT_NAME` separately for each row using that row's exact checkpoint and role. Use the candidate checkpoint for the candidate row and the fixed reference model for the reference row. Record each resolved experiment name, Ray job ID, and assigned pool immediately.

6. Monitor both jobs until `SUCCEEDED`. Refill a newly free eligible pool with a pending row. On failure, inspect that job's logs, repair the root cause, sync changed code, and rerun only the failed row with a new unique experiment name. Never mark a row complete from partial metrics.

7. Capture final metrics and provenance. Save the unmodified Ray log locally at
   `tmp/eval_mix_openml_aa_reports/<model-label>/<role>/ray_job.log` and upload it to `<run-root>/ray_job.log`. Create `<run-root>/artifact_manifest.json` containing role, config path, Git revision, model path, pool, Ray job ID, verified `trainer.nnodes`, W&B entity/project/run ID/name/URL (or explicit unavailability reason), Ray log path, and all 34 detail prefixes.

8. Verify artifacts before reporting:
   - The log contains the expected final metric key for every one of the 34 datasets.
   - Every `<run-root>/val_data_gen/<data_source>/0.jsonl` exists and is readable.
   - The manifest, log, and detail paths are durable `az://` paths, not only node-local paths.
   - Candidate and reference manifests agree on config revision, dataset schema, and topology-sensitive overrides.

9. Build the workbook with the bundled [report builder](./scripts/build_report.py):

   ```bash
   python .github/skills/eval-mix-openml-aa-report/scripts/build_report.py \
     --baseline tmp/eval_mix_openml_aa_reports/<model-label>/reference/ray_job.log \
     --candidate tmp/eval_mix_openml_aa_reports/<model-label>/candidate/ray_job.log \
     --candidate-model '<candidate-hf-path>' \
     --candidate-label '<model-label>' \
     --out tmp/eval_mix_openml_aa_reports/<model-label>.xlsx
   ```

   The builder rejects incomplete sources. It creates `results` and `provenance` sheets, formulas for group/overall averages and error reduction, percentage formatting, and a red-white-green reduction scale centered at zero.

10. Add the candidate/reference W&B URLs, remote Ray-log paths, detail roots, and manifest paths to the `provenance` sheet if they are not already represented by the local source paths. Do not alter the formulas or group boundaries.

## Quality Gates

- Candidate and reference Ray jobs succeeded using the exact fixed config and matching data/scoring settings.
- Both jobs used `trainer.nnodes` equal to the verified pool node count.
- Candidate checkpoints with different paths have different `trainer.experiment_name` values; each name contains the checkpoint step/name and path hash.
- Exactly 34 dataset rows exist: 1 MixLang, 13 OpenASR-ML, 2 Artificial Analysis, and 18 in-house.
- `earnings22_cleaned_aa` and `voxpopuli_cleaned_aa` appear only under `AA`.
- All expected metrics are present and non-negative; no auxiliary metric silently replaces the required final `val-core` metric.
- Each role has a readable remote manifest, unmodified Ray log, and 34 detailed `0.jsonl` files.
- Baseline/candidate/error-reduction cells use percentage formatting, and reduction formulas are `1 - candidate/reference`.
- Group and overall averages remain Excel `AVERAGE(...)` formulas rather than stored values.
- Reopen the workbook with `openpyxl`; confirm sheets `results` and `provenance`, 40 rows in `results`, formulas in the reduction column, and valid styles/conditional formatting.
- Run `python -m pytest tests/skills/test_build_mix_openml_aa_report.py -q` successfully before delivery.
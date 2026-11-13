# Base3a Main integration: report and code walkthrough

This directory packages the completed experiment and implementation explanation
for branch `boren/main_2611_dev`. The report was collected on 2026-11-13 at
19:08 UTC. Packaging the artifacts did not rerun training or inference.

## Start here

- [15-page PDF report](report.pdf)
- [Comparison report with figures and Excel links](report.md)
- [Detailed code changes and model integration workflow](CODE_WORKFLOW.md)
- [Compact evidence and source identities](evidence.compact.json)
- [Step-260 Excel comparison](full_eval_s260/comparison.xlsx)
- [Final step-868 Excel comparison](full_eval_s868/comparison.xlsx)
- [Main agent-loop extension points](../../agentic_rollout_support.md)
- [Skill import provenance](../../../.github/skills-import.json)

## Completion and quality are separate

Training completed 868/868 updates. Final validation, checkpoint/export868,
and all three full evaluations completed. Each new full evaluation covered
35 datasets and 192,883 raw segments. The quality gate is unchanged: at most
+0.1 percentage-point regression per dataset, evaluated at full precision.

| Comparison | Dataset gates passed | Result |
| --- | ---: | --- |
| Final validation at 868 | 4/9 | FAIL |
| Full OpenASR at 260 | 31/35 | FAIL |
| Full OpenASR at 868 | 29/35 | FAIL |

Raw cohorts and scoring code match for the full comparisons. Some normalized
reference denominators depend on hypotheses and differ; the exact counts are
retained in each `full_eval_s*/data/comparison_all35.json` file. Original
training used legacy rewards through step120, then resumed with newer scoring;
the candidate used the requested scorer from step0. This difference confounds
attribution. No cause of the quality regressions has been established.

The shutdown DataLoader finalizer warning remains recorded in the evidence.
The driver exited successfully, and all16 final checkpoint ranks and optimizer
state were independently inspected. A separate live distributed reload of868
was not performed.

## Code and import provenance

- Main baseline: `662507f9582e16d09c12484387f27727b5d0f947`.
- Selected recipe/model source: `dffabc00d8261dd9abf3087958259d5c8944f5a2`
  from `boren/rl_2611_dev`.
- Completed-job runtime snapshot: `489bd78dd537d4282c49fe866340eb536c03710f`.
- Skill source, fetched for this publication:
  `89be15eeffde8aa901fedb9954b80e4c5052a1bc` from the same development branch.

The implementation files under `plugins/`, `verl/`, and the audio-model test
remain identical to the runtime snapshot. Of58 imported dependency files,
56 match the selected source exactly; two package registration files were
adapted. [The import manifest](provenance/import_manifest.json) records them.
The imported skill trees contain19 active and24 archived skills plus their
scripts, tests, references, templates and assets. Those trees retain exact
source bytes. Main's existing three skills and discovery links are preserved.
The copied skills retain their original environment and recipe assumptions;
importing them does not establish that every workflow is portable to Main.

## Verification at publication

A focused CPU invocation passed82 tests:78 imported skill checks and4 recipe
configuration/scoring checks. It deselected five adapter checks that require
the training dependency stack. The grouping check was also attempted separately
and could not import `ProcessorMixin` because this workstation's Torchvision
raises `operator torchvision::nms does not exist`. No product change was made
to work around the workstation environment.

```bash
PYTHONPATH="$PWD/plugins/phimm/src:$PWD/plugins/qwen35_audio/src:$PWD" \
  python -m pytest -q tests/skills \
  .github/skills/openasr-report/tests \
  .github/skills/audio-align-subset/scripts/tests \
  .github/skills/error-analysis-and-fix/scripts/test_select_subset.py \
  plugins/phimm/tests/test_adapter.py \
  -k 'not parent_grouping and not vllm_no_repeat and not remax_advantage and not training_reward_metrics and not source_on_policy'
```

This used the existing `/home/boren/.virtualenvs/openai/bin/python` environment;
Main's small local `.venv` lacks the experiment dependencies. Historical GPU
validation includes9 adapter tests,12 Main TQ/checkpoint/trainer checks,
8 model/audio checks, real audio forward/backward, and the complete868-step run.
Those checks are historical evidence rather than newly rerun GPU tests.

Focused Ruff checks pass for the authored adapter, entrypoints, CUDA runtime,
changed Main interfaces and current tests. Broader source-preserving checks
retain these exceptions: six dependency files have inherited whitespace/CRLF;
`verl/audio_cache.py` has inherited I001/B904 lint findings; and the runtime
versions of `verl/utils/model.py` and the CUDA import test differ from the local
formatter. The recorded runtime files were preserved instead of silently
rewriting imported source. Unrelated legacy top-level archived-skill tests
were excluded; the skill trees' own files are copied in full.

## Artifact integrity and full evidence

The PDF is byte-identical to the verified regenerated original:

```text
SHA256 2e7dedbcb9abc2a1cc860a668aece5cd4a5223e6a309cf362c959bfa19ae0297
Pages  15
```

The tracked package contains the report, portable walkthrough, figures,
workbooks, exact endpoint counts, selected provenance and numeric plotting
observations. The compact evidence omits duplicated utterance payloads and
unrelated per-update metrics. Local raw-log paths in evidence describe their
historical source; the report's Markdown asset links are repository-relative.
The complete90-file original package, including70MB raw evidence, remains in
the verified ZIP backup:

```text
az://orngwus2cresco/data/boren/outputs/v2611/remax_main_base3a_2611r2_main_agentic_20261112/debug/reports/20261113T190802Z/base3a_results_20261113T190802Z.zip
Bytes  20405468
SHA256 8475fd5fbcceb5c8dcd9ee03fbc47c0d35ae42d230d2dae1f8e07516acf824a8
```

[The local package manifest](package_manifest.json) records hashes of the files
tracked here. [The numeric verification record](provenance/report_numeric_verification_20261113T190802Z.json)
records the original report checks. Original immutable report packages and
external research notes are preserved separately.

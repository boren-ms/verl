# Workspace Skills

Choose the narrowest skill that owns the requested outcome. The entries below
are complementary workflows, not alternative implementations of the same job.
Invoke another skill only for the handoff it owns; a child job must not restart
its parent workflow.

## Routing

| Request | Owner | Boundary / handoff |
|---|---|---|
| Submit, inspect, monitor, or repair a particular ASR Ray job | [verl-asr-run](verl-asr-run/SKILL.md) | Owns the job lifecycle and generic HF export, not benchmark definitions. A status check does not start training or install automation. |
| Evaluate steps of one model on the standard 2609 suite | [eval-2609-benchmark-report](eval-2609-benchmark-report/SKILL.md) | Owns the suite, compatible baselines, work matrix, artifacts, and one consolidated workbook per model. |
| Run the fixed 2607 MixLang/OpenML/Artificial Analysis suite | [eval-mix-openml-aa-report](eval-mix-openml-aa-report/SKILL.md) | Separate 34-dataset contract and report builder; not a variant of the 2609 suite. |
| Report existing en-US digits metrics against the historical baseline | [digits-report](digits-report/SKILL.md) | Standalone CER/WER workbook; never supplies the baseline for 2609 digits evaluations. |
| Compare utterance-level ASR results or validation checkpoints | [asr-detail-compare](asr-detail-compare/SKILL.md) | Analysis and HTML, not remote evaluation; supports a single result against its reference. |
| Split a parent reference across ordered audio chunks | [align-long-audio-transcription](align-long-audio-transcription/SKILL.md) | Produces aligned presegment manifests, not a result-comparison report. |
| Generate written/spoken text records | [generate-text-spoken-jsonl](generate-text-spoken-jsonl/SKILL.md) | Text-only generation; hand off to `generate-audio-dataset` for speech. |
| Synthesize speech and publish a training dataset | [generate-audio-dataset](generate-audio-dataset/SKILL.md) | TTS, manifest, and data YAML from supplied text. |
| Transfer N existing ChunkFiles and prepare training metadata | [digits-chunk-training-prepare](digits-chunk-training-prepare/SKILL.md) | Existing chunk data, not synthetic generation or checkpoint transfer. |
| Flatten a mixed dataset's components into source JSONL | [mix-source-jsonl](mix-source-jsonl/SKILL.md) | Manifest extraction, not reference alignment or TTS. |
| Transfer a full checkpoint as a delta against a baseline | [ckpt-delta-transfer](ckpt-delta-transfer/SKILL.md) | Full-state delta reconstruction, not PEFT adapter extraction. |
| Extract actual LoRA tensors and publish the fixed-baseline merge | [lora-weight-transfer](lora-weight-transfer/SKILL.md) | Its fixed baseline and scaling are specific to this workflow, not defaults for generic HF export. |
| Decode a small audio list with HF/vLLM on Brix | [remote-qwen35-audio-decode](remote-qwen35-audio-decode/SKILL.md) | Bad-case transcription, not full benchmark scheduling. |

## Shared Contracts

- [Remote execution](references/remote-execution.md) is the single source for
  occupancy checks, topology, workspace sync, and recurring-monitor ownership.
- [The 2609 benchmark contract](eval-2609-benchmark-report/SKILL.md#benchmark-contract)
  defines the required suite: in-house DTER, en-US digits, OpenASR-ML, and
  MixLang. Tier 1 digits are opt-in. An explicitly requested subset is partial,
  not a complete-suite result.
- [Job state](verl-asr-run/SKILL.md#step-1b-maintain-the-pipeline-cache) defines
  the training cache format. A coordinating caller is the sole writer while
  children return state updates to it.
- Training alone does not authorize post-training evaluation. Monitoring alone
  does not authorize persistent automation.
- Historical embedded baseline numbers are usable only with verified matching
  model/config/scoring provenance; a label alone does not establish compatibility.
- Examples use repository-root-relative commands unless stated otherwise.
  Resolve paths, checkpoint provenance, and live node availability at execution
  time; example nodes and personal Python paths are not universal defaults.

## Maintenance

Keep one `SKILL.md` per skill name, matching its directory name. Descriptions
should state the owned outcome and distinguishing triggers rather than claim
every ASR task. Put shared procedures in references and link to them instead of
copying them into each skill. Keep workflow-specific commands and validation
alongside their owner. Supporting scripts remain the source of truth for CLI
flags and output shapes.

User/plugin skills (for example `remote-development`, `copy-to-corp-blob`,
`asr-word-error-analysis`, and entity-analysis helpers) are external dependencies,
not additional workspace copies. Invoke them only when available and relevant;
do not invent a local path or reproduce their implementation here.

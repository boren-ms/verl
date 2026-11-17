# Qwen audio SFT

This entry point uses verl's native SFT engine for teacher-forced training of
the Qwen3.5-Audio model registered by `hf_qwen35_audio`. It uses the same six
training sources, 55,664-row source budgets, preprocessing, language prompts,
and nine validation cohorts as `remax_main_base3a_2611r2`. Both recipes import
`config/data/base3a.yaml`, so data edits stay shared.

Use the existing Qwen audio training environment with `verl`, the `phimm`
plugin, and `hf_qwen35_audio` installed and Orange Blob access configured.
The model must expose a compatible HF tokenizer and audio processor accepting
`text`, `audios`, and `return_tensors`. The default checkpoint, LoRA modules,
rank 320 / alpha 640, learning rate 6e-7, and two epochs follow Base3a. These
are initial SFT settings, not tuned SFT hyperparameters. Set `model.path` to
train another compatible Qwen audio checkpoint; set `model.lora_rank=0` for
full parameter training.

## How an update differs from ReMax

| Stage | Standard synchronous Base3a ReMax | Audio SFT |
| --- | --- | --- |
| Input | Prompt and audio from the Base3a mix | Same source transforms and source budgets |
| Training response | Sample model responses and a greedy baseline | Read the formatted reference from `reward_model.gt_output` |
| Objective | Reward-derived ReMax advantages, policy loss, and configured KL loss | Teacher-forced next-token cross-entropy on the assistant reference |
| Model execution | Training actor, reference log probabilities, vLLM rollout, and ASR scoring | Native verl SFT training workers and the HF audio model |
| Sequence representation | Generated response tokens and rollout metadata | Prompt/audio tokens followed by the complete assistant target |
| Recognition validation | Decode and aggregate corpus WER/CER | Export and decode with the separate Base3a validation config below |

For each SFT example, `ASRSFTDataset` first renders the prompt with an assistant
generation prefix. It renders a second conversation including the formatted
reference and verifies that the prompt prefix is identical. The Qwen processor
receives `(waveform, sample_rate)` and extracts the audio features once; the
tokenizer encodes the remaining assistant suffix. The dataset returns token
IDs, positions, an assistant-only loss mask, and `multi_modal_inputs`. The
native engine supplies causal next-token alignment and optimization. Tags in
the formatted reference are part of the target, so the model learns the same
response convention used by the recognition scorer.

The Ray entry point populates the base-model/config/tokenizer cache on every
participating GPU node before constructing the native SFT trainer. This is
necessary because its model configuration carries resolved local paths to the
workers. The driver owns the global dataloader and logs metrics on CPU; native
workers distribute the batch over FSDP2 ranks. The torchrun entry point instead
uses the native SPMD trainer and a distributed sampler.

With 55,664 rows and global batch 128, dropping the incomplete training batch
gives 434 optimizer updates per epoch and 868 across two epochs. The existing
Base3a mix supplies its seeded source/group shuffles. Both SFT entry points
honor the shared `data.shuffle=false`, avoiding an extra sampler shuffle.
Matching these row and update budgets controls part of the comparison;
ReMax generates multiple responses and performs different work per update.

## Launch

For one node with eight GPUs, run from the repository root:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m recipe.phimm.sft.main \
  trainer.nnodes=1 \
  trainer.experiment_name=sft_base3a_2611r2_run1
```

For the default two-node setup, run the following on **each** node, providing
the same reachable `SFT_MASTER_ADDR` and a distinct `SFT_NODE_RANK` (0 or 1):

```bash
torchrun --nnodes=2 --nproc_per_node=8 \
  --node_rank="$SFT_NODE_RANK" \
  --master_addr="$SFT_MASTER_ADDR" --master_port=29500 \
  -m recipe.phimm.sft.main \
  trainer.experiment_name=sft_base3a_2611r2_run1
```

Inspect the effective YAML without initializing the model or downloading data:

```bash
python -m recipe.phimm.sft.main --cfg job --resolve
```

On an existing two-node Ray pool, the ASR submitter launches the same SFT engine
with native Ray training workers and a CPU driver:

```bash
python plugins/phimm/submit.py --sft --submission-id sft_base3a_2611r2_run1
```

Use `recipe.phimm.sft.main_ray --cfg job --resolve` for this entry point's dry
configuration check. The Ray trainer retains the validation remainder by
repeating at most `world_size - 1` rows for even data-parallel dispatch.

The config is `config/v2611_dev/sft/sft_base3a_2611r2.yaml`. It uses global
batch 128, FSDP2, and native checkpoint/resume support. Checkpoints are saved
every 10 updates and at completion, with validation every 20 updates and at
completion. Use a unique experiment name for each new run; `resume_mode=auto`
resumes an existing local run directory. Native SFT checkpoints contain model,
optimizer, and extra state; HF export is a separate operation.
Checkpoints are saved locally and published to `trainer.default_hdfs_dir` on
Azure Blob. The ASR SFT handler routes each rank through the existing FSDP Blob
publisher, uploads dataloader state and LoRA metadata, then commits the remote
tracker last. Auto-resume prefers a committed remote checkpoint and restores
both model/optimizer and dataloader state. The audio wrapper receives padded
sequences with separate attention masks (`model.use_remove_padding=false`);
this avoids relying on external packed-sequence boundary forwarding through
the custom model wrapper.

Only formatted assistant reference tokens and their chat-template closing
tokens contribute to loss. Prompt and acoustic placeholder tokens are masked.
The dataset retains extracted audio features for the model forward pass.
Oversized token sequences or training audio beyond 40 seconds fail explicitly
instead of cropping audio against a complete reference. Override
`data.max_audio_dur=null` and increase the token budgets if full longer clips
are intended. Validation loads full audio, including lists of audio paths,
and computes loss on each source row without RL parent grouping. Increase
`data.max_length` and `data.max_token_len_per_gpu` together if validation
examples exceed the default 8192 tokens.

Validation reports teacher-forced `val/loss`; generation-based WER and reward
evaluation are separate. The validation loader retains the final partial
batch. With multiple data-parallel ranks, the distributed sampler may repeat
up to `dp_size - 1` validation rows to keep every rank's forward count equal.
The OpenLong diagnostic cohorts retain Base3a's inferred-reference caveat.

For recognition validation, export a complete SFT checkpoint to a merged HF
model, then run the original nine Base3a validation cohorts:

```bash
python plugins/phimm/submit.py --evaluate \
  --submission-id sft_base3a_step20_validation \
  --config-path config/v2611_dev/sft \
  --config-name eval_sft_base3a_2611r2 \
  actor_rollout_ref.model.path=<merged-HF-model-path> \
  trainer.experiment_name=sft_base3a_step20_validation
```

This config inherits Base3a's generation settings, parent grouping, and corpus
WER/CER scorers. It disables training resume and new LoRA adapters. Native SFT
checkpoints store shards directly under `global_step_<N>`, without RL's `actor`
subdirectory. Use the producer's recorded LoRA rank/alpha when merging.
Compare recognition errors as `sum(n_err) / sum(n_ref)` for each cohort, retaining
the raw counts and checking identical reference coverage. Keep the SFT
teacher-forced loss separate from ReMax recognition metrics. Equal optimizer
steps across SFT and ReMax do not establish equal compute or token budgets.

## Generation validation at every saved interval

The separate checkpoint validator reuses the native ASR generation workers to
decode SFT checkpoints every 20 optimizer steps, plus final step 868. It loads
only model shards from each committed SFT directory, merges the recorded LoRA
weights for rollout, and uses the same nine cohorts and corpus scorers. It does
not restore optimizer/dataloader state or perform training updates. Run it on a
separate two-node pool from the producer:

```bash
python plugins/phimm/submit.py --sft-evaluate \
  --submission-id sft_base3a_every20_validation \
  asr.sft_validation.checkpoint_root=<SFT-training-Blob-root> \
  trainer.experiment_name=sft_base3a_every20_validation
```

Use `asr.sft_validation.steps=[20]` for an initial check against the exported-HF
decode. By default the validator waits for later checkpoints while training
continues. Each validation records the source optimizer step directly, checks
the producer's base model, rank/alpha, world size and all model shards, and
publishes `<step>.complete.json` only after all nine metrics and detail files
have uploaded. Completed steps can be reused only when configuration, validator
code and artifact fingerprints still match. Final HF export and full OpenASR
evaluation remain separate from this nine-cohort validation trajectory.

Keep the decoding method consistent across the trajectory. The native worker
merges the original FP32 adapters before synchronizing BF16 rollout weights.
The converter's historical default rounds adapters to BF16 before merging;
those exports need not produce the same weights or recognition results. For a
final merged export that retains adapter precision through the FP32 merge, add
`--preserve-lora-precision` together with `--match-lora-merged`. The converter's
default remains available for reproducing historical exports. Record this
choice in export provenance and verify the resulting model before evaluation.

Greedy decoding alone does not ensure identical outputs across executions when
vLLM batch invariance is disabled. Preserve attempt identities and raw error
counts; do not attribute every rerun difference to the checkpoint or merge
precision. If a batch validation is resumed, retain the original experiment and
the complete original `steps` list so its committed configuration fingerprints
continue to match.

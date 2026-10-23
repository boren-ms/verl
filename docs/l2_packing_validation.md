# Qwen3.5 packing: MoE to verl mapping and L2 validation

Reference: `/home/boren/code/MoE`, Azure remote `moe`, branch
`ruchaofan/main-audio-1001`, refreshed to `191647e35` on 2026-10-23.
The implementation already exists in verl commit `649892aaaa`; this worktree
retains that implementation and adds training-update and config regression
coverage rather than replacing it with a second patch.

## Mapping

| Contract | MoE | verl |
| --- | --- | --- |
| Gather valid tokens once | `phyagi/models/qwen3_5/modeling_qwen3_5.py`, `_pack_tensor`, before the decoder loop | `verl/workers/actor/dp_actor.py`, `unpad_input`, before model forward |
| Full-attention boundaries | `flash_attn_varlen_func` with `cu_seqlens` | Existing FlashAttention integration derives boundaries from reset text positions |
| Convolution boundaries | `causal_conv1d_fn(seq_idx=seq_idx)` | `verl/models/transformers/qwen3_5.py`, `_packed_linear_forward` |
| GatedDeltaNet boundaries | Delta-rule variable-length state reset | Same function passes `cu_seqlens` to the imported FLA kernel |
| Restore batch layout | Unpack hidden states after decoder layers | Actor restores log probabilities/entropy with `pad_input` |
| Missing fast kernels | Reference has a conditional packed path | One right-padded batched fallback call, then concatenation of valid prefixes; keeps FSDP collective counts independent of sample count |

The actor's gathered text positions must reset to zero at every sample start.
Four-axis position IDs use their explicit text axis. Three-axis multimodal
RoPE positions must not be mistaken for text boundaries. Cache decoding and
Ulysses SP greater than one are not supported for packed linear attention.
Metadata travels through decoder kwargs, including checkpoint recomputation.
Model parameters and checkpoint keys are unchanged.

This mapping covers the decoder. It does not transplant MoE's audio-encoder
packing or alter the audio frontend, training data, rewards, or loss formula.

## L2 control

Use `recipe/phimm/config/v2611_dev/dev/remax_dev_base0_v1h2a_fix_L2.yaml`
as the control and its `_L2_pack.yaml` sibling as the candidate. Hydra composition
at this worktree's HEAD differs only in
`actor_rollout_ref.model.use_linear_attention_packing: false -> true`.
Both already use remove padding. Preserve save frequency 10, validation frequency
20, data mixture, reward weights, optimizer, initialization and rollout settings.
The config regression compares the entire composed configuration after reverting
that one flag, preventing silent experiment drift.

## Verification

Run with Transformers 5.7.0 and the model's supported Torch/kernel environment:

```bash
python -m pytest tests/models/transformers/test_qwen3_5_packing.py \
  tests/recipe/phimm/test_l2_pack_config.py -q
```

Tests cover logits against independent samples, full-attention isolation,
zero cross-sample input gradients, parameter gradients, grouped value heads,
checkpoint recomputation, original padded/cache behavior, and real CUDA kernels
when available. The added multi-step test compares four AdamW updates, losses,
all gradients, parameters and optimizer moments across changing sequence lengths,
with and without checkpointing. It uses independent unpadded samples as the
semantic control, because an unpatched concatenation leaks state.

Numerical equivalence on a small model does not establish ASR non-inferiority.
Training quality must be checked with the real model and the same validation
datasets at matched optimizer steps, keeping every dataset separate. Likewise,
successful CUDA correctness tests do not quantify a throughput gain.

## Existing training and evidence

At initial inspection, an independently monitored L2_pack run was active on `verl-n2-i7`:
`remax_L2_pack_r1_retry1_20261022T235757Z`, W&B run
`genai/v2611/297053a879ca45874c656b5929510335`, pinned source `d29228b1cf`.
Do not duplicate, replace, or sync a new checkout over that active run.
Its packing implementation hash matches the implementation inherited by this
worktree before the documentation-only module edit.

The historical L2 baseline is W&B run
`genai/v2611/06a732c00714070c46b72a90837f6121`, source `c6a69fdb95`.
It was deliberately stopped; comparisons do not authorize resuming it.
Those source revisions also differ in reward/log-probability code, so the
historical validation comparison is descriptive, not a controlled packing-only
experiment. A clean training A/B requires pinning both arms to the same source,
environment, initialization and seed, with only the packing flag different.

The existing `L2_repo` control, W&B run
`genai/v2611/73f8f5f95288a6bcc9fe79fa0775448e`, also uses `d29228b1cf`.
Its saved composed config and the active pack r1 preflight config differ only
in the packing flag, experiment name/output paths and resume policy. Use its
completed matched-step validation as the closer control. Do not use the pack
root `composed.yaml`: that file belongs to the older first attempt and contains
a different log-probability setting. The current attempt's authoritative
preflight is `attempts/restart1_20261022T234924Z/config_preflight.json` under the
existing pipeline evidence directory. Recovery and runtime environment still
need to be accounted for before making a formal non-inferiority claim.

Evidence and the provisional per-dataset comparison for this task are saved in
`tmp/l2_pack_validation/`. Keep runtime reports out of commits. Retain explicit
training steps, source revisions and historical-versus-live collection times.
Do not declare no training effect from finite loss or a single validation point.


## Completed 50-step smoke test (2026-10-23)

`L2_moe_pack_ab50` (W&B `1e6eb266d7fecff73ac1b0483001a88b`) and
fresh L2 (`592570550fc809e0d29eb2f43e6a746f`) each completed 50 optimizer
steps, validation at 0/20/40/50, and five complete checkpoints. Each of their
four GPU pods passed 47 tests with zero skips. Both owned pools were then
paused and verified to have zero pods; independent workloads were untouched.

Against primary L2_repo at shared step 40, candidate p_err improved on earnings,
openml and ls_clean, and slightly worsened on ls_other. The same direction held
at step 50 against fresh L2. Against the fresh control, steady-window median
actor update time was 7.411 versus 7.389 seconds, whole-step time 43.360 versus
44.026 seconds, and peak allocated memory 88.355 versus 88.474 GiB. These
cross-pool observations do not establish a speed or memory benefit, and this
single short run does not prove long-run non-inferiority. No packing-ratio
counter was logged. Full trajectories and resource-release receipts remain in
`tmp/l2_pack_ab50/report.md` and its sibling artifacts, outside Git.

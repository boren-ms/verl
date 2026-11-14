# Current image verification

Candidate update: Main ASR plugin packaging, normalizers, NumPy 2.2.6 and
Liger 0.8.2 (matching Main's dependency floors), and editable-only setup.
The Torch 2.11/CUDA 13/vLLM 0.24/Ray 2.54 stack is retained.

Published build: [2282198](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2282198),
completed 2026-11-14 02:41 UTC. Build source is bootstrapping commit
`b90f8d3a4a48a7274779d2dc21d72f171f7afa41`; internal wheels reuse verified
artifact `verl-context` from build 2200290 with checksum and package-list checks.

```text
iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-2282198
iridiumsdc.azurecr.io/boren/verl-remax@sha256:d92ae2602a53dbe81fea9ee6e6c51325275b56ad2f71f26659b93155f47ff830
```

Docker build checks and push succeeded. The overall pipeline result is
`partiallySucceeded`: the injected Secure Supply Chain Analysis task reported
10 external-image-reference violations. This warning is separate from the
successful Docker task and has not been resolved by this update.

Fresh-container verification passed on 2026-11-14 in the dedicated
`base3a-image-n2-critical` pool (`prod-westus2-cw-6`, namespace `boren`,
`team-critical`, two nodes with eight H100 80GB GPUs each):

| Pod | UID | Physical node |
| --- | --- | --- |
| `base3a-image-n2-critical-0` | `5f3f8ebd-eb32-418e-aa8c-d8f67de056cc` | `dh115-r175-node-02` |
| `base3a-image-n2-critical-1` | `119fd79a-2cbe-4e69-8a65-dfdf7f9d7ca5` | `dh115-r209-node-02` |

Both actual `main` imageIDs match the digest above. On each node, before
checkout installation, the verifier loaded `verl` from `/opt/verl-source` and
passed package versions, backward on every GPU, causal-conv1d backward,
FlashAttention backward, FlashInfer RMSNorm, FLA backward, ReMax/config imports,
Chinese/Vietnamese/French normalizers, and managed Ray worker versions.
Two-node torchrun covered all 16 GPUs: NCCL all-reduce returned the expected
sum **136** and both ranks' launchers exited successfully.

The first verifier invocation exposed a helper error: Hydra config subdirectories
are not Python packages. The corrected verifier uses `initialize_config_dir`,
as the existing recipe tests do, and was synced from the checkout. The published
image's embedded helper predates this correction; the checks above used the
corrected helper against the image's unchanged source and dependencies.

Normal `quick_install.sh` then installed only the three editable source packages,
without changing runtime dependencies. Both nodes passed the verifier again,
this time loading `verl` from `/root/code/verl`. Snapshot revision was
`8f23049cba07f4d1f4abe44f71137431929c4836` (Brix dirty-worktree snapshot).
After these checks, both pods retained their original UIDs and digest, remained
Ready, and had zero container restarts.

Full plugin runtime tests passed on this environment:
`python -m pytest -q plugins/phimm/tests plugins/qwen35_audio/tests`:
**243 passed**, 31 warnings, 49.59 seconds.

Startup skipped broad TorchFlow setup. Optional Beacon setup reported a missing
`beacon_courier` monorepo dependency; Ray nevertheless started with the expected
two nodes and 16 GPUs. TorchFlow's declared Ray 2.46 pin conflicts with the baked
Ray 2.54 package, while both managed-worker runtime checks passed on Ray 2.54.

Real Base3a preflight passed: 55,664 training rows with the exact six-source
mix, target 868 updates, and 2,767 validation parents / 3,102 segments.
All six training sources passed HF/vLLM prompt-token equivalence with real
audio and the configured model. No random segmentation was enabled.

Original Base3a training was submitted as
`remax_main_base3a_image_2611r2_20261114`, using source snapshot
`1e7711a024034cf47f44346ded4361f793f52199`. At 2026-11-14 04:47 UTC,
Ray was RUNNING and logs confirmed 91 of 868 optimizer updates, with completed
nine-dataset validation at steps 0, 20, 40, 60 and 80. The step-90 checkpoint's
62 files, including all 16 model/optimizer/extra-state ranks and loader state,
matched local and Blob sizes; restoration and file hashes have not been tested.

At shared validation step 80, seven of nine datasets met the per-dataset
maximum regression of 0.1 percentage points. `earning_chunk_verb` regressed by
0.1884 points and `earnings22_u2` by 0.1692 points. Reference validation uses the
verified uniform rescore, with official metrics retained separately. The
reference's first 120 training steps used an older scorer, and its effective
no-repeat window was 100 versus the candidate's corrected 512. This is a
multivariable comparison and does not establish a Docker-only effect.

Training to step 868, final export and the full 35-source model-quality
comparison remain pending. Environment checks and early validation do not
establish convergence or final quality parity.

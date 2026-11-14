# Main-merge Base3a completion verification

Final observation: **2026-11-14 22:51 UTC**. Image build **2282628**, with
editable ASR compatibility fixes through `ebfcf9368`, completed the original
BF16 Base3a recipe without random segmentation or FP8.

| Stage | Verified result |
| --- | --- |
| Training | `remax_main_base3a_mainmerge_2611r2_20261114_r2`: Ray `SUCCEEDED`, optimizer 868/868 |
| Final validation/checkpoint | Nine sources at step 868; 62 uploaded files, including all 16 ranks' model/optimizer/extra states |
| HF export | `base3a_mainmerge_export868_20261114_r2`: `SUCCEEDED`; 176 LoRA pairs, 781 tensors, finite values and matching base keys/shapes |
| Full evaluation | `eval_base3a_mainmerge_868_openfull_20261114`: `SUCCEEDED`; all 35 sources, 148,013 parent rows |
| Evaluation consistency | Per-source error/reference counts match official metrics; uploaded file sizes verified; all 35 parent-ID/full-reference hashes match the reference |
| Resource release | Dedicated pool suspended, zero matching pods at 22:51 UTC; 16 H100 GPUs released |

The reference is original Base3a step 868, evaluation
`eval_main_base3a_2611r2_s868_openfull_zip`. The canonical 33-source comparison
has **14 improvements, 18 regressions and one tie**; two supplementary French
sources both regress. OpenEN/OpenML/OpenLong/OpenExt relative WERR is
-0.05% / -0.29% / -1.87% / +0.25%. These results do not establish quality parity
or isolate the effect of Docker: Main and dependencies changed together, the
reference scorer changed after step 120, and the effective no-repeat window
changed from 100 to 512. The cohort check did not compare audio bytes.

Run links:

- [Training](https://msaip.wandb.io/genai/v2611/runs/upm061t3)
- [Full evaluation](https://msaip.wandb.io/genai/v2611_eval/runs/x3durylv)

Durable artifacts:

```text
HF export:
az://orngwus2cresco/data/boren/outputs/v2611/remax_main_base3a_mainmerge_2611r2_20261114/global_step_868/hf_merged/
Evaluation details and metrics:
az://orngwus2cresco/data/boren/outputs/v2611_eval/eval_base3a_mainmerge_868_openfull_20261114/val_data_gen/
Final Excel/PDF comparison and evidence:
az://orngwus2cresco/data/boren/data/verl/base3a-mainmerge-20261114/reports/20261114T224749Z/
```

All observations below are historical checkpoints in the build and recovery
process; their pending/running labels are superseded by this completion record.

## Historical model-load and audio-generation smoke

2026-11-14: sourcefixebfcf9368 preserves originalmtp./lora_A./lora_B.
exclusions using a filtered input stream becausevLLM0.32 removed
AutoWeightsLoader.skip_prefixes. Native strict-loader and processor regressions:
4passed,11.42s. Unrelated unknown weights still fail.

A real single-engine smoke passed on dedicatedpool node1GPU0: originalBF16
model weights19.68GiB loaded, audioencoderforward,FlashInfer sampling JIT,
KVcache andCUDAgraphs initialized,1secondsyntheticaudio generated tokenIDs
[271,13,248046]. SameLoRAcapacity andngram15/window512;smokecontext1792
andmax4tokens only. This proves inference usability,not ASRquality or a
trainingoptimizerstep. Engineinit145.06s;process exited0 andGPUreleased.
_r1 had failed beforeoptimizer0 at the loaderinterface;it is superseded.
Next recovery uses a newRayID and the same experiment/output,target868.

Earlier observations below are historical:

# ASR rollout runner compatibility recovery

2026-11-14: the first Main-merge job failed during vLLM startup before
optimizer0. vLLM0.32 selected V2, which rejects DeepSeek OCR's V1
NGramPerReqLogitsProcessor. Source commit434117422 explicitly selects
VLLM_USE_V2_MODEL_RUNNER=0; original ngram15/window512, BF16 and recipe
remain unchanged. Actual native frontend validation passed;45config/sampling/
agent tests passed. No local or remote checkpoint was created by the failed
startup. Recovery uses a new Ray submissionID suffix_r1 with the same
experiment/output and auto-resume, preserving absolute target868.
Model weights/rollout optimizer validation remains pending. Earlier RUNNING
observations below refer to the first submission and are historical.

# Main-merge Base3a training submitted

2026-11-14 07:58 UTC: Ray job `remax_main_base3a_mainmerge_2611r2_20261114`
is RUNNING in case-owned2x8H100 pool. Runtime confirms55664training rows,
2767validation parents and868total steps. Source snapshot5762c29a995d6d24400d675fe6eda6df3c2e72df.
OriginalBF16 recipe retained; no random segmentation/FP8. Worker/model startup
in progress; no optimizer update, completed validation or checkpoint yet.
This validates image/runtime/preflight usability only; quality comparison pending.

# Main-merge real ASR preflight passed

2026-11-14: real Base3a preflight passed after source fix `c1b225273`:
55664training rows,2767validation parents/3102existing segments,target868.
HF and vLLM expanded prompt IDs match for all6training sources.
The fix adapts the ASR plugin to vLLM0.32's normalized-audio callback and
integer token-ID prompt targets (including cached targets). No recipe or
dependency changed; synchronized editable source is required over the baked
image's older processor. Runtime suite after callback fix:299passed; final
ragged-audio and token-target regression tests:2passed. GPU/NCCL evidence below.
Training submission is next; no optimizer or quality result yet.

# Main-merge image GPU/runtime verification passed

2026-11-14: image2282628 (digest4a45b64662f9660b4faae9b822d1c88a1ac5967462ef73d78ff8d91f2c31c099)
passed baked-source and editable-source GPU verifiers on both8-H100 nodes.
Cross-node NCCL passed across16GPUs (all-reduce sum136); both launchers exited0.
`quick_install.sh` reused all dependencies and installed only3editable source packages.
Runtime suite: **298 passed**,33warnings,53.70s (ASR/model plugins plus upstream
trainer_base/checkpoint/metric/rollout-config CPU tests).
Snapshot a0fa19fb409ba98f5e8dd00253c69f4900dabb62 on both nodes.

Pool base3a-mainmerge-n2-critical, prod-westus2-cw-6/boren:
- pod0 UID92e4933c-8691-4284-b2dc-07e8f10f89a4, node dh115-r262-node-01.
- pod1 UIDf705edd5-79a2-4694-a10d-622e005b4999, node dh115-r416-node-02.
Both actualimageID match the published digest; Ready, zero restarts after checks.
Reports archived at az://orngwus2cresco/data/boren/data/verl/base3a-mainmerge-20261114/verification/.
Real ASR preflight is running; no optimizer step or model-quality result yet.
CuPy reports both cuda12x/cuda13x distributions inherited in the environment;
required GPU checks and runtime tests passed, with no CuPy-path failure observed.

Publication and earlier evidence follows:

# Main-merge image published; GPU validation pending

Build [2282628](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2282628)
published 2026-11-14 07:10 UTC from bootstrapping `2df12332db8ef931d5f339ef7d6df124f99d1049`.
Docker build/CPU checks and push succeeded. Overall `partiallySucceeded` reflects
injected external-image-reference scan warnings (11); the Docker task succeeded.

```text
iridiumsdc.azurecr.io/boren/verl-remax:torch213-vllm032-2282628
iridiumsdc.azurecr.io/boren/verl-remax@sha256:4a45b64662f9660b4faae9b822d1c88a1ac5967462ef73d78ff8d91f2c31c099
```

The NVVM13.3.73 compilation fix passed image construction. Fresh-container
GPU/NCCL/runtime tests and ASR preflight remain pending. Only case-owned pool
`base3a-mainmerge-n2-critical` was patched with UID/old-image preconditions
after both nodes showed no Ray jobs, GPU processes, reservations or demands.
No training submitted; original Base3a BF16 target remains868.

Earlier build and old-image evidence follows (historical):

# Main-merge package update (pending)

Current build: [2282628](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2282628), bootstrapping source `2df12332db8ef931d5f339ef7d6df124f99d1049`.
Target: Torch2.13.0/cu130, vLLM0.32.0, Transformers5.16.1, TransferQueue0.1.11.

Build2282558 failed compiling causal-conv1d: nvcc/ptxas13.3.73 used an
unbounded nvidia-nvvm13.5.37 frontend, emitting PTX9.5 for an assembler that
accepts9.3. Fixed by pinning nvidia-nvvm13.3.73 and adding a minimal sm90 CUDA
compile before the full extension build. AST/shell/heredoc/YAML/Ruff/diff
checks passed; corrected image compilation and all GPU/runtime checks pending.
No image was published by the failed build. Training is not submitted.

Previous attempt details (historical):

# Main-merge package update (pending)

Requested 2026-11-14: use Main's recommended Torch 2.13.0/cu130,
vLLM 0.32.0, Transformers 5.16.1 and released TransferQueue 0.1.11.
The Dockerfile also follows Main's FSDP FLA/Liger pins and TileLang/cuSPARSELt
overrides. Orange Ray stays at 2.54.0; protobuf is upgraded to 5.29.6 for vLLM.
The old Torch 2.11 FlashAttention/causal-conv1d wheels are excluded.

Build submitted: [2282558](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2282558),
bootstrapping source `1609e9cf2d6cfb121d18b4659de40bc94b5c1bb0`, branch
`boren/verl-main-torch213-vllm032-20261114`. The package list changed, so this
build prepares a new wheel artifact. Superseded build 2282554 was canceled
before image publication to correct the verifier's Transformers 5.16 FLA path.
The staging condition now honors cancellation.

Latest observation: source/wheel staging completed. The image-build job is
pending because all 12 `iridium-builder` agents are offline. There is no new
image digest. Staging's injected Secure Supply Chain Analysis reported 11
external-image-reference violations (`succeededWithIssues`); Docker execution
has not started, so this is not a Docker build result. The existing 30-minute
case monitor tracks build 2282558 and the subsequent image/runtime checks.

Local dependency resolution (310 requirements), shell/heredoc syntax, Python
AST, pipeline YAML and changed-file diff checks passed. Image build, immutable digest, fresh-container GPU,
NCCL, runtime plugin tests and real-data preflight are pending. Do not submit
training on the old image using this checkout. Only this task's
`base3a-mainmerge-n2-critical` pool may be updated after ownership and occupancy
checks; other experiments retain their current images.

The historical evidence below applies only to the previous image.

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

Optimizer training and model-quality comparison remain pending. These checks
establish environment and input readiness; they do not establish convergence
or quality parity.

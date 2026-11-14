# Main-merge image: completed Base3a validation

As of 2026-11-14 22:51 UTC, build **2282628** has completed the BF16
Base3a training, export and full-evaluation pipeline with the compatible
editable source fixes through `ebfcf9368`:

- Training `remax_main_base3a_mainmerge_2611r2_20261114_r2` succeeded at
  optimizer step **868/868**, with final nine-source validation and a complete
  16-rank checkpoint uploaded.
- The exact step-868 HF export and all **35** evaluation sources succeeded;
  148,013 parent rows were collected. Parent IDs and full-reference hashes
  match the original Base3a evaluation for every source.
- The canonical 33-source comparison has 14 improvements, 18 regressions and
  one tie. Both supplementary French sources regressed. Completion establishes
  runtime usability, but does not establish quality parity. Main, dependencies,
  scorer history and the effective no-repeat window differ from the reference.
- The dedicated `base3a-mainmerge-n2-critical` pool was verified suspended with
  zero matching pods, releasing 16 H100 GPUs.

See [VERIFICATION.md](VERIFICATION.md) for run identities and durable artifacts.
The earlier observations below describe intermediate states and are historical.

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

# ASR training image

The image contains the Main trainer, `plugins/phimm`, the Qwen audio plugin,
text normalizers and the pinned CUDA/vLLM stack. It preserves Orange startup
and managed Ray. `quick_install.sh` checks the image and installs only editable
source with uv; a runtime mismatch fails instead of upgrading a running cluster.

Published and GPU-verified Main-merge image
([build 2282628](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2282628)):
Torch 2.13.0/cu130,
vLLM 0.32.0, Transformers 5.16.1, TransferQueue 0.1.11, TorchVision 0.28.0,
Liger 0.8.4 and FLA 0.5.2. The TileLang 0.1.15 and x86_64 cuSPARSELt
0.8.0 overrides match upstream `pyproject.toml`. TorchAudio stays at its last
release, 2.11.0, matching upstream. Ray stays at Orange-tested 2.54.0;
protobuf moves to 5.29.6 to meet vLLM's minimum. FlashAttention comes from the
upstream Torch 2.13 wheelhouse; causal-conv1d is rebuilt for the new ABI.

The current checkout's installer/verifier require this new stack. Do not use
them with the older image below. Its verification belongs to the old stack
and does not establish compatibility of the new Main-merge stack.

Previous published image with fresh-container GPU and 16-card NCCL checks:

```text
iridiumsdc.azurecr.io/boren/verl-remax@sha256:d92ae2602a53dbe81fea9ee6e6c51325275b56ad2f71f26659b93155f47ff830
```

Unique tag: `torch211-vllm024-2282198`. See [VERIFICATION.md](VERIFICATION.md)
for the build warning, corrected verifier invocation, and training-validation
status. Use the matching historical checkout's verifier for that older image.

Build using pipeline 2910 in the Mimco bootstrapping repository. Follow
[the Docker skill](../../.github/skills/verl-orange-docker-update/SKILL.md) for
a dedicated build branch, verified wheel-artifact reuse, unique tag and digest.
Stage into an **empty** source directory to prevent files removed from the
checkout remaining in the image. Newly added runtime files must first be added
to the Git index. Do not copy credentials or temporary experiment state.

```bash
python docker/remax/stage_source.py <bootstrapping>/images/verl-remax/source
```

Copy `Dockerfile` beside the staged source and `orange-pipeline.yaml` to the
build branch's `.azuredevops/kube-orange-box.yaml`. The pipeline validates the
reused wheel list and checksums, builds current source, and publishes tag and
digest artifacts. It does not update shared image tags.

For a new dedicated two-node training pool, set `VERL_DOCKER_IMAGE` to the
published immutable digest and create with `skip_rcall_user_config=true` and
`extra_rcall_configs=<checkout>/docker/remax/image.rcall.py`. Before installing
anything, verify the fresh pod's actual imageID, then run `verify.py` against
`/opt/verl-source` and `verify_nccl.py` across all allocated GPUs. Run setup and
repeat the verifier from `/root/code/verl`; retain both reports and pod identity.

After `bpush <pool>`, submit the original Base3a with:

```bash
python plugins/phimm/submit.py --name <unique-name>
```

The submitter prepares editable packages on every live GPU Ray node before
submitting. Setup temporarily requests one GPU on each node so Ray exposes CUDA
for its availability check. It composes the YAML and checks the requested
node/GPU topology before starting the job, retaining receipts under
`tmp/phimm-submissions/<name>/`. It checks for a duplicate submission ID before setup and refuses
to reuse one. The name also supplies `trainer.experiment_name` unless explicitly
overridden. `--config <installed-name>` chooses a different config;
`--evaluate` selects full evaluation. Hydra overrides remain positional:

```bash
python plugins/phimm/submit.py --name <unique-eval-name> --evaluate \
  actor_rollout_ref.model.path=<verified-merged-HF-path>
python plugins/phimm/submit.py --name <unique-name> --dry-run
```

No random-segmentation option is added by the submitter. Topology, source mix,
rewards and training target come from the selected YAML; check live capacity
and the composed config before submission. The submitter uses the existing
remote checkout and explicitly sets its repository/plugin source paths for Ray
workers, including workers started before editable setup. It does not upload an
additional Ray working-directory copy.

Publication and fresh-pod evidence for this update are recorded in
[VERIFICATION.md](VERIFICATION.md).

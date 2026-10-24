# ReMax environment verification

## Fresh base-image node

Pool `verl-image-verify-oct23`, cluster `prod-westus2-cw-6`, namespace `boren`.
Pod UID `e2596f3c-4823-4e1f-8987-9f44afe7fd71`, host `dh115-r406-node-02`.
Eight NVIDIA H100 80GB GPUs. Base image `iridiumsdc.azurecr.io/rcall:crow-1676263`,
digest `sha256:acd0aa02a41ac9739da6ad41a1fef9334af132eed904e94ff8453754ea7d404e`.

Passed on the base node after installing the environment:

- Final `quick_install.sh`, including FLA forward/backward and convolution checks.
- Repeated setup: 22.8 seconds, preserving the installed stack.
- Linear-attention install-only mode with no visible GPU.
- CUDA backward on all eight GPUs, FlashAttention backward, FLA backward,
  ReMax imports, and a task on the managed Ray cluster.
- Eight-GPU NCCL all-reduce: expected sum 36.0 on every rank.

The base-node runtime used Ray 2.54.0 and protobuf 5.29.5. This runtime Ray version
is distinct from the untouched base Docker image's Ray 2.46.0.
FlashInfer RMSNorm was added to the verifier later and was not part of these
base-node results. This pool was paused after its checks.

## First published image and startup issue

[Build 2200290](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2200290)
published `iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-2200290`:

```text
sha256:0d7e66dc8a52b697c8fb1b7c0cb3e3cb883537e32e5e8c9ba2d9d682382a9e45
```

Its build, all 29 wheel downloads, checksum validation, and push succeeded.
The overall pipeline was partially succeeded because of injected supply-chain
scan warnings about external registry references elsewhere in bootstrapping.

The fresh custom-image pool `verl-image-smoke-oct23` eventually became Ready.
Runtime verification failed at the Torch version check: the image's freeze
record contained Torch 2.11.0+cu130, but startup had replaced it with OpenAI
Torch 2.6.0. Brix startup logs identify the cause as TorchFlow's monorepo package
installation. The launch hook recreates TorchFlow's existing setup marker before startup
to preserve the baked environment. Candidate runtime checks below confirm
this behavior; the final immutable image still requires fresh-pod verification.

RCall launch precedence was also checked against the owning loader.
`image.rcall.py` loads normal user settings first, then overrides the job image.
Launch it with `skip_rcall_user_config=true`; otherwise the Orange user hook
loads last and replaces `DOCKER_IMAGE`. `INFRA_IMAGE` remains the base image.

## Self-contained Dockerfile rebuild

All installation commands, dependency pins, CUDA setup, and build checks are
inline. The Dockerfile does not execute either installer or read an external
requirements file. Source and internal wheels are build-context inputs.

The rebuild reuses the checksummed wheel artifact from 2200290, validates the
exact package list, and builds the current source and Dockerfile. An initial
build exposed the base image's reliance on a login-shell Python path; the
Dockerfile now sets that path explicitly. NumPy is installed from the index
before resolving the offline Torch wheel bundle. The custom image uses its own
version/kernel verification instead of TorchFlow's monorepo setup-version check.

Superseded build:
[2201217](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2201217),
bootstrapping commit `a95e02d3916a1804d7a0051695d31f9a19a13b0a`.
The Docker build, CPU-only version/extension checks, and push succeeded.
Tag: `iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-2201217`.
Digest: `sha256:93e436d5214c6c3fd25df4992a1f0c230b938f3081b8ff603bee8241731940d4`.
The overall run is partially succeeded because the injected supply-chain scanner
reports external-registry references elsewhere in bootstrapping; the Docker
task succeeded. Fresh-pod verification failed: FlashInfer RMSNorm exposed a
CUTLASS/TVM FFI incompatibility, and Orange bootstrap selected Ray 2.59.0,
which timed out during cluster startup. Candidate fixes pin apache-tvm-ffi
0.1.10, CUTLASS DSL 4.8.1, and Ray 2.54.0. Ray 2.54.0 was confirmed installed
before startup, but the timeout persisted. Ray runtime-agent logs identified
`ModuleNotFoundError: No module named trainflow`: skipping the TorchFlow installer
also skipped the required runtime-plugin registration. The launch hook now
registers the mounted TorchFlow source with `--no-deps`, and the image installs
Ray dashboard dependencies with `ray[default]`.

The candidate stack passed on pod `31af9f0f-806f-430e-b638-23d1eae906e1`
with zero restarts after the plugin-registration fix: all eight GPUs, convolution
and FlashAttention backward, FlashInfer RMSNorm, FLA backward, ReMax imports,
and a real managed Ray worker. Eight-rank NCCL all-reduce returned 36.0.
These checks used temporary pre-start package installs on the test pool; they
do not verify the replacement immutable image.

Replacement build [2201665](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2201665)
uses bootstrapping commit `8e45cf4719f49604ed8ae50146af1ee63aacffd9`.
The replacement immutable image was built and verified as recorded below.

## Final immutable image

[Build 2201665](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2201665)
built and pushed the self-contained Dockerfile from bootstrapping commit
`8e45cf4719f49604ed8ae50146af1ee63aacffd9`, reusing the checksummed wheel artifact
from build 2200290. CPU package-version and compiled-extension checks passed.

```text
Tag: iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-2201665
Digest: sha256:d3624316721afeb2cf072b803cecac5a01e45f691b84e58e2cb43fc70d565b4e
```

Fresh pod: `verl-image-smoke-oct23-0`, UID
`a2546a89-c731-4013-900d-eb4513eaaefe`, node `dh115-r229-node-01`,
cluster `prod-westus2-cw-6`, namespace `boren`. This was a fresh container on
an existing GPU node. The actual container imageID matched the digest above.
The pod became Ready with zero restarts.

The temporary candidate package-install hook was removed before this pod was
created. The production launch hook registered the mounted TorchFlow source
with `--no-deps` and skipped its broad dependency installer. Orange's separate
bootstrap registered its infrastructure packages; it retained the baked Ray
2.54.0, protobuf 5.29.5, and Torch 2.11.0+cu130.

Before running `quick_install.sh`, `verify.py` passed against `/opt/verl-source`:

- Pinned package versions, including TVM FFI 0.1.10 and CUTLASS DSL 4.8.1.
- CUDA backward on all eight NVIDIA H100 80GB GPUs.
- Causal convolution and FlashAttention backward.
- FlashInfer RMSNorm against a Torch reference.
- Qwen3.5 FLA forward/backward and ReMax imports.
- A managed Ray worker with matching Ray, protobuf, and Torch versions.

Eight-rank NCCL all-reduce also passed with sum 36.0. Normal
`quick_install.sh` recognized the baked marker, skipped the main dependency
installation, rebound editable packages to `/root/code/verl`, and passed its
GPU kernel checks. The full verifier passed again from that checkout, retaining
Ray 2.54.0, protobuf 5.29.5, and Torch 2.11.0+cu130.

The overall Azure DevOps run was `partiallySucceeded` because the injected
supply-chain scanner reported ten external-registry-reference violations in
the bootstrapping repository. The Docker build/push task succeeded. The base
monorepo and vLLM also have package-metadata conflicts reported by pip; these
results establish the explicit runtime smoke checks, not a clean `pip check`.

The verifier covers pinned versions, managed Ray driver/worker consistency, all-GPU backward,
causal-conv1d backward, FlashAttention backward, FlashInfer RMSNorm against a
Torch reference, FLA backward, ReMax imports, and a managed Ray worker.
NCCL all-reduce is verified separately across eight ranks.
These checks do not establish a completed ReMax optimizer step, model rollout
quality, or validation metrics.

Detailed build logs, pod identities, and runtime receipts are retained under
ignored `tmp/remax-image/`; private launch configuration is not committed.

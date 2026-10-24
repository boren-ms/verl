# Orange ReMax environment image

This image extends the same `iridiumsdc.azurecr.io/rcall:crow-1676263` used by
current verl pools. It retains Orange's startup and managed Ray implementation.
The Dockerfile contains all installation commands and dependency pins; it does
not execute installer scripts or read an external requirements file. The build
context still supplies the application source and internal wheels. Build-time
constraints select Ray 2.54.0 and preserve protobuf 5.29.5. The verifier checks
the runtime versions and driver/worker consistency. Job setup
does not replace Ray after the cluster starts.

## Build with Orange's existing pipeline

The Orange wiki documents two distinct networks/identities:

- `orange-builder` stages Blob wheels using `sc-msi_orange-builder` and the existing Orange Tailscale setup template.
- `iridium-builder` builds and publishes using `sc-msi_iridium-builder`.

For rebuilds with an unchanged `packages.list`, set pipeline parameter
`reuseContextBuildId` to a successful prior build (for example `2200290`). This
skips Orange downloads, validates the artifact checksums and exact package list,
and builds the current source and Dockerfile on `iridium-builder`.

The implementation is `orange-pipeline.yaml`. On a dedicated bootstrapping
branch it replaces `.azuredevops/kube-orange-box.yaml`, allowing pipeline 2910
(Kube Orange Devbox) to use its existing authorized connections. It publishes
only `iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-<build ID>`.
It does not publish `latest` or overwrite the shared Orange images.

The pipeline downloads the exact 29 files in `packages.list`, with four concurrent downloads and a 30-minute bound per file. This avoids depending on wildcard listing behavior across Azure CLI versions.

Stage the current runtime source with:

```bash
python docker/remax/stage_source.py /path/to/bootstrapping/images/verl-remax/source
cp docker/remax/Dockerfile /path/to/bootstrapping/images/verl-remax/Dockerfile
```

For a builder with both Blob and registry access, `prepare_context.sh` prepares
a standalone context. Build using Docker BuildKit. Blob package files are mounted
for installation and are not retained as a duplicate wheel layer. The package
index configured by the base image must also be reachable for remaining Python
dependencies. No credentials are copied into the build context.

```bash
bash docker/remax/prepare_context.sh /absolute/context
DOCKER_BUILDKIT=1 docker build --progress=plain -t "$VERL_DOCKER_IMAGE" /absolute/context
```

The Dockerfile installs the environment directly and checks pinned versions and
compiled extension loading without requiring GPUs. Dependency pins are explicit
in the Dockerfile; keep them aligned with the runtime installers when updating
the supported stack. The candidate image marker is
validated on job startup, and CUDA tests still run. The launch hook creates TorchFlow's `/tmp/DID_TORCHFLOW_SETUP` marker so
startup preserves the baked Torch stack.
`SKIP_TORCHFLOW_SETUP_CHECK=1` selects this curated environment instead of
TorchFlow's monorepo setup-version check; the Docker build and fresh-pod verifier
validate its package versions and kernels explicitly. Normal `quick_install.sh`
also binds editable installs to the current pushed job checkout.
Brix startup does not preserve the image's `/tmp` setup marker, so
`image.rcall.py` recreates it immediately before TorchFlow's startup hook.
The launch hook also registers the mounted TorchFlow source with `--no-deps`
so Orange can import its required `trainflow.ray_plugin` without reinstalling
the monorepo dependency stack. Use this launch configuration for custom images.

## Published image

[Build 2201665](https://dev.azure.com/project-argos/Mimco/_build/results?buildId=2201665)
published `iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-2201665`.
Its immutable reference is:

```text
iridiumsdc.azurecr.io/boren/verl-remax@sha256:d3624316721afeb2cf072b803cecac5a01e45f691b84e58e2cb43fc70d565b4e
```

The earlier 2200290 and 2201217 images are superseded by startup and
FlashInfer compatibility fixes. See `VERIFICATION.md` for the build and fresh-pod test results.

## Verify on a new pool

Use a unique pool name and select the image only for that launch. Include
`skip_rcall_user_config=true`: this launch file loads the normal user configuration
first, then overrides the job image. Otherwise the user's Orange hook loads last
and replaces the custom image. The infrastructure image stays at the base image.


```bash
export VERL_DOCKER_IMAGE=iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-<build-ID>
twdev create-ray-devbox cluster=prod-westus2-cw-6 num_pods=1 num_gpu=8 \
  job_name=verl-image-verify-unique priority_class=team-critical \
  team=team-moonfire-speech \
  skip_rcall_user_config=true \
  extra_rcall_configs=/absolute/path/to/docker/remax/image.rcall.py
bpush verl-image-verify-unique
brix ssh verl-image-verify-unique -- 'bash -lc "cd /root/code/verl && bash quick_install.sh && python docker/remax/verify.py"'
```

Confirm the main container image and digest using `kubectl --context
prod-westus2-cw-6 -n boren get pod <pool>-0 -o json`, including
`spec.containers[].image` and `status.containerStatuses[].imageID`.
`verify.py` exercises all allocated GPUs, FlashAttention and FLA forward/backward,
FlashInfer RMSNorm against a Torch reference, ReMax imports, and a real managed Ray worker. The report is written to
`/tmp/verl-env-verification.json`. This is an environment smoke test; a completed
ReMax optimizer step is a separate training validation.

Run `python -m torch.distributed.run --standalone --nproc-per-node=8 docker/remax/verify_nccl.py` to verify NCCL all-reduce across the eight GPUs.

After verification, pause only the dedicated verification pool. Preserve build
logs, the image digest, and verification results before releasing it.

Sources: Penny.wiki `Personal-notes/antonsc/Docker-Images/Building-Docker-image.md`,
`9.-Platform-Infra/Transfer-rcall-image-from-OAI-to-iridium-ACR.md`, and the actual
bootstrapping `.azuredevops/kube-orange-box.yaml` plus its authentication templates.

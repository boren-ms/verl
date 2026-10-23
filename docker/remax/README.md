# Orange ReMax environment image

This image extends the same `iridiumsdc.azurecr.io/rcall:crow-1676263` used by
current verl pools. It retains Orange's startup and managed Ray implementation.
The installed dependency stack comes from `quick_install.sh`; Ray must remain
at the base image version and protobuf must remain 5.29.5.

## Build with Orange's existing pipeline

The Orange wiki documents two distinct networks/identities:

- `orange-builder` stages Blob wheels using `sc-msi_orange-builder` and the existing Orange Tailscale setup template.
- `iridium-builder` builds and publishes using `sc-msi_iridium-builder`.

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

The image build runs `quick_install.sh --image-build`: installs packages and
checks extension loading without requiring GPUs. The candidate image marker is
validated on job startup, and CUDA tests still run. Normal `quick_install.sh`
also binds editable installs to the current pushed job checkout.

## Verify on a new pool

Use a unique pool name and select the image only for that launch:

```bash
export VERL_DOCKER_IMAGE=iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-<build-ID>
twdev create-ray-devbox cluster=prod-westus2-cw-6 num_pods=1 num_gpu=8 \
  job_name=verl-image-verify-unique priority_class=team-critical \
  team=team-moonfire-speech \
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

# Verification record — 2026-10-23

## Fresh Orange base node

Pool `verl-image-verify-oct23`, cluster `prod-westus2-cw-6`, namespace `boren`.
Pod UID `e2596f3c-4823-4e1f-8987-9f44afe7fd71`, host `dh115-r406-node-02`.
Eight NVIDIA H100 80GB GPUs. Base image `iridiumsdc.azurecr.io/rcall:crow-1676263`,
digest `sha256:acd0aa02a41ac9739da6ad41a1fef9334af132eed904e94ff8453754ea7d404e`.

Passed:

- `quick_install.sh`, including final FLA forward/backward and convolution check.
- Repeated `quick_install.sh`: 22.8 seconds, preserving the installed stack.
- `CUDA_VISIBLE_DEVICES= bash quick_install_linear_attention.sh --install-only`.
- `verify.py`: CUDA backward on all GPUs, FlashAttention backward, FLA backward,
  ReMax import, and a task on the pre-existing managed Ray cluster.
- `torchrun --standalone --nproc-per-node=8 docker/remax/verify_nccl.py`:
  NCCL all-reduce passed on all eight GPUs, expected sum 36.0.

Final versions: Torch 2.11.0+cu130, vLLM 0.24.0, Transformers 5.7.0,
FlashAttention 2.8.3, FlashInfer 0.6.12, FLA/fla-core 0.6.0,
causal-conv1d 1.7.0, TileLang 0.1.9, apache-tvm-ffi 0.1.9,
ml-dtypes 0.5.4, NumPy 1.26.4, tensordict 0.10.0, bitbank 0.2.0.
Ray remained 2.54.0 and its worker reported protobuf 5.29.5.

The initial installer passed its convolution-only check but FLA backward failed
on Hopper with Triton 3.6.0. The revised installer pins TileLang and exercises
backward. During development an unconstrained dependency install attempted to
replace Torch/CUDA; the staged supported stack was restored before the final
checks above. The final script uses `--no-deps` for the added backend packages.

This establishes environment and distributed kernel functionality. It does not
establish a completed ReMax optimizer step, model rollout quality, or validation
metrics.

Detailed local receipts are under `tmp/remax-image/` (not committed):
`final-installer-evidence.txt`, `base-final-receipt.txt`, and
`base-pod-identity.txt`.

## Custom image

Dedicated bootstrapping branch: `codex/verl-remax-image-oct23`.
Pipeline 2910 uses Orange's builder identities and publishes into
`iridiumsdc.azurecr.io/boren/verl-remax` with a build-ID tag.
Image build and fresh-image pod verification are tracked separately from the
base-node checks above; they are not yet recorded as passed here.

# ASR training image

The image contains the Main trainer, `plugins/phimm`, the Qwen audio plugin,
text normalizers and the pinned CUDA/vLLM stack. It preserves Orange startup
and managed Ray. `quick_install.sh` checks the image and installs only editable
source with uv; a runtime mismatch fails instead of upgrading a running cluster.

Latest published image with fresh-container GPU and 16-card NCCL checks:

```text
iridiumsdc.azurecr.io/boren/verl-remax@sha256:d92ae2602a53dbe81fea9ee6e6c51325275b56ad2f71f26659b93155f47ff830
```

Unique tag: `torch211-vllm024-2282198`. See [VERIFICATION.md](VERIFICATION.md)
for the build warning, corrected verifier invocation, and training-validation
status. Use the current checkout's verifier when checking this image.

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

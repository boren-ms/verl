# Orange ReMax environment image

This image extends the same `iridiumsdc.azurecr.io/rcall:crow-1676263` used by
current verl pools. It retains Orange's startup and managed Ray implementation.
The Dockerfile contains all installation commands and dependency pins; it does
not execute installer scripts or read an external requirements file. The build
context still supplies the application source and internal wheels. Build-time
constraints select Ray 2.54.0 and preserve protobuf 5.29.5. The verifier checks
the runtime versions and driver/worker consistency. Job setup
does not replace Ray after the cluster starts.

## 日常更新：使用脚本

在本地仓库根目录执行。入口是 `docker/remax/update_image.py`，只使用 Python
标准库，凭据从现有 Azure DevOps Git credential helper 获取。
`AZURE_DEVOPS_EXT_PAT` 可覆盖 REST API 认证；clone/push 仍使用 Git 自身配置的认证。
脚本不把凭据写入构建上下文或日志。
构建需要 `git`、Python 3，以及已配置的 Azure DevOps Git 访问权限。

```bash
# 1. 修改 Dockerfile 中的依赖；同步更新 verify.py 和相关 installer 的版本检查。
#    如果增加/替换内部 wheel，同时更新 packages.list。

# 2. 预览流程，不访问网络、不修改文件、不触发构建。
python3 docker/remax/update_image.py build --dry-run

# 3. 整理当前代码、提交/推送构建分支、触发一次流水线并等待发布。
python3 docker/remax/update_image.py build
```

`build` 会在忽略的 `tmp/remax-image-update/bootstrapping` 中管理专用构建
checkout，默认使用 `codex/verl-remax-image-oct23` 分支和流水线 2910。
它只在这个构建仓库提交、推送，不会自动提交或合并本地 verl 仓库。
构建源码包括当前工作区中选定的 tracked 文件内容，因此已跟踪文件的未提交修改
也会进入镜像；新增运行文件请先 `git add`，让 `stage_source.py` 能找到它们。

默认复用构建 2200290 的 wheel artifact。脚本比较该 artifact 对应提交的
`packages.list`；清单不同会自动改为重新下载。也可以明确指定：

```bash
# 强制重新下载 wheel 包。
python3 docker/remax/update_image.py build --fresh-packages

# 使用另一次提供 verl-context artifact 的构建。
python3 docker/remax/update_image.py build --reuse-context-build 2200290

# 只提交构建，不持续等待。
python3 docker/remax/update_image.py build --no-wait

# 断开终端或中断后，查询/继续等待同一个构建；不会重复触发。
python3 docker/remax/update_image.py status 2201665
python3 docker/remax/update_image.py status 2201665 --wait
```

完成后输出流水线链接、镜像 tag、不可变 digest，并保存 `<build-ID>.json`、
timeline 和 Docker 构建日志。默认等待上限为三小时；超时或 Ctrl-C 只停止本地等待，
远程构建继续。若提交请求的响应丢失，脚本保存 queue intent 并退出；先在流水线中
按提交 SHA 检查是否已经创建构建，再决定是否重试。再次执行 `build` 会创建新构建。
如需改变存储位置，全局参数放在子命令之前：

```bash
python3 docker/remax/update_image.py --state-dir /absolute/private-dir build
```

构建完成后，按照下方“Verify on a new pool”创建一个专用的 8-GPU 测试池，
指定本次输出的 digest。池 Ready 后可用同一个脚本验证：

```bash
python3 docker/remax/update_image.py verify \
  --pool verl-image-verify-unique \
  --image 'iridiumsdc.azurecr.io/boren/verl-remax@sha256:<实际 digest>'
```

`verify` 不创建、修改或暂停池。它先核对 pod 请求镜像和实际 imageID，要求 Ready
且零重启，然后用 `bpush` 同步当前代码，依次执行镜像内环境检查、8 卡 NCCL、
`quick_install.sh` 和当前 checkout 检查，最后确认 pod UID 未变且没有重启。
只对新建、未运行过依赖安装的专用测试池使用此命令；脚本无法证明此前的容器历史。
需要本机的 `zsh`/`bpush`、`brix`、`kubectl` 和相应集群权限。
默认集群为 `prod-westus2-cw-6`，namespace 为 `boren`；可通过
`--cluster`、`--namespace`、`--kubeconfig` 指定实际配置。
日志保存在 `tmp/remax-image-update/verify-<pool>-<pod-UID>/`。
成功后按脚本输出的命令暂停专用测试池，释放 GPU。

本地回归测试（不构建镜像、不访问 GPU）：

```bash
python3 -m unittest discover -s docker/remax -p 'test_update_image.py'
```

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

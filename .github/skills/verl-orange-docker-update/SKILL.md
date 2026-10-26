---
name: verl-orange-docker-update
description: 更新 verl/ReMax 的 Orange Docker 训练环境，通过 Azure DevOps 构建并推送镜像，再在全新 Brix GPU 容器上验证。用于更新 Dockerfile、重新制作训练镜像、发布镜像或验证新镜像；普通训练代码同步不需要触发镜像重建。
---

# verl Orange Docker 更新

将依赖变化制作成可追溯的镜像，完成构建、发布与远程验证。直接执行下面的操作流程；不新增流程脚本，也不依赖 `docker/remax/update_image.py` 作为入口。现有源码整理工具和 GPU 验证器可以复用。

## 先确定本次范围

- 定位当前 verl checkout、分支、工作区改动，以及当前任务已关联的 worktree。用户要求隔离时使用专用 worktree；不要覆盖其他任务的改动。
- 判断用户要的是更新依赖、仅重建、继续已有构建，还是仅验证已发布镜像。继续任务先找到已有 build ID 和收据，避免重复提交。
- 代码或 recipe 变化通常只需 `bpush`；Torch、vLLM、CUDA 或编译算子依赖变化才需要重建环境镜像。
- 构建、验证只作用于本次专用构建分支和测试池。是否提交、合并、推送 verl 仓库，以及是否替换正式训练池镜像，按用户本次授权执行。

## 读取实际配置

在当前 checkout 阅读这些文件，以文件内容为准，不把下列历史值当作永久配置：

| 文件 | 用途 |
|---|---|
| `docker/remax/Dockerfile` | 自包含安装逻辑、关键依赖版本与 CPU 构建检查 |
| `docker/remax/orange-pipeline.yaml` | wheel 准备、构建发布、artifact 复用参数 |
| `docker/remax/packages.list` | 内部构建包的精确清单 |
| `docker/remax/stage_source.py` | 选择并整理运行源码；检查当前文件选择与排除逻辑 |
| `docker/remax/image.rcall.py` | 专用镜像选择、TorchFlow 启动适配 |
| `docker/remax/verify.py`、`verify_nccl.py` | 环境、GPU 算子、Ray 与多卡通信检查 |
| `quick_install.sh`、`quick_install_linear_attention.sh`、`requirements_vllm.txt` | 普通节点安装、镜像复用与版本检查的一致性 |
| `docker/remax/README.md`、`VERIFICATION.md` | 操作说明、上次发布与验证证据 |

构建身份、网络或镜像策略不清楚时，查阅本地 Penny.wiki：

- `Orange-wiki/Personal-notes/antonsc/Docker-Images/Building-Docker-image.md`
- `Orange-wiki/9.-Platform-Infra/Transfer-rcall-image-from-OAI-to-iridium-ACR.md`

已验证的构建入口是 `https://dev.azure.com/project-argos/Mimco` 的流水线 **2910 / Kube Orange Devbox**，构建仓库为 `https://dev.azure.com/project-argos/Mimco/_git/bootstrapping`。先确认入口和模板仍存在。

## 1. 更新依赖并做本地检查

1. 在 Dockerfile 内修改安装命令和版本约束。保持自包含：不执行 `quick_install*.sh`，不读取外部 requirements 文件作为依赖安装入口。源码和 wheel 是构建输入，构建仍可能需要包索引与 apt 网络。
2. 同步 verifier 和相关 installer 的版本检查。Torch、CUDA、FlashAttention、causal-conv1d 等编译包要匹配 Python/Torch/CUDA ABI；不要只改版本断言。
3. wheel 增删或替换时同步 `packages.list`。镜像构建时通过 BuildKit 临时挂载 wheel，避免把整份安装包额外留在镜像层中。
4. 保留 Orange 兼容的基础镜像和启动入口。Ray 版本如需变化，应在镜像构建或集群启动前完成；不要替换已运行集群的 Ray。
5. 运行 `git diff --check`、相关 shell 的 `bash -n`、Python 语法检查和流水线 YAML 解析；同时检查 Dockerfile 的 shell/Python heredoc。构建期没有 GPU，只检查版本和编译扩展加载，不宣称 GPU 已通过。

有明确兼容性故障时，可以先在专用测试容器验证候选依赖，以减少昂贵的重建。记录这类临时安装；它不能替代最后对不可变镜像的无额外依赖安装验证。

## 2. 准备构建分支与材料

在 bootstrapping 的专用 checkout/分支中更新以下内容；不要修改共享流水线默认分支或覆盖共享镜像 tag。

```bash
# 在 verl checkout 内执行；将占位路径替换成实际专用构建 checkout。
python docker/remax/stage_source.py <bootstrapping>/images/verl-remax/source
cp docker/remax/Dockerfile <bootstrapping>/images/verl-remax/Dockerfile
cp docker/remax/orange-pipeline.yaml <bootstrapping>/.azuredevops/kube-orange-box.yaml
```

检查 staging 结果确实包含本次运行文件、排除了本机凭据和临时日志，并处理已删除文件的残留。新增文件是否会被整理工具发现，要检查当前实现。只 stage 预期构建路径；sparse checkout 下使用 `git add --sparse`。审阅 diff 后提交、正常 push，记录完整提交 SHA。遇到并发远端更新时先同步，禁止强推覆盖。

构建采用两类机器：

- `orange-builder`：通过 Orange 的身份、网络模板从 Blob 准备 wheel，生成 `verl-context` artifact 和 SHA256 清单。
- `iridium-builder`：下载 artifact、验证 checksum、拉基础镜像、构建并推送 ACR。

选择 wheel 来源：

- 清单或 wheel 内容改变：让 `reuseContextBuildId` 为空，重新准备 wheel。文件名相同而内容改变时也必须重新准备。
- wheel 未变：可指定一个已验证、仍可下载的 `verl-context` artifact 所属 build ID。历史构建 **2200290** 曾可复用，使用前确认可用性和适用性。
- 复用时保留流水线的 checksum 与精确 `packages.list` 比较；使用当前提交的源码和 Dockerfile，不能把旧 artifact 的源码一起复用。

本机 ACR 返回 401 时，优先使用 wiki 里的流水线授权身份。不要把本机 `az login` 当作这条远程路线的前置要求。凭据通过已有认证机制提供，禁止打印 token、完整 credential-helper 响应、用户启动配置或环境变量。

## 3. 提交一次构建并持续跟踪

使用现有 Azure DevOps 连接、CLI 或 REST API 提交一次流水线，参数包括：

- definition ID：确认后的流水线 ID；当前为 `2910`。
- sourceBranch：专用构建分支的完整 ref。
- sourceVersion：已推送的完整提交 SHA，不能用缩写。
- templateParameters：`reuseContextBuildId`，根据上一步决定。

马上保存 build ID、完整 source SHA、构建 URL、wheel artifact 来源和本次预期依赖变化。构建 URL 格式：

```text
https://dev.azure.com/project-argos/Mimco/_build/results?buildId=<BUILD_ID>
```

中断后查询或继续等待同一个 ID。排队或长时间构建不构成重提理由。提交响应不确定时，先按分支、SHA 和时间核对远端是否已有构建；确认不存在后才重试。采用有间隔的状态查询，只报告阶段变化和有意义的阻塞。

完成后核对 **Build and publish dedicated verl image** 任务、Docker 构建检查和 push 结果，从实际输出保存 tag 与 digest。流水线 `partiallySucceeded` 不能直接解释为成功或失败：区分 Docker 任务结果与注入扫描器等其他问题，记录真实原因。Docker 任务失败则定位失败步骤、修复后再提交一个新构建。

镜像必须用本次唯一 tag；当前命名模式为 `iridiumsdc.azurecr.io/boren/verl-remax:torch211-vllm024-<BUILD_ID>`，依赖版本变更时检查命名是否也应调整。验证和后续部署使用 `repository@sha256:...`。

## 4. 在全新专用 GPU 容器上验证

远程操作遵循可用的 `remote-development` skill：代码用 `bpush`，命令用 `brix ssh`，非代码文件用 `bbb`。

创建唯一命名的专用验证池，使用本次 digest 和项目 launch config。确认实际集群、配额、GPU 类型、数量和优先级；已有配额队列按用户要求保留，不挪用其他训练池。

当前 launch 方式需要：

- `VERL_DOCKER_IMAGE=<repository@sha256:...>`。
- `skip_rcall_user_config=true`，由 `image.rcall.py` 先加载正常用户配置再覆盖训练镜像，防止配置加载顺序把镜像覆盖回去。
- 保留基础设施镜像。TorchFlow 的 `/tmp` marker 要在启动 hook 中重建，防止默认安装流程覆盖 baked Torch。
- 用 `--no-deps` 注册挂载的 TorchFlow 源码，让 Ray 能加载 `trainflow.ray_plugin`；不要为解决模块缺失而重新安装整套 monorepo 依赖。

若复用本任务已有专用池来替换镜像，先核对 pool UID、当前镜像和所有权。使用带 UID/旧镜像前置检查的修改；确认控制器确实创建了新 pod。需要手动重建 pod 时对删除操作使用 UID 前置条件，保留 pool 配额。不要把原容器中的升级结果算成新镜像通过。

验证顺序很重要：

1. 保存 pool/pod UID、节点、实际 `imageID`、Ready 和 restartCount。按容器名字查找 `main`，不要依赖数组下标。确认实际镜像 digest 匹配，且新容器没有临时依赖安装历史。
2. `bpush <pool>` 完成后，先在镜像源码上运行 verifier，不能先运行可能修复环境的 installer：

   ```bash
   brix ssh <pool> -- 'bash -lc "cd /opt/verl-source && VERL_VERIFY_REPORT=/tmp/verl-image-verification.json python /root/code/verl/docker/remax/verify.py"'
   ```

3. 运行 `python -m torch.distributed.run --standalone --nproc-per-node=<实际GPU数> /root/code/verl/docker/remax/verify_nccl.py`，要求覆盖分配的全部 GPU。
4. 在 `/root/code/verl` 运行正常 `bash quick_install.sh`，核对它复用 baked 环境、没有重新安装主依赖，并把 editable 源码绑定到当前 checkout。再次运行 verifier，另存 report。
5. 最后重新核对 UID、实际 imageID 和 restartCount，确认检查没有跨重启混用证据。保存启动日志中跳过广泛安装的证据、版本、GPU、Ray worker 和 NCCL 输出。

启动失败时先查当前/previous 容器日志，再通过 Brix 查看 `/tmp/ray/session_latest/logs/` 的 `runtime_env_agent.err`、`raylet.err`、dashboard 日志。只提取相关行，原始日志私有保存。Ray 超时可能是插件 import 失败；短暂 SSH 拒绝也可能来自重启，不能未经诊断归因于登录失效。发现故障后做有依据的修复，避免无限重启或重复构建；若外部授权、配额或可访问性确实阻塞，记录未完成的验证边界。

## 5. 留证、清理与交付

- 更新 `docker/remax/README.md` 的最新镜像和启动方式，更新 `VERIFICATION.md` 的 build/source、digest、pod UID/节点、测试结果、实际警告与验证范围。
- 严格区分：候选依赖测试、Docker 构建检查、最终镜像 GPU 检查、正常 setup 后检查。环境 smoke/NCCL 通过不等于完成 ReMax optimizer step。
- 收据和日志保存后，暂停本次专用验证池，并确认 pod 已删除、GPU 释放；保留用户明确要求继续运行的池，不改变其他 workload。
- 用户要求提交/合并时，仅提交本次文件；合并前重新核对目标分支与工作区状态，保留其他改动。未经要求不推送 verl 仓库或替换正式训练镜像。
- 最终报告给出镜像 tag/digest、流水线链接、通过的验证、未覆盖项、测试池状态，以及已授权提交/合并的结果。

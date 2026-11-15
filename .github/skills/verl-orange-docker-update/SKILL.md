---
name: verl-orange-docker-update
description: 更新 verl/ReMax 的 Orange Docker 训练环境，通过 Azure DevOps 构建并推送镜像，再在全新 Brix GPU 容器上验证。用于更新 Dockerfile、重新制作训练镜像、发布镜像或验证新镜像；普通训练代码同步不需要触发镜像重建。
---

# verl Orange Docker 更新

将依赖变化制作成可追溯的镜像，再在全新专用 GPU 容器验证。
普通训练源码变化只需 `bpush`，不触发镜像重建。

## 定位当前构建源

当前 verl checkout 已删除 `docker/remax` 和根目录 quick installer。
不要运行历史路径、从旧提交恢复整套工具或把历史镜像 digest 当成默认值。
先定位本任务已有 build 收据、专用 bootstrapping checkout 和分支，读取
其中实际的 Dockerfile、wheel 清单、流水线、launch hook 和 verifier。
确认完整源 SHA、依赖版本、镜像 tag/digest、源码整理规则和验证范围。
若没有可用构建源，先记录缺失输入，继续可独立完成的本地检查。

历史构建入口为 `https://dev.azure.com/project-argos/Mimco` 的流水线
2910 / Kube Orange Devbox，仓库为
`https://dev.azure.com/project-argos/Mimco/_git/bootstrapping`。
核对当前入口和模板后再使用，不修改共享流水线默认分支或共享镜像 tag。
构建身份或网络问题可查阅本地 Penny.wiki：

- `Orange-wiki/Personal-notes/antonsc/Docker-Images/Building-Docker-image.md`
- `Orange-wiki/9.-Platform-Infra/Transfer-rcall-image-from-OAI-to-iridium-ACR.md`

## 修改与构建

- 使用本任务专用分支和验证池，保留其他工作区改动；提交、推送、合并和正式训练池镜像替换按用户授权范围执行。
- Dockerfile 内安装依赖并固定兼容的 Python/Torch/CUDA ABI；构建不依赖已删除的 quick installer。同步实际 wheel 清单和版本检查，不在运行中的 Ray 集群升级 Ray。
- 检查源码整理结果包含本次运行代码、无删除残留、无凭据和临时日志。运行 diff、shell/Python 语法、流水线 YAML 和 Docker heredoc 检查；CPU 构建检查不算 GPU 验证。
- 通过当前流水线准备 wheel artifact，再由构建机校验 checksum、构建并推送唯一 tag。仅当 wheel 清单和内容均未变化且 artifact 仍可用时复用 wheel；源码和 Dockerfile始终来自本次提交。
- 提交一次构建，保存 ID、完整分支/SHA、模板参数和 URL。中断后跟踪同一 ID；响应不确定时先查远端，避免重复提交。禁止强推覆盖并发更新。
- 核对 Docker build/push 任务结果与实际 digest。`partiallySucceeded` 要区分镜像任务与注入扫描器问题，保留真实警告。凭据使用已有认证，不打印 token 或用户启动配置。

## 全新 GPU 容器验证

按 `remote-development` 使用 `bpush`、`brix ssh`、`bbb`。读取当前构建源的
launch 配置，确认 image override 的加载顺序、正常网络/存储/基础设施配置、
TorchFlow marker 与 Ray plugin 注册方式。不要调用已删除的项目 hook。

1. 创建本次唯一专用验证池，使用本次不可变 digest 和已核验的 launch hook；保留请求的拓扑、优先级及配额。已有池替换镜像前核对所有权、UID、当前任务和冲突工作。
2. 按容器名字查找 `main`，保存 pool/pod UID、节点、实际 imageID、Ready、restartCount；确认新容器没有临时安装历史。
3. 从当前构建 checkout 运行 GPU verifier，先测 baked 源码，再绑定同步源码。覆盖关键算子、Ray worker 和实际分配的全部 GPU；多节点覆盖跨节点 NCCL。
4. 在每个 GPU 节点从 `/root/code/verl` 用 `uv pip install --system --no-deps --no-build-isolation -e . -e plugins/phimm -e plugins/qwen35_audio` 注册同步源码，再复测并另存报告。依赖缺失单独诊断，不能以广泛重装掩盖镜像缺陷。
5. 重新核对 UID、imageID 和 restartCount，不跨重启混用证据。通过 Blob 保存原始日志、版本、验证器输出和构建收据。

启动失败时查看容器日志和 `/tmp/ray/session_latest/logs/` 中的 runtime-env、
raylet、dashboard 错误。插件 import 失败可能表现为 Ray 超时；不能未经诊断
反复重建或替换集群依赖。临时候选依赖实验与最终不可变镜像验证分别记录。

## 交付与资源

在实际构建源或任务收据中更新 tag/digest、build/source、验证日志、pod UID、
警告和未覆盖项；不要重新创建已删除的 verl 工具路径。
环境和 NCCL 通过不代表训练收敛或模型质量通过。
证据持久保存后核对专用验证池无冲突工作，再暂停并验证 Pod/GPU 释放；
保留用户要求继续运行的资源。最终给出镜像、流水线链接、验证范围和资源状态。

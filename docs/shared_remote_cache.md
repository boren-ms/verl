# 共享模型缓存与精简模型目录

## 第一步：节点共享缓存

默认下载行为保持原样。仅在已验证的测试/新任务环境中设置：

```bash
export VERL_REMOTE_CACHE_NAMESPACE='<immutable-model-version-or-manifest-sha256>'
# Ray 2.54: allow the same namespace in job and nested ray.init runtime_env.
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
```

所有本地进程必须使用相同的namespace、HOME/cache_dir和可写本地文件系统。相同完整Azure URL与namespace映射到同一目录；文件锁串行首次下载，临时目录成功后原子发布。同一节点恢复可复用，同一进程切换namespace也能正确失效；命中缓存无需访问Azure。各节点独立保留一份。不要把它理解为跨节点共享磁盘或GPU显存共享。

namespace是内容不可变的约定，不自动查询远端etag或全目录版本。远端内容变化时必须更换namespace；同一个namespace内不要重写远端目录。推荐将已完成的模型manifest内容摘要用作namespace。训练checkpoint应使用已完成的不可变step路径，不能用不断覆盖的latest目录。

显式local_path维持旧语义。共享模式下资源不存在会报FileNotFoundError，允许下次调用重试。下载异常只清理本次staging，不发布完成标记；SIGKILL可能留下隔离的staging，不会被读取，也不会自动删除。异常published目录拒绝覆盖。锁等待上限1800秒。完成标记确认下载函数成功，不提供每次读取内容校验或抗断电fsync保证。

该变更不回收旧UUID目录，也不限制不同资源/namespace的总缓存量。不应在运行任务中删除任何缓存或改变其源代码。

## 第二步：精简目录

先在测试环境完成第一步真实下载/模型加载验证，再准备新的模型目录。以下命令会完整读取并复制保留的文件，应仅在独立空闲测试目标或离线环境执行；不要直接对运行节点缓存执行：

```bash
python -m recipe.phimm.prepare_model_directory \
  /path/to/source/qwen_hf /path/to/new/qwen_hf_compact \
  --exclude-dir baseline_split_68d42572
```

工具只排除显式指定且存在的顶层目录，其余文件全部保留；不改动源文件，不覆盖目的目录。拒绝源目录内部目的路径、保留树内符号链接、缺失/被排除的索引权重分片、JSON中指向排除目录的直接相对路径。复制期间检查源文件是否变化，记录每个文件的SHA256和总字节，成功后原子发布新目录及verl-model-manifest.json。

manifest的content_sha256是按路径/大小/文件摘要计算的稳定内容标识。它适合标识新模型资源；整个manifest还含本地source路径，不应直接把整个JSON文件哈希当可迁移的内容标识。

这些检查不能证明自定义Python代码或任意配置不会动态引用归档目录。真实发布之前必须在同一模型版本、同一环境验证tokenizer、processor、权重完整加载与固定样本输出；本工具不上传、不更改YAML、不修改原blob路径。

## 独立workspace验证

```bash
python -m pytest -q tests/recipe/phimm/utils/test_remote_cache.py tests/recipe/phimm/test_prepare_model_directory.py tests/recipe/phimm/utils/test_checkpoint_retention.py
```

17项测试覆盖模块集成、8个独立进程、恢复复用、版本失效、网络失败重试、强杀半成品隔离、精简目录文件哈希与源文件不变、索引/引用保护，以及双节点/共享文件系统的本地保留。blobfile与bbb在本地集成测试中替换成可控IO，不能代替Remote Azure集成。

本次实现与所有正式验证位于独立workspace：
`/home/boren/.codex/worktrees/shared-model-cache/verl`。
原workspace与L2_repo受控实验没有应用这些改动。独立双节点实验已经完成；结果见下文。临时调查脚本与证据留在独立workspace，不作为运行依赖提交。

## 本地checkpoint上限

独立workspace的L2_pad配方现设置trainer.max_actor_ckpt_to_keep=2，Hydra组合验证其他有效字段不变。这项配置仅在部署后的新启动生效，当前运行进程仍使用None。候选FSDP代码已改为所有rank参与本地旋转，同一文件系统以目录外文件锁串行清理，上传完成并经过全局barrier后才开始删除。8进程模拟双节点/共享磁盘以及未设上限三项测试通过，真实双节点训练也已验证。历史restart目录/下载缓存不受本次manager列表管理。

## 真实双节点实验结果（2026-10-23 UTC）

独立配方`remax_disk_cache_validation`从原始L2派生，仅改变独立experiment_name、resume_mode=disable、total_training_steps=50和max_actor_ckpt_to_keep=2；保留原数据、训练设置和save_freq=10。通过环境变量启用共享缓存，未启用目录精简。Ray2.54.0/protobuf5.29.5保持原生。

两节点真实Azure验证各8进程两轮16calls，均只下载一份38.69GB模型，第二轮约1.2s；随后真实模型加载、validation、训练及五次checkpoint保存成功，任务Ray SUCCEEDED。训练期间每节点始终一份共享模型，没有新增UUID模型副本。

step30/40/50保存分别将本地actor旋转为20/30、30/40、40/50；终态两节点仅40/50 actor。远端10/20/30/40/50的全部16rank model/optim/extra_state分片与data.pt均完整，latest=50。旧global_step父目录和小文件可以保留，不属于actor保留数量。

按各自step1文件系统占用归零，同为完整optimizer step49，原L2 head增长100.44GiB，新实验head45.06GiB、worker49.86GiB。step50完整optimizer指标未取得，不能补成实测点；最终checkpoint50上传后head增长45.07GiB。原L2来自W&B报告文件系统、新实验来自pod overlay，不代表host nodefs/imagefs，也不能将不同初始占用归因修复。该实验支持共享缓存和本地保留有效，不证明吞吐提升或数值输出完全等价。

实验专用pool已暂停并核实零pods；远端checkpoint未删除。精简目录工具仍只有本地测试，发布精简模型前须另行完成真实加载/固定样本输出对照。

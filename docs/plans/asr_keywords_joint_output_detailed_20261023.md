# LibriSpeech 转写与关键词联合输出：详细实验计划 v3

日期：2026-10-23。状态：方案、源代码调研和本地数据审计完成；功能、recipe 与模型训练均未实施。

开发隔离继续使用 `boren/asr-transcription-keywords`，worktree 为 `/home/boren/.codex/worktrees/asr-transcription-keywords/verl`。本方案按用户最新决定收敛为单段联合输出，数据选择细化自[第一版方案](asr_transcription_keywords_rl_20261023.md)。

图解：[`paper_notes/asr_keywords_joint_output_architecture.html`](../../paper_notes/asr_keywords_joint_output_architecture.html)。
可复现证据：[`reports/asr_keywords_plan_20261023/local_data_audit.json`](../../reports/asr_keywords_plan_20261023/local_data_audit.json) 与同目录 `audit_local_data.py`。

## 1. 本次实验定位与范围

**主要假设：** 在现有转写任务上增加关键词输出协议，结合完整 GT rollout 与独立列表 reward，使模型在保留 transcription 质量的同时输出当前音频中的关键词。

本次实验的主要定性为“关键词联合输出”。每条音频独立推理，输入保留原 prompt 并追加关键词输出要求；输出保留正文、语言/模式头，追加 `<keywords>` 块。验收聚焦正文质量、列表质量、协议与训练信号正确性。

用户此前提到的 variation 已确认指 biasing。跨段预测关键词反馈、history 策略、连续识别与相关对照全部移至[后续计划备忘](asr_keywords_biasing_followup_20261023.md)，本轮不实施、不调参、不作为验收条件。

## 2. 已找到且已核验的数据

### 2.1 本地可用的数据与音频

| 数据 | 已读路径 | 已核验规模 | 用途 |
|---|---|---|---|
| LibriSpeech 原始训练 TSV | `/home/boren/data/LibriSpeech/asr_train_transcribe.tsv` | 280,393 行、5,466 个 speaker/chapter、0 重复 id；132,266 条音频本地存在 | 核验训练源，构造 train/dev/internal-test 和训练内 probe |
| test-clean 原始 keyword reference | `/home/boren/data/librispeech_biasing/ref/test-clean.biasing_100.jsonl` | 2,620 行；2,620 条本地音频，5.4035 h | 独立正文与 keywords 测试 |
| test-other 原始 keyword reference | `/home/boren/data/librispeech_biasing/ref/test-other.biasing_100.jsonl` | 2,939 行；2,939 条本地音频，5.3415 h | 独立正文与 keywords 测试 |
| 固定常用词表 | `/home/boren/data/librispeech_biasing/words/common_words_5k.txt` | 由审计脚本记录实际集合大小 | 确定 rare-keyword 标注协议 |

两个测试集的本地 flac 均通过 soundfile header 检查，为 16 kHz mono，无 header 读取错误。此检查不等价于逐帧解码或人工音质检查。

训练 TSV 与官方测试：id 重合 0、speaker 重合 0，但归一化 reference 文本有 7 条训练记录与测试文本相同。实施数据准备时把这 7 条从训练源排除，并给出同源文本敏感性统计，避免把“id 不重合”误写成完全无重复。训练源并非全部音频本地就绪，GPU 全量训练前必须完成远端路径/音频可读性检查。

### 2.2 已定位、尚未读取的云端候选

```text
az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe_rare_keywords_nonempty.jsonl
az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe_rare_keywords_nonempty_sample10k_seed42.jsonl
az://orngwus2cresco/data/boren/data/librispeech_biasing/ref/test-clean.biasing_100.think.jsonl
az://orngwus2cresco/data/boren/data/librispeech_biasing/ref/test-other.biasing_100.think.jsonl
```

训练配置：`recipe/phimm/config/data/train_data/ls_rare_nonempty_verb_langhint.yaml`；验证配置：`recipe/phimm/config/data/val_data/ls_kw_verb_langhint.yaml`。云端账号当前缺少本机可用读取凭据，本轮没有重试获取密钥或启动 GPU 节点。文件规模、字段和与本地文件的等同性尚未验证。

本方案可基于已有本地 raw TSV 和固定词表继续做协议/数据准备设计，不依赖声称云端文件已经可读。若后续采用云端现有 keywords，先与下面明确的规则逐行核验；不同规则应独立命名并报告，不能混为同一监督目标。

### 2.3 已验证的关键词定义

原始测试 JSONL 的 `text` 是正文；`ground_truth` 是关键词 list；`distractors` 是含干扰词的候选集。后者不作为目标。

在本地全部 5,559 行上，下面的集合规则与已有关键词标注完全一致：

```text
tokens = lowercase English words, preserving apostrophes and hyphens
target_keywords = unique(tokens(reference)) - fixed_common_words_5k
```

审计使用 `re.findall(r"[\w'\-]+", text.lower())`。这证明其在当前本地 reference 上一致，不宣称适用于所有语言或任意带标点的新文本。训练前确认数字、连字符、缩写、Unicode apostrophe 的处理，冻结 normalization 版本与词表 SHA。

直接与 `all_rare_words.txt` 取交集会使 test-clean 108 行、test-other 161 行不一致。因此**首轮建议采用固定 common5k 排除规则**，不给 rare words 再添加外部词典存在性限制，也不采用 `rare_ratio=0.5` 的随机裁剪。

GT 列表使用参考转写出现顺序和表面形式；评分使用 canonical set。对当前全小写训练/测试源，两者差异较小。空列表有真实监督，不能只用 nonempty 文件训练后忽略空输出行为。

## 3. 数据划分与“训练侧测试 / 独立测试”的边界

先将完整原始训练源按 speaker 做稳定划分，再过滤 nonempty 或抽样，确保不同阶段使用同一 split。建议 SHA256(`ls-keywords-v1:<speaker>`) %100：0..4 dev，5..9 internal-test，其他 train。该规则已审计出各 fold 的规模，详见 JSON；排除 7 条重复文本后数字会变化。

- **train：** 用于 RL 更新，可抽一份固定小样本作为 train-probe。probe 展示拟合与 GT 路径是否工作，不能证明泛化。
- **dev：** 用于选 checkpoint、reward 权重与输出长度预算；与 train 按 speaker 隔离。
- **internal-test：** 来自原始训练 corpus 的 held-out speaker，参数冻结后测试，不参与调参。
- **official test-clean / test-other：** 两个 split 分别汇报，不用于选阈值或 checkpoint；同一批音频同时报告正文和关键词指标。已有实验可能接触过官方测试，报告中如实注明其为现有 benchmark，而不声称全新盲测。

speaker 隔离不能证明 book 隔离。若 SPEAKERS/CHAPTERS metadata 可用，额外检查共享 book；配对统计可按 chapter 聚类，减少同一章内样本相关性的影响。

该划分在原始训练 TSV 上的实际审计（尚未排除重复文本、尚未生成正式训练清单）：

| Fold | 行数 | speaker 数 | 非空关键词 | 空关键词 | 本地音频存在 |
|---|---:|---:|---:|---:|---:|
| train | 254,249 | 2,119 | 232,066 | 22,183 | 120,039 |
| dev | 13,297 | 110 | 12,218 | 1,079 | 6,096 |
| internal-test | 12,847 | 109 | 11,929 | 918 | 6,131 |

上述 7 条与测试 reference 相同的记录都在 train fold；dev/internal-test 无此归一化文本重合。原始训练源共 24,180 条空关键词，足够设计真实空目标监督。计数属于 fixed-common5k 新派生规则，不等同于未读取的云端 nonempty manifest。

训练源保留真实空样本，报告自然比例。若小试验为了足够空样本而分层抽样，固定 seed 并记录比例；不将测试集空样本搬进训练。可以先从本地有音频的 train fold 中选 10k 做 pilot，这与云端现有 sample10k 不是同一个产物，必须使用新名字。

将来准备的清单字段：

```text
id, split, speaker_id, chapter_id,
audio_path, reference_text, target_keywords, keyword_annotation_valid,
annotation_policy, annotation_policy_sha, source_manifest_sha
```

reference 和 target_keywords 只给数据构建、GT 示范和评分；正常生成及评估的 prompt 不包含它们或 distractors。

## 4. 联合输出与 prompt contract

输出格式沿用用户确认的 `<keywords>` 独占行 + 下一行英文逗号列表；空目标保留空行。正文和语言/VERBATIM 头保留原样。

原 prompt 追加以下输出要求：

```text
Also output the rare words that occur in this audio. After the complete
transcription, write <keywords> on a new line, then a comma-separated list
on the next line. If there are none, leave that line empty.
```

训练监督严格由固定常用词表生成。prompt 中不列出 5,000 词表，也不暴露当前 GT 词；“rare”属于监督规则下学习的任务概念，是否精确泛化由 held-out F1 检验。

本轮不添加输入端关键词候选，不修改现有 biasing 提示格式，也不引入动态 prompt history。所有实验臂按单条音频独立生成。

## 5. 模块级改动清单（名称为拟议接口，不是已实现 API）

| 文件 / 位置 | 拟议改动 | 必须保持的不变量 / 主要验证 |
|---|---|---|
| `recipe/phimm/data/prompts.py:get_task_prompt` | 可选 `output_keywords`；组合原 task、原语言 hint 与新输出指令 | 无开关时原 prompt 不变；audio 占位符不重复；不注入当前 GT |
| 同文件 `get_task_output` | 接收 canonical keywords，调用旧 formatter 后追加块 | prefix+completion 可解析；列表按 reference 排序；纯 ground_truth 不被污染 |
| `recipe/phimm/data/dataset.py:add_task_info` | 构建 target_keywords、gt_output 和标注状态，显式保留 annotation_policy | 不接受“缺标注=空”；清理/切分完成后建目标 |
| 同文件 `verl_format_ds` / `clean_tagged_text` | 保留纯文本 GT、完整 gt_output 与目标字段；避免空覆盖 | raw keyword-list 的 ground_truth 与 RL string ground_truth 显式映射 |
| `recipe/phimm/reward/asr_response.py` | 结构化 envelope parser + 保持旧 segment API；传入显式协议模式 | keywords 不进入 WER；坏尾部不能被隐藏；缺头≠空块 |
| `recipe/phimm/reward/asr_measure.py` | `kw_list_precision/recall/f1/exact`、block fmt、duplicate、in-transcript 指标 | 预测不按 GT 先过滤；空/缺失/错误明确计分 |
| `recipe/phimm/reward/asr_vote.py` | 接入新列表指标，离线校准权重与 reduction | baseline/sample 指标一致；GT 对缺块 greedy 有正向信号 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | GT 仅 train sampling；检查完整长度和 EOS；显式分组 | greedy/val/eval 0 GT；不能静默截断完整列表 |
| `verl/workers/rollout/gt_rollout.py`、`verl/trainer/ppo/rollout_log_probs.py` | 原有 teacher-forced scoring/温度对齐原则不变，补联动验证 | GT 全部 token 有有限 log-prob；mask 重排后仍对齐 |
| `verl/trainer/ppo/ray_trainer.py` | 如需要，增加 rollout purpose/group uid，分离 baseline 与 sample | baseline 数量及 group 映射须复核；不要让 GT 进入基线 |
| `recipe/phimm/main_asr_gen.py` | 保留 raw_response、parsed_transcription、predicted_keywords 和协议状态 | 原 eval 仍只用正文；日志能区分 label/hint/prediction |
| 新数据/实验 YAML | 独立 train/dev/internal/official recipes、联合输出开关、长度与预算 | 新 cache key、experiment name、输出目录，不覆盖旧 recipe |
| `tests/recipe/phimm/...` 和 rollout/trainer tests | 协议 round-trip、空目标、GT/greedy 隔离、字段与 mask 对齐 | 旧任务回归不变；新机制有实际训练/推理信号 |

是否提取共用 helper 由实现复杂度决定；不为了本实验重构整个训练框架。现有独立样本评估路径增加结构化结果记录即可。

## 6. GT rollout 与 reward 的联动检查

初始 n=8，每个 prompt 1 GT+7 sampled。完整 `gt_output` = 格式化 transcription + keywords + 合法终止。rollout 的现有 response_length 切片需要在新任务改为提前长度检查/过滤，不容许尾部缺失。

重用 actor 温度对齐；GT 行仍是人为示范轨迹，不是 on-policy sample。输出格式奖励与列表 F1 为 GT 提供相对早期 greedy 的正向信号，但须检查 KL、advantage、dedup/group filtering 是否使该信号消失。

先用同一批小样例离线验证奖励排序：

1. 正文正确、列表正确得最高；正文正确但缺列表有明确较低奖励。
2. 胡列词的 precision 降低；空目标乱列词被惩罚。
3. 将插入错误移到关键词尾部不能得到高奖励。
4. GT 在 train sampling 中有有效 token/advantage；ReMax baseline 和所有评估均无 GT。

默认保留已有正文奖励语义；新增列表 F1 和 block fmt 项，先不固定最终 beta。权重用 dev/离线 sanity cases 校准后冻结，并报告与旧 reward 相比的尺度。

## 7. 本轮实验矩阵：单条音频独立生成

| Arm | 模型 | Prompt | 主要目的 |
|---|---|---|---|
| A0 | 原 base | 原转写 prompt | 正文 ASR 基线；关键词列表指标不适用 |
| A1 | 原 base | 新联合输出 prompt | 测量只改 prompt 的格式遵循、抽词能力与正文代价 |
| A2 | 新 RL | 与 A1 相同的新 prompt | 验证 RL 后联合输出能力，对比 A1 的训练收益及 A0 的正文质量 |

三个臂都不接收关键词 hints 或历史信息。正式测试无 GT rollout；关键词 reference 只用于事后评分。A0 没有要求输出列表，不以其缺块率作公平的格式能力比较。

可选的 GT rollout 消融留在同一单段联合输出任务内：只有 pilot 需要解释 GT 示范贡献、且资源允许时，增加相同预算的无 GT 训练对照；它不是本轮必需项。

## 8. 指标与可靠性判据

- 正文：各数据集分别报告 p_err/WER/CER；从解析后的 transcription 评分，关键词尾部不得计入正文。
- 关键词：micro/macro precision、recall、F1、集合 exact；报告 normalization、空集合的计分规则和有效样本数。
- 协议：有效块率、缺块、坏块、重复关键词、截断率；合法空块与缺块分别统计。
- 空目标：假阳性率；非空目标：召回及漏词案例。预测列表不能先按 GT 过滤后再评分。
- 成本与训练信号：输出 token、推理时延；GT/sample 分开的 reward、advantage、KL、有效 token 和长度。

train-probe 只用于检查拟合。dev 选 checkpoint、reward 权重和长度预算；冻结后跑 internal-test 与 official test-clean/test-other，逐集汇报。使用同音频配对的正文误差差值，按 chapter bootstrap 给 95% CI；资源不足以多 seed 时注明不确定性。

正文非劣界（例如 WER 最多 +0.1 个百分点）仍是待讨论提议；关键词 F1 和格式达标阈值也需在正式训练前约定，不能看官方测试后设门槛。只有格式正确而关键词错误，不能视为联合输出成功。

## 9. 实施里程碑与进入条件

| 阶段 | 工作 | 进入下一阶段的条件 |
|---|---|---|
| P0（本轮） | 收敛方案、数据审计与流程图 | 明确仅研究联合输出；无功能实现或训练 |
| P1 数据协议 | 冻结 common5k、speaker split、排除 7 条重复正文、音频验证、生成清单/hash | train/dev/internal/official 隔离；真实空标注；字段映射正确 |
| P2 Prompt/GT/parser | 原任务 opt-in、完整 formatter、严格 parser | prefix+GT round-trip；正文不含列表；长度检查；旧任务兼容 |
| P3 Reward/GT 联动 | 列表 reward、baseline 隔离、分组与 log-prob/mask 检查 | 奖励排序合理；train 每组恰好 1/n GT；greedy/评估 0 GT |
| P4 单段 pilot | 固定 10k 或更小训练；dev/probe 监控 | 列表学习信号有效、正文质量可接受、无长度/advantage 异常 |
| P5 冻结正式测试 | 固定 checkpoint、prompt、评分协议，运行 A0/A1/A2 | 逐集报告正文、列表、协议指标、CI、错误案例与限制 |

下一轮是否开展 biasing 另行讨论，不作为本轮 P5 完成的依赖。

## 10. 关键测试案例

- 旧/新格式、空目标、缺标注、CSV 引号与逗号、保留字、语言头和 prefix round-trip。
- 纯文本 ground_truth 与完整 gt_output 分离；本地 keyword JSON 的 ground_truth list 显式映射，不能覆盖 RL 正文 GT。
- 超长 GT 在 rollout 前显式失败或过滤，不得静默截掉关键词尾部。
- GT 仅用于 train sampling；greedy baseline、validation、standalone eval 0 GT；shard/pad/dedup/reorder 后 token、mask、group uid 对齐。
- 缺块不能当作合法空块；错误正文不能通过移动到列表中规避正文惩罚；格式正确但胡列关键词仍降低 precision。
- 原任务关闭新开关时 prompt/GT/parser 行为兼容；评估保留 raw_response、parsed_transcription、predicted_keywords、协议状态。

后续实现仍限定在独立 worktree 与远程分支 `boren/asr-transcription-keywords`。当前仅更新文档和图解；未修改模型功能、recipe，未启动训练。

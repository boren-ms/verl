# LibriSpeech 联合关键词输出与连续 biasing：详细设计 v2

日期：2026-10-23。状态：方案、源代码调研和本地数据审计完成；功能、recipe 与模型训练均未实施。

开发隔离继续使用 `boren/asr-transcription-keywords`，worktree 为 `/home/boren/.codex/worktrees/asr-transcription-keywords/verl`。本方案扩展并在数据选择方面细化[第一版方案](asr_transcription_keywords_rl_20261023.md)。

图解：[`paper_notes/asr_keywords_continuous_architecture.html`](../../paper_notes/asr_keywords_continuous_architecture.html)。
可复现证据：[`reports/asr_keywords_plan_20261023/local_data_audit.json`](../../reports/asr_keywords_plan_20261023/local_data_audit.json) 与同目录 `audit_local_data.py`。

## 1. 本次新增的两个假设

**H1：** 使用完整 GT rollout 加独立列表 reward，可以在保留 transcription 质量的同时学会输出当前音频中的 rare keywords。

**H2：** 将前一段的预测关键词按模型已有 biasing 提示格式加入下一段 prompt，可以改善重复关键词的后续识别，同时不明显增加无关词插入。用户已确认这里的策略是 prompt biasing。

H1 和 H2 分开验收。输出列表正确并不自动意味着下一段 WER 会下降；前一段预测不到的词不会通过这个机制首次发现。第一版只提供跨段文本提示，不设计音频编码器状态或 KV cache 的持续复用。

## 2. 已找到且已核验的数据

### 2.1 本地可用的数据与音频

| 数据 | 已读路径 | 已核验规模 | 用途 |
|---|---|---|---|
| LibriSpeech 原始训练 TSV | `/home/boren/data/LibriSpeech/asr_train_transcribe.tsv` | 280,393 行、5,466 个 speaker/chapter、0 重复 id；132,266 条音频本地存在 | 核验训练源，构造 train/dev/internal-test 和训练内 probe |
| test-clean 原始 keyword reference | `/home/boren/data/librispeech_biasing/ref/test-clean.biasing_100.jsonl` | 2,620 行；2,620 条本地音频，5.4035 h | 独立正文与 keywords 测试；87 个有序 chapter |
| test-other 原始 keyword reference | `/home/boren/data/librispeech_biasing/ref/test-other.biasing_100.jsonl` | 2,939 行；2,939 条本地音频，5.3415 h | 独立正文与 keywords 测试；90 个有序 chapter |
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
- **dev：** 用于选 checkpoint、reward 权重、history TTL 和 token budget；与 train 按 speaker 隔离。
- **internal-test：** 来自原始训练 corpus 的 held-out speaker，参数冻结后测试，不参与调参。
- **official test-clean / test-other：** 两个 split 分别汇报，不用于选阈值或 checkpoint；同一批音频同时报告正文和关键词指标。已有实验可能接触过官方测试，报告中如实注明其为现有 benchmark，而不声称全新盲测。

speaker 隔离不能证明 book 隔离。若 SPEAKERS/CHAPTERS metadata 可用，额外检查共享 book；连续识别的 bootstrap 单位为 chapter/session。

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
id, split, speaker_id, chapter_id, session_id, segment_index,
audio_path, reference_text, target_keywords, keyword_annotation_valid,
annotation_policy, annotation_policy_sha, source_manifest_sha
```

reference 和 target_keywords 只给数据构建/评分，连续推理 prompt builder 不接收它们。

## 4. 联合输出与 prompt contract

输出格式沿用用户确认的 `<keywords>` 独占行 + 下一行英文逗号列表；空目标保留空行。正文和语言/VERBATIM 头保留原样。

模型没有历史关键词时，原 prompt 加：

```text
Also output the rare words that occur in this audio. After the complete
transcription, write <keywords> on a new line, then a comma-separated list
on the next line. If there are none, leave that line empty.
```

训练监督严格由固定常用词表生成。prompt 中不列出 5,000 词表，也不暴露当前 GT 词；“rare”属于监督规则下学习的任务概念，是否精确泛化由 held-out F1 检验。

有历史 bias 时保留原有任务和语言提示，再加入**已存在的提示措辞与星号包裹候选词格式**：

```text
Pay extra attention to the following phrases/words. *agency*, *apostles*, *prophets*
```

对应现有实现为 `BIASING_PROMPTS`、`biasing.tag_pieces`、`ls_bias_dataset` / `bias_sampling`。需核对 checkpoint 实际使用过的分隔方式；first pass 优先保持模型熟悉的格式。输入候选词的星号不意味着输出正文也应加星号。追加“仅转写实际听到的内容”的约束可在 dev 上验证，不修改关键词输出头。

不能直接切换 `task=biasing` 后丢掉 `lang_asr_verb` 的语言/模式格式。不能直接调用现有随机 `bias_sampling` 来实现在线历史，它会随机构造 context、修改 text/keywords，随后还可能被 add_task_info 覆盖 prompt。应新增确定性的 prompt composition helper，供训练/独立推理共用。

## 5. 模块级改动清单（名称为拟议接口，不是已实现 API）

| 文件 / 位置 | 拟议改动 | 必须保持的不变量 / 主要验证 |
|---|---|---|
| `recipe/phimm/data/prompts.py:get_task_prompt` | 可选 `output_keywords` 与 `bias_keywords`；组合原 task、原语言 hint、旧 biasing 格式、新输出指令 | 无开关时原 prompt 不变；audio 占位符不重复；不注入当前 GT |
| 同文件 `get_task_output` | 接收 canonical keywords，调用旧 formatter 后追加块 | prefix+completion 可解析；列表按 reference 排序；纯 ground_truth 不被污染 |
| `recipe/phimm/data/dataset.py:add_task_info` | 构建 target_keywords、gt_output 和标注状态，显式保留 annotation_policy | 不接受“缺标注=空”；清理/切分完成后建目标 |
| 同文件 `verl_format_ds` / `clean_tagged_text` | 保留纯文本 GT、完整 gt_output、历史输入与目标字段；避免空覆盖 | raw keyword-list 的 ground_truth 与 RL string ground_truth 显式映射 |
| `recipe/phimm/data/rl_dataset.py` | 新 metadata schema；抽取单样本 prompt tokenization 为可复用方法（如需要） | 在线更新 bias 后必须重新 tokenize/processor；不能只改字符串却复用旧 input_ids |
| `recipe/phimm/reward/asr_response.py` | 结构化 envelope parser + 保持旧 segment API；传入显式协议模式 | keywords 不进入 WER；坏尾部不能被隐藏；缺头≠空块 |
| `recipe/phimm/reward/asr_measure.py` | `kw_list_precision/recall/f1/exact`、block fmt、duplicate、in-transcript 指标 | 预测不按 GT 先过滤；空/缺失/错误明确计分 |
| `recipe/phimm/reward/asr_vote.py` | 接入新列表指标，离线校准权重与 reduction | baseline/sample 指标一致；GT 对缺块 greedy 有正向信号 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | GT 仅 train sampling；检查完整长度和 EOS；显式分组 | greedy/val/eval 0 GT；不能静默截断完整列表 |
| `verl/workers/rollout/gt_rollout.py`、`verl/trainer/ppo/rollout_log_probs.py` | 原有 teacher-forced scoring/温度对齐原则不变，补联动验证 | GT 全部 token 有有限 log-prob；mask 重排后仍对齐 |
| `verl/trainer/ppo/ray_trainer.py` | 如需要，增加 rollout purpose/group uid，分离 baseline 与 sample | baseline 数量及 group 映射须复核；不要让 GT 进入基线 |
| `recipe/phimm/main_asr_gen.py` | 保留 raw_response、parsed_transcription、predicted_keywords 和协议状态 | 原 eval 仍只用正文；日志能区分 label/hint/prediction |
| `recipe/phimm/main_long_eval_asr.py:generate_segments` | 保留现有独立批解码路径；新入口采用按 session 推进的调度器 | 不能在所有段独立生成后再“补 history” |
| 拟新增 `recipe/phimm/main_asr_keyword_stream_eval.py` | 独立连续评估入口，复用 build_model/worker_group/评分工具 | 一会话最多一段 in flight，跨会话并行 |
| 拟新增 `recipe/phimm/data/keyword_history.py` | 无模型依赖的历史更新、预算、重置、序列化策略 | 函数不接收当前/未来 reference；session 状态互不共享 |
| 新数据/实验 YAML | 独立 train/dev/internal/official recipes、输出/反馈开关、长度与预算 | 新 cache key、experiment name、输出目录，不覆盖旧 recipe |
| `tests/recipe/phimm/...` 和 rollout/trainer tests | 协议 round-trip、空目标、反馈因果性、GT/greedy 隔离、恢复一致性 | 旧任务回归不变；新机制有实际训练/推理信号 |

是否提取共用 helper 由实现复杂度决定；不为了本实验重构整个训练框架。新 history 层先限 eval，不在 PPO dataloader 中引入可变会话状态。

## 6. GT rollout 与 reward 的联动检查

初始 n=8，每个 prompt 1 GT+7 sampled。完整 `gt_output` = 格式化 transcription + keywords + 合法终止。rollout 的现有 response_length 切片需要在新任务改为提前长度检查/过滤，不容许尾部缺失。

重用 actor 温度对齐；GT 行仍是人为示范轨迹，不是 on-policy sample。输出格式奖励与列表 F1 为 GT 提供相对早期 greedy 的正向信号，但须检查 KL、advantage、dedup/group filtering 是否使该信号消失。

先用同一批小样例离线验证奖励排序：

1. 正文正确、列表正确得最高；正文正确但缺列表有明确较低奖励。
2. 胡列词的 precision 降低；空目标乱列词被惩罚。
3. 将插入错误移到关键词尾部不能得到高奖励。
4. GT 在 train sampling 中有有效 token/advantage；ReMax baseline 和所有评估均无 GT。

默认保留已有正文奖励语义；新增列表 F1 和 block fmt 项，先不固定最终 beta。权重用 dev/离线 sanity cases 校准后冻结，并报告与旧 reward 相比的尺度。

## 7. 连续识别状态机与因果边界

对一个 session，记第 t 段音频为 A_t、预测正文为 T_t、预测列表为 K_t。

```text
H_0 = []
P_t = compose(original_prompt, bias_keywords=H_(t-1), output_keywords=true)
(T_t, K_t, status_t) = parse(decode(A_t, P_t))
H_t = update(H_(t-1), K_t, T_t, status_t)
```

**第一版 prev_only：** 若当前输出块有效，保留 K_t 中非空、去重、在 T_t 中出现且在预算内的词，作为 H_t；合法空块则 H_t=[]；坏块/超时/截断则清空，不把坏原文传给下一段。首次 A_0 不带 bias。

这允许使用预测正文做一致性检查，但不能证明词在音频中真实出现：模型若幻觉正文和列表同时一致，仍会污染 history。因此必须同时测错误反馈和无关插入。

**第二版 rolling3（可选）：** 前 3 段的合格列表去重，按最近观测排序，TTL 按段数过期。初始提议最多 32 个词、128 个 hint token，均在 dev 校准。没有校准过的置信度时不引入“模型很确定”的假分数；若使用 token log-prob，单独定义其聚合/标定方法。

rolling3 需限制由旧 bias 自我复制造成的无限续期；记录 first_seen/last_seen/source_step 和复用次数，必要时硬限制生命周期。对照实验固定预算，不能只让 feedback arm 拥有更大的上下文。

**因果规则：** H_(t-1) 只来自该 arm、该模型、该 session 已完成的过去预测。任何当前 GT、未来 GT、GT rollout completion、其他实验 arm 的预测都不得更新真实反馈状态。当前段发现新词只影响 t+1，不能事后改善 t 的输出。

**调度：** session 内串行；每轮从多个 session 各取一个下一段组成 batch。收到本轮 decode 并解析、更新状态后，才能构造同 session 下一段输入。tokenizer/processor 在新 prompt 下重新运行，音频特征可复用到符合模型接口的程度，不复用过时的 prompt IDs。

**中断恢复：** 原子保存 session_id、next_segment_index、history、上一步 id、checkpoint SHA、prompt/protocol/history config hash 和输出偏移。相同配置恢复后与不中断 greedy run 的结果一致。padding 的复制行不更新真实 session；重复响应按请求 id 幂等处理；不同会话结束立即释放状态。

**评分：** 每段只将 T_t 交给 WER/CER，session 正文按原顺序聚合；keywords 独立计分。跨 chapter/speaker/split/实验模式均重置 H。

## 8. LibriSpeech 有序会话的可用性与上限

按 `(split, speaker_id, chapter_id)` 建 session，再按 utterance index **整数**排序。本地 177 个 chapter、5,382 对相邻句全部编号连续，不按 JSONL 文件当前行顺序推断时间。

已有真实重复词例子：

```text
test-clean 2830-3980-0010 -> 2830-3980-0011：apostles
test-other 3764-168670-0022 -> 3764-168670-0023：fauchelevent
```

| split | 相邻句对 | 有重复关键词的句对 | 后续句关键词条目 | 上一句可覆盖 | 前三句可覆盖 |
|---|---:|---:|---:|---:|---:|
| test-clean | 2,533 | 169 | 5,449 | 192（3.52%） | 367（6.74%） |
| test-other | 2,849 | 167 | 5,073 | 193（3.80%） | 415（8.18%） |
| 合计，仅作机会诊断 | 5,382 | 336 | 10,522 | 385（3.66%） | 782（7.43%） |

这里的覆盖来自历史 **GT** 集合的离线交集，只衡量重复词机会，不是实际模型召回率、WER 改善或保证的收益上限。实际预测可能漏词，也可能输出额外候选。

因此预期整体 WER 改善可能小于重复关键词子集上的改善。全量结果仍作为主报告；预先声明重复/不重复子集作诊断，不能只筛“明显有效”的例子证明可靠性。

LibriSpeech 已切好的 utterances 缺少原始连续录音时间轴，本实验验证的是 chapter 顺序下的跨段状态反馈。不能报告为真实低延迟流式 ASR。后续若需验证真实连续音频、重叠切分和跨边界词，另找保留时间轴的长录音数据；本轮不扩展数据范围。

## 9. 对照实验矩阵

所有正式推理禁 GT，固定 checkpoint/greedy 参数/声学输入/segment 顺序；统计每个 arm 的提示长度和生成长度。

| Arm | 模型 | 输出要求 | bias 来源 | 回答的问题 |
|---|---|---|---|---|
| A0 | 原 base | 原转写任务 | 无 | 原始 ASR 基线 |
| A1 | 原 base | 新转写+关键词 prompt | 无 | 仅改 prompt 的收益与代价 |
| A2 | 新 RL | 新联合任务 | 无 | H1：是否学会抽词，正文是否保持 |
| A3 | 同一新 RL | 新联合任务 | 上一段自身预测 | H2：最小真实闭环收益，主 feedback arm |
| A4 | 同一新 RL | 新联合任务 | 前三段自身预测 | history 策略敏感性，先在 dev 选参数 |
| A5 | 同一新 RL | 新联合任务 | 过去段 GT 关键词 | 过去信息若抽取完美时的诊断；不是可部署模式 |
| A6 | 同一新 RL | 新联合任务 | 匹配长度的随机无关词 | 抗干扰、提示复制和错误反馈压力测试 |
| A7 | 原 base | 原转写任务 | 过去段 GT 关键词 | 核验现有 checkpoint 的 biasing 能力；诊断模式 |

若资源有限，先 A0/A1/A2/A3，再做 A5/A6。A4 只在 dev 做策略选择，冻结后对官方集报告。A5/A7 只能读取过去 GT，不是当前 GT；如要当前 GT 提示作理想诊断，必须单列 clairvoyant arm，绝不混进真实闭环结果。

独立 keyword 测试文件中的原有 distractor lists 可另做“固定候选 biasing”诊断，区别于由过去预测生成的 hints；默认正式闭环不读取它们。

## 10. 指标与可靠性判据

### H1 联合输出

按 test-clean / test-other、internal-test 分别报告正文 p_err/WER/CER、keyword micro/macro P/R/F1、集合 exact、块格式成功率、缺块/重复/截断率；空目标假阳性率与非空目标召回分开。训练 probe 只描述拟合。

### H2 反馈识别

报告 A3-A2 的总体 WER 差和相对 WERR、正文 keyword/B-WER、非关键词 U-WER、插入数；预定义可重复关键词子集的识别改善。B-WER/U-WER 的分母和 edit alignment 使用现有 score.py/ASR bias scorer 同一规则并记录版本，不凭字符串命中率替代词错误率。

记录 hint precision（提示词是否在当前 GT 出现，仅离线评分）、history reuse/hit、错误传播持续段数、空目标插入、首段 vs 后续段、提示 token 数、每段时延/RTF、会话串行带来的吞吐变化。prompt hints 不是硬约束，不能把“被提示的词一定输出”当验收标准。

固定各 arm 配对，按 chapter/session bootstrap 给差值 95% CI。因样本状态有依赖，不能把同 session 的各句当作独立重复。选择超参只看 dev；多个 seed/对照资源不足时如实报告不确定性。

实施前提议冻结非劣界，例如正文 WER 最多允许 +0.1 个百分点；这是待讨论的工程界限，不是已达成结果。若 H1 成功但 A3 未优于 A2，结论为“已学会列表输出，反馈收益未证实”，不能以格式成功率代替 H2。

## 11. 实施里程碑与退出条件

| 阶段 | 工作 | 进入下一阶段的条件 |
|---|---|---|
| P0（本轮） | 方案、源码定位、本地 manifest/audio header 审计、流程图 | 文档和证据可浏览；无模型/训练功能修改 |
| P1 数据协议 | 冻结 common5k policy；检查云端现有 manifest；排除重复文本；生成 split/probe 清单与 hash | train/dev/internal/official 分离；空标注真实；音频可读；字段适配验证 |
| P2 Prompt/GT/parser | 新任务 opt-in、完整 formatter、严格 envelope parser | 前缀+GT round-trip；WER 不含列表；超长不静默截断；旧任务回归 |
| P3 Reward/GT 联动 | 列表 reward、baseline 隔离、组 uid、log-prob/mask | sanity 排序正确；train 恰好1/n GT；所有 eval 0 GT |
| P4 无反馈 pilot | 固定 10k 或更小的训练试验、dev/probe 监控 | H1 指标达标、正文质量可接受、没有 token/advantage 异常 |
| P5 闭环评估 | 实现 prev_only、wave batching、状态恢复；冻结模型先跑 A2/A3/A5/A6 | 因果性/隔离测试通过；会话顺序与真实 hints 可追溯 |
| P6 冻结正式测试 | 选定 checkpoint 和 history 策略后跑 internal/official | 分数据集报告 CI、错误案例、无反馈对照和限制 |
| P7（可选）反馈适配训练 | 若现有 biasing + 新输出组合不稳定，用过去预测 hints 做训练 context | 先证明瓶颈；不默认扩大 RL rollout 为多步环境 |

P7 若启用：缓存同一 checkpoint 在训练 split 上的 past-only 预测，训练输入混合无 hints/正确历史/噪声历史；目标仍为当前段真实 transcription+keywords。缓存记录 producer checkpoint/prompt hash，不能用测试预测训练。训练用 GT history 属于理想条件增强，不代表真实闭环；后续仍必须用预测 history 测试。端到端跨多段 RL 和延迟 reward 暂不纳入第一版。

## 12. 关键测试案例

- 旧格式、新格式、空目标、缺标注、CSV 引号/逗号、保留字、语言头和 prefix round-trip。
- 当前 GT 与输出列表分别保持纯 reference 和完整 completion；超长示范在 rollout 前显式失败或过滤。
- GT 注入仅用于 train sampling，greedy/validation/standalone eval 0 GT；mask 在 shard/pad/dedup/reorder 后对齐。
- 下一段 prompt 真正包含前一段**预测**关键词，且相应 input_ids 已更新；没有 refs/distractors 字段进入构建函数。
- 第一段无 history；空预测/坏格式/超时按策略重置；跨 session、speaker/chapter、split、arm 无状态串扰。
- worker 回传乱序用 request/session/index 对齐；重复/填充行不重复更新 history；一次解码失败不读取后续 reference 来补救。
- 同 session 不允许两段同时 in flight；多个 session 可并行；重启恢复与完整 greedy 执行一致。
- 错误关键词可导致正文/list 自洽但与音频不符，A6 必须暴露该风险；TTL 不因无限提示复制而永不过期。

后续实现、测试和实验准备仍在上述独立 worktree/远程分支中进行。本轮提交仅为文档、图解和可复现的数据审计工具/结果。

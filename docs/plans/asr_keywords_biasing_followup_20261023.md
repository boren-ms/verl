# 后续计划备忘：预测关键词用于跨段 biasing（暂不开展）

日期：2026-10-23。根据用户最新决定，本次实验仅研究转写与关键词联合输出。下面保存此前的连续 biasing 设计供下一轮讨论，不属于本次实施、调参、测试或验收范围；其中参数和实验编号均为历史提议，后续需重新确定。

前置条件：先完成[关键词联合输出实验](asr_keywords_joint_output_detailed_20261023.md)，验证正文质量与关键词准确性，再单独讨论是否开展反馈实验。本轮不新增 history、动态 bias prompt、连续调度器或相关训练。

## 后续想法

将第 t 段预测出的关键词按模型现有 biasing 提示格式用于第 t+1 段。需要单独验证因果边界、错误传播与真实识别收益；关键词输出成功本身不能证明反馈有效。

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


# BCR 机制和公平性核验

数值使用父任务冻结的 `reports/bcr_error_analysis_20261022T1811Z/sources/r16_fix_lineage.json` 和 `r64_history.json`，仅保留当前模型祖先中的 optimizer updates。复算脚本为 `analyze_mechanism.py`，完整数值在 `mechanism_analysis.json`，源码版本和 SHA256 在 `sources/code_manifest.json`。后来的 live 观察独立保存在 `live_observation.json`，不回写冻结曲线。

## 已确认的结论

目前结果不足以把退化单独归因于 replay。可以确定，同一步数下新数据曝光不同；R16 保留前 50 步另一套 loss 实现；R16 还出现了可解析语言标签指标归零，而 delta reward 对 sampled/greedy 同时归零的输入没有这个分量的差分奖惩。应优先结合 validation 原始输出解释该异常，再做干净的 replay 对照。

### 实际保留的输入呈现次数

| 截止 optimizer step | 实验 | fresh | replay | 原始输入总次数 | 相对 Fix fresh |
|---:|---|---:|---:|---:|---:|
| 440 | Fix | 56,320 | 0 | 56,320 | 100.00% |
| 440 | R16 | 49,480 | 6,780 | 56,260 | 87.86% |
| 440 | R64 | 28,160 | 28,032 | 56,192 | 50.00% |
| 580 | Fix | 74,240 | 0 | 74,240 | 100.00% |
| 580 | R16 | 65,160 | 9,004 | 74,164 | 87.77% |
| 580 | R64 | N/A | N/A | N/A | 只覆盖到 458 |

Fix 用每步 fresh128 的配置和完整 step 覆盖计算；BCR 对每一条 retained update 的 `replay/fresh_count`、`replay/replayed` 求和。这里是输入呈现次数，不是 unique audio、有效 token 或 GPU 成本。具有 `train/num_gen_batches` 指标的记录均为1，R64部分旧记录缺项；共同窗口401–440三组各40步完整覆盖且均为1。replay计数由逐步日志汇总，`ReplayMetrics` 本身会累计同一步的多次generation。

R16 保留的分段是 1–50 fresh116/replay12、51–230 fresh112/replay16、231–430、431–570、571–609；五段实际 replay 为 588、2,864、3,184、2,224、608，且每段第一步 buffer 都为空。R64 保留原提交 1–390 和 recovery390 后 391–458，实际 fresh29,312/replay29,184；原分支391–396、失败 recovery、R16 被回滚的分支不计入。

### replay 到底选择了什么

现代 R16 与 recovered R64 的 `bad_case_replay.py` 完全相同：输入重新送入当前 policy，重新生成 sampled 和 greedy 输出。选择分数默认是 greedy raw `word`，不是训练 scalar reward。`random` 仅表示在当前低分 hard pool 内均匀抽取，不是从所有历史输入做均匀 reservoir replay。

`prepare()` 把抽中的身份暂时移出 resident map，抽取数为 `floor(min(replay_size, buffer_size)/DP)*DP`。当前 batch 的所有 finite fresh/replay 身份按最新分数与剩余 buffer 合并，去重后保留全局最低的 capacity 个。没有阈值、年龄、冷却、使用次数上限、毕业条件或 source quota。buffer 不存 checkpoint，恢复后为空；空缺 replay 不用额外 fresh 填补。

`replay/score_mean/min/max` 是**当前 generation batch 中全部 finite fresh+replay 输入**的 greedy 选择分数统计，代码为 `scores[valid].mean()`。若一次训练步存在多次生成，`ReplayMetrics` 按 `score_count` 汇总。它既不是 resident buffer 摘要，也没有分别统计 fresh/replay。

`replay/reinsert_rate` 表示本次抽中的身份有多少在全局容量排序后留下，是 retention 比例，不是是否学会。现有日志未记录 buffer 的身份/source 分布、单个身份累计复用次数、年龄；不能从高 retention 或 score_mean 推出某类数据占据了 buffer。

### 共同窗口 401–440

| 指标（每步均值再平均） | Fix | R16 | R64 |
|---|---:|---:|---:|
| raw word | 0.904409 | 0.829178 | 0.892065 |
| scalar reward | −0.005900 | −0.003827 | −0.006282 |
| actor KL | 0.032995 | 0.086654 | 0.029013 |
| raw lang | 0.938916 | 0 | 0.954932 |
| gen_kept_frac | 0.667773 | 0.721652 | 0.685352 |

R16 scalar reward 更高并不表示绝对识别质量更好。它奖励的是 sampled 相对于同输入当前 greedy 的差分，训练混合与模型 baseline 都在变。其 raw word 明显低于 Fix，validation 也应按固定数据集独立判断。R16 actor KL 更高是同时出现的现象，不构成原因证据。

R64 窗口 fresh2,560/replay2,560，reinsert2,314，按 replay 次数加权 retention=90.390625%；R16 fresh4,480/replay624、reinsert579，加权 retention=92.78846%。R16 的 per-step reinsert_rate 简单平均为90.46875%，因为431冷启动记0；解释保留率应使用579/624。

| 最终训练 rollout 行的来源 | Fix 行数（占比） | R16 行数（占比） | R64 行数（占比） |
|---|---:|---:|---:|
| openml | 23,424（57.19%） | 24,544（60.11%） | 22,344（54.55%） |
| ls_rare_verb | 11,832（28.89%） | 10,640（26.06%） | 12,608（30.78%） |
| earnings_fy27 | 5,704（13.93%） | 5,648（13.83%） | 6,008（14.67%） |
| 总数 | 40,960 | 40,832 | 40,960 |

`metric_utils.py` 统计的是重复和过滤后的 final training batch rollout 行（n=8），不区分 fresh/replay，不等同 buffer 组成或 unique prompt。R64 的混合在这个窗口少一些 openml、多一些 ls_rare；无法证明这是 buffer source concentration，更无法单凭此表说明 validation 退化的成因。

### lang=0 的实际含义

Fix、现代 R16、recovered R64 的 `asr_vote.py`、`asr_measure.py`、`asr_response.py`、`languages.py` 在各自 pinned Git blobs 中相同。18:17 R16 head/worker 的当前磁盘 hash 也一致。历史进程实际加载的 module identity 没有独立证据，不能因此声称 scorer 版本差异。

2609 parser 需要 `<src=...><tgt=...>\n`。`<src=German\n正文` 无法匹配，解析成 src/tgt/text=None；`check_lang()` 对比解析出的 target language code 序列，单一 German 参考与空预测有一次 deletion，因此 lang=0。它不做正文语言识别。`get_hyp_text()` 在没有解析文本时退回原串，`clean_text()` 会去掉开头 `<src=German`，所以德语正文仍可正常进入 word scorer。没有 `<VERBATIM>` 本身不导致 lang=0，只要合法 tgt header 仍在。

复算脚本执行保存的源函数，以相同德语正文对合法 header、无 VERBATIM、缺 > 和 tgt、错误 tgt 的**人工例子**核验：清理后的正文相同，lang 分别为1/1/0/0。这些例子用于验证机制，不冒充观测样本。父任务实际 validation cases 中的 `<src=German\n...` 可证明 validation 格式异常；没有训练 raw dump 时，不能说训练每条都采用该形态。

R16 609条训练记录中441条 `reward-core/lang/mean=0`，从224一直到609连续为0。401–440的40步 `lang`、`d_lang`、`v_lang` 的 min/max/mean 全为0；在这一记录覆盖的样本总体内，sampled与相应greedy的语言分量相同且为0，训练没有收到该分量的差分奖惩。当前 recipe 是 `lang: {delta: abs, beta:0.2, vote:delta}`，再除以1.6：

| sampled lang | greedy lang | lang delta | 对scalar的贡献 |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | +1 | +0.125 |
| 0 | 1 | −1 | −0.125 |
| 1 | 1 | 0 | 0 |

因此可说“当前共同失败时该 reward 分量没有差分信号”，不能说不可恢复或全部训练 loss 为0；探索出合法 header 仍会获得正贡献，其他 reward 分量也继续起作用。

### 其他公平性边界

R16 前50次 retained updates 使用 `ReplayLossNormalizer`；它按 all-reduce 后的真实 token/row 总数归一化，并补偿 FSDP rank 平均，不等同原 dynamic microbatch 的行数加权。该分支在全批时也会生效，不能当成仅 padding 的实现细节。后续换回 original actor，不会消除前50步在模型祖先中的影响。现代 R16/R64/Fix actor source hash 相同；R64 从头使用 original actor。

R16 capacity400/replay16，R64 capacity1280/replay64，两个量一起改变。R64 同步数仅有一半 fresh；R16 早期 loss 和 replay 配比也变过。numeric dataloader cursor 恢复不证明同一条音频序列，因为 grouping/shuffle 的完整种子连续性尚未证实。

## 可以检验的机制假设

较少 fresh 曝光可能降低新样本覆盖；长期保留的困难输入可能包含噪声或不可学习样本，并给某些身份重复加权；R16 header 异常可能削弱输出协议和语言分量训练。以上都与结果相容，但现有日志缺少 buffer IDs/source/reuse/age、fresh/replay 分拆分数和 training raw output，不能给出单因果结论。高 retention 只证明身份依然排名靠低，高 batch score 只说明当前批次分数。

## 最小可证伪下一实验（仅建议，未启动）

从一个共同、header 健康的 checkpoint 出发，固定 source/parser/original loss、优化器、数据种子与验证解码，做三臂200步：off（fresh128）；uniform all-seen reservoir（fresh112+replay16，capacity400）；现有 hard pool（fresh112+replay16，capacity400）。这能区分 fresh 曝光/重复训练的一般效应与“按难度选择”本身的效应；R64大小和容量变化留到后一轮逐一改变。

每20步分别报告四个 validation dataset 的 p_err；另在 off175 与 replay200处报告相同22,400 fresh呈现预算视角，明确optimizer预算不同。保留固定step与相同fresh两个视角，补充unique audio、有效token和GPU时。三臂从相同音频stream按确定性ID读取并保存顺序；每步保存抽样的 sampled/greedy 原始输出、解析成功率、buffer身份/source/年龄/复用次数，以及 fresh/replay 分拆的 greedy raw word、sampled raw word和delta reward。

若 hard 比 uniform差且buffer集中/复用偏差可复现，才支持困难池机制问题；若 uniform与hard都只在固定step差、相同fresh后恢复，更支持曝光差异；若统一loss且header健康后差异消失，原R16不能作为干净的replay负结论。若header仍坏，应先区分原始生成协议异常与评分器解析行为，再讨论replay策略。

## 可定位的源码证据

- `bcr16_modern__verl__trainer__ppo__bad_case_replay.py`：230不持久化；245–278配置和fresh；286–310抽样/移出；333–383合并、保留和当前批score；394–421聚合。
- `bcr16_modern__recipe__phimm__reward__asr_response.py`：5–9严格header正则；105–128解析失败；131–134词文本fallback；25–47文本清理。
- `bcr16_modern__recipe__phimm__reward__asr_measure.py`：432–450目标语言序列对齐；540–550 delta定义。
- `bcr16_modern__recipe__phimm__reward__asr_vote.py`：53–87差分、clip、beta；191–210 greedy baseline和reduce。
- `bcr16_modern__recipe__phimm__config__v2611_dev__dev__remax_dev_base0_v1h2a_fix.yaml`：17–23 reward系数。
- `fix__verl__trainer__ppo__metric_utils.py`：103–104 final rollout行定义；213–215 source计数。
- `bcr16_early__verl__workers__replay_loss.py`：40–63全局分母；72–86权重和scale。对应actor 440–478控制分支。

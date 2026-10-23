# Validation 样例与可比性审计

审计冻结：2026-10-22T17:17:11.751869+00:00。本文件只汇总已有的完整配对dump与真实table证据；每项保留原抓取时间和实际step，不代表新一轮实时文本。

9个run均有完整validation dump样例覆盖；可直接配对的step因run而异。dev四集分别报告，Earnings以6个完整parent计数（341 chunks），LibriSpeech clean2620、other2939及OpenML393条合计5958评分单元。Hindi独立753条。

| 配对 | 样例step | 原证据时间UTC | 覆盖 |
|---|---:|---|---|
| fix vs historical_v1h2a | 80 | 2026-10-22T07:34:57.360498+00:00 | dev四集完整配对 |
| L0 vs fix | 420 | 2026-10-22T15:45:37.109277+00:00 | dev四集完整配对 |
| L1 vs fix | 100 | 2026-10-22T07:36:55.408099+00:00 | dev四集完整配对 |
| L2 vs fix | 280 | 2026-10-22T15:46:01.679967+00:00 | dev四集完整配对 |
| bcr64 vs bcr | 100 | 2026-10-22T07:39:38.331111+00:00 | dev四集完整配对 |
| bcr vs fix | 540 | 2026-10-22T17:06:26.187159+00:00 | dev四集完整配对 |
| hindi_v1 vs hindi_v0 | 640 | 2026-10-22T16:50:39.778148+00:00 | Hindi完整753条 |

这些样例用于解释对应step的观察，不能将step100模式当作最新380/500结果的机制。样例按stable dataset/id及相同reference配对；rank和行号不作为身份。配对缺失、分母变化和原始JSONL出处保存在examples_audit.json及原证据中。

dev组fresh W&B配置的data/model/rollout/val_reward完全相同。L0/L1/L2的actor/algorithm也相同；BCR仅多replay设置及不同optimizer horizon。历史bee...是v1h2a，不能误标v1h2a3。代码与恢复史仍不同，相同step并非相同训练prompt或曝光量。

Reward总分分组解释：fix与BCR为五项delta、clip±10、除1.6；L0为六项delta、除3；L1/L2新增char，clip±2、除4，absolute lang/fmt近满分时分别引入约+0.05/+0.005。L0/L1/L2的language正确性定义也不同于fix，因此跨公式score和lang曲线不用于质量排名。

BCR16的前50步是replay12/fresh116及ReplayLossNormalizer，51步起replay16/fresh112；BCR64为replay64/fresh64、buffer1280，原actor loss。容量与replay同时变化。同step380累计fresh呈现42760 vs24320，不能称同曝光实验。已回退分支和恢复重验证须从主曲线隔离。

已检查到的有界现象：

- 历史v1h2a的Earnings120/180/220存在malformed-tag放大deletion评分证据；图中保留官方p_err，并明确这些点的评分语义。后续高点未同样审计，不能类推。
- L0@420、L1@100、L2@280、BCR@540都存在数据集tradeoff；完整配对计数保存在JSON。OpenML同reference的有效n_ref可能随hyp归一化而变，收益中格式/计分影响需要单独核对。
- BCR16@540的全部6个Earnings parents比fix有更多错误；其raw输出严格2609头为0/341，对照299/341。其余dev集也有完整raw-output/parser证据；缺头不等于声学语言错误。
- Hindi@640为753个相同ID/reference/lattice配对，v0→v1错误2317→1216、p_err17.030503%→8.942492%。v0/v1 language semantics、Ray版本及恢复历史不同，不能归因absolute reward。

Table覆盖：L0/L1/L2和BCR16有实际val/generations对应dump的匹配证据；历史baseline table去掉累计重复；BCR64仅复用历史step100完整dump，最新step配对/table覆盖N/A。Hindi本地未发现实际table，完整753行dump覆盖其唯一validation集。selected table不能估计总体错误率。

后续最有价值的核对是：保持官方p_err，针对OpenML括号/compound归一化与Earnings新增删除逐条核对；从相同模型谱系补齐最新配对dump，再比较同step。任何追加训练都需单独任务授权。

[机器可读样例证据](examples_audit.json) · [模型/数据/公式/预算审计](cohort_audit.json)

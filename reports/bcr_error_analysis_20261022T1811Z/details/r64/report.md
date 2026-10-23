# R64 与 Fix：step 440 完整验证对比

正式结果来自冻结的完整 dump 和对应 retained lineage；四个数据集逐一核对，没有跨数据集合并平均。所有 S/D/I 重算均准确复现 logged n_err/n_ref 和 val-core p_err。

| 验证集 | 单元 | Fix p_err | R64 p_err | Δ pp | Fix S/D/I | R64 S/D/I |
|---|---:|---:|---:|---:|---:|---:|
| earning_chunk_verb | 6 parents / 341 chunks | 4.494966% | 4.540435% | +0.045469 | 340/270/82 | 361/262/76 |
| openml_bad | 393 | 8.454707% | 8.574472% | +0.119765 | 311/71/94 | 315/81/87 |
| ls_clean | 2620 | 1.152743% | 1.164063% | +0.011320 | 492/67/52 | 503/64/50 |
| ls_other | 2939 | 2.692985% | 2.666490% | -0.026495 | 1182/106/135 | 1174/111/124 |

R64 在 Earnings、OpenML、LS clean 的错误数分别增加 7、7、6，在 LS other 减少 14。Earnings 的两条变差 parent 合计增加 33 个错误，四条改善 parent 合计减少 26；其替换错误增加 21，删除和插入分别减少 8、6。LS clean 主要表现为替换增加 11，LS other 则替换和插入分别减少 8、11，删除增加 5。

| 验证集 | 改善 | 新增错误样本 | 原坏样本变差 | 原坏样本等率 | 一直正确 |
|---|---:|---:|---:|---:|---:|
| earning_chunk_verb | 4 | 0 | 2 | 0 | 0 |
| openml_bad | 20 | 16 | 11 | 142 | 204 |
| ls_clean | 31 | 23 | 15 | 380 | 2171 |
| ls_other | 85 | 41 | 31 | 766 | 2016 |

OpenML 的括号诊断对两侧的 raw reference 和 clean_output 一致地只删除 ASCII `(`、`)`，保留内部文字，再走同一语言分派及 normalizer。393 条逐条原分数均已复现。Fix 476/5630 → 464/5630；R64 483/5633 → 466/5632，因此错误数差距 +7 → +2，净差距中 5 个错误对格式敏感。三个例子 `8031031497862620363`、`16940047964722835471`、`9071737899953350186` 分别从差距 +4/+2/+2 变为 0，但同时存在 Fix 的括号型收益被抵消，不能只加总目标模型的个例收益。正式结果仍采用原计分。

内容变化的反例也同时存在：OpenML `2114_1656_000013` 恢复开头 “qui me menait”，错误 4→0；LS clean `7176-92135-0037` 把 “a meal” 写成 “Amiel”，0→2，而 `1995-1837-0005` 把 “inhuman” 还原为 “and human”，2→0；LS other `2609-169640-0000` 恢复 “the sea is full of them”，7→3，而 `1998-29454-0017` 的 “ain’t / she’ll” 变为 “an’t / you’ll”，0→3。

覆盖：所有 5958 个验证单元完整配对，Earnings 的 341 chunks / 6 parents 均完整。两侧 W&B step440 各 20 条低分表样本均唯一回连到完整 dump；表不含 Earnings，不能据表推断各集错误率。OpenML 相同 raw references 中有两条经联合归一化后分母不同，保留实际 5630/5633。

注意：技能生成的 OpenML summary.json 的 baseline_wer 使用了 target 分母；本报告和 analysis.json 使用正确的逐侧官方分母。详细 transcript HTML 不含该总指标头。单 step、单 seed 只能描述当前差异，尚不能归因到 BCR 的因果机制。

机器可读证据：`analysis.json`、`examples_evidence.json`、`matched_validation_cases.jsonl`；诊断：`parentheses_diagnostic.json` 和可重跑脚本 `parentheses_diagnostic.py`。

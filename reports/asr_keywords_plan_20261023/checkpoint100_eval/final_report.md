# LibriSpeech 转写 + <KEYWORDS> 实验最终报告

模型学会了输出关键词列表，但本次100步训练未改善正式测试集的转写准确率。test-clean / test-other 的 WER、BWER 均高于 A0 和 A1；非关键词 UWER 略有改善。关键词列表 micro precision 仅为33.08%–48.27%，空关键词样本的误报率为92.73%–98.46%，说明格式学习成功，但关键词选择仍明显过量。

## 完成状态与来源

- 训练：100/100步成功，任务 `raysubmit_6PPiWDZg5QJ3zagk`。
- 选择：仅使用固定 dev1000 的最小 corpus WER（同分取较晚步），选中 checkpoint100，p_err=0.007720556342711737；测试集不参与选择。
- 导出：LoRA rank256/alpha512，176对适配器合并；781个有限值BF16张量，与匹配基座的键和形状一致，训练目标权重差异已确认。
- 评估：`raysubmit_xJU7mCyiJTL2XtAy` SUCCEEDED；双节点、greedy、无GT注入，仅评估 internal-test / test-clean / test-other。
- 输出：11,690条、11,690个全局唯一ID，与三个冻结manifest的ID/参考转写/关键词/来源哈希完全匹配；逐条验证 n_err=nb_err+nu_err、n_ref=nb_ref+nu_ref。
- JSONL 内 step=0 是独立评估的计数器，实际模型为训练后的 checkpoint100。原生vLLM结束原因已保留。
- 训练与评估均成功结束，16块GPU已释放。

## 训练模型的转写指标

| 数据集 | 条数 | WER | BWER | UWER / NWER |
| --- | --- | --- | --- | --- |
| internal-test | 6131 | 1.937% | 8.995% | 1.134% |
| test-clean | 2620 | 2.573% | 9.998% | 1.660% |
| test-other | 2939 | 4.295% | 17.140% | 2.832% |

WER/BWER/UWER 均由语料级错误计数之和除以参考词数之和得到；NWER 是 UWER 的别名。

## 关键词列表指标

| 数据集 | micro P | micro R | micro F1 | exact | format | 空目标误报率 |
| --- | --- | --- | --- | --- | --- | --- |
| internal-test | 48.271% | 62.763% | 54.571% | 4.045% | 96.412% | 98.462% |
| test-clean | 38.222% | 63.141% | 47.618% | 6.756% | 94.885% | 94.219% |
| test-other | 33.084% | 60.575% | 42.795% | 6.465% | 93.297% | 92.732% |

P/R/F1 为语料级 micro 聚合；exact/format 为逐条成功比例。空目标误报率的分母仅为参考关键词为空的样本。

## 与已有基线比较

| 数据集 | 模型 | WER | BWER | UWER / NWER |
| --- | --- | --- | --- | --- |
| test-clean | A0 原始提示 | 2.358% | 7.794% | 1.690% |
| test-clean | A1 联合提示 | 2.353% | 7.655% | 1.700% |
| test-clean | A2 训练100步 | 2.573% | 9.998% | 1.660% |
| test-other | A0 原始提示 | 4.157% | 15.215% | 2.898% |
| test-other | A1 联合提示 | 4.174% | 15.346% | 2.903% |
| test-other | A2 训练100步 | 4.295% | 17.140% | 2.832% |

A0/A1 均为同一匹配基座，分别使用原始转写提示与联合关键词提示；两组已有官方测试输出均为完整5559条，本次未重跑。A1 关键词列表 recall/F1/exact/format 均为0，precision未定义（未输出关键词块）。internal-test 没有已有的匹配A0/A1结果，因此不报告该集相对改善。

- test-clean 相对 A0：WER 增加 0.215 个百分点（相对错误率 +9.11%），BWER 增加 2.204 个百分点（+28.29%）。
- test-other 相对 A0：WER 增加 0.138 个百分点（相对错误率 +3.31%），BWER 增加 1.925 个百分点（+12.65%）。

## 解读与限制

训练 reward 的上升主要来自关键词列表和格式学习；不能把它等同于转写质量提升。固定训练dev是按未训练模型最低BWER筛选的1000条容易样本，筛选时WER/BWER为0，因此dev上选出的最好checkpoint仍须由独立测试验证。本次测试显示关键词词错误增加，非关键词错误略降，整体WER变差。模型在参考为空的样本上仍频繁生成关键词；后续可优先分析词表标注标准与空目标/误报惩罚，再决定新实验。本报告不据此自动更改训练设置或启动额外训练。

## 产物与复现

- [完整指标](metrics.json)、[输出核验](verification.json)、[基线差异](baseline_comparison.json)、[CSV](all_metrics.csv)。
- [基线比较图](baseline_comparison.png)、[比较图PDF](baseline_comparison.pdf)。
- [训练/validation交互图](../training_charts_20261024/index.html)、[3条原始validation样本](../validation100_first3_raw.md)。
- [导出来源与权重SHA256](../export100_provenance.json)、[checkpoint选择依据](../checkpoint_selection.json)。
- [评估W&B](https://msaip.wandb.io/genai/2609-Keyword/runs/d96bbcbf363a1d7cab0cff8c5d03160a)。
- 训练模型：`az://orngwus2cresco/data/boren/outputs/2609-Keyword/remax_ls_joint_pilot_dev1k/global_step_100/qwen_hf`。
- internal-test 原始输出：`az://orngwus2cresco/data/boren/outputs/2609-Keyword/eval_joint_checkpoint_ls/val_data_gen/ls_internal/0.jsonl`，SHA256 `aaa1dd1e6f8c5f15435f2d9d00f116ba559e6b2b75f9d00d50a19bc5e91ef64b`。
- test-clean 原始输出：`az://orngwus2cresco/data/boren/outputs/2609-Keyword/eval_joint_checkpoint_ls/val_data_gen/ls_clean/0.jsonl`，SHA256 `38aebe51d7aeffff76d1a7961906fb831bb7e0f5eeb31dc0cdc2d10df3a246f8`。
- test-other 原始输出：`az://orngwus2cresco/data/boren/outputs/2609-Keyword/eval_joint_checkpoint_ls/val_data_gen/ls_other/0.jsonl`，SHA256 `ffa39b1c9047487c4eb29ea57db37df72c26e0362fa14a757ad5724ac51f309e`。

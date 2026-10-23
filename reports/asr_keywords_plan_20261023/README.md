# LibriSpeech 关键词方案：本地数据核验

核验日期：2026-10-23。这里的脚本仅做只读数据审计，没有训练、推理或修改源数据。

复现命令（在仓库根目录）：

```bash
python reports/asr_keywords_plan_20261023/audit_local_data.py \
  --data-root /home/boren/data \
  --output reports/asr_keywords_plan_20261023/local_data_audit.json
```

Python 标准库用于 manifest 审计；安装 soundfile 时额外检查测试音频 header。脚本逐行解析原始训练 TSV，使用 `ast.literal_eval`，不执行源文件代码。完整运行结果和源文件/词表 SHA-256 见 [local_data_audit.json](local_data_audit.json)。

| 检查 | 结果 |
|---|---|
| 原始训练 TSV | 280,393 行、无重复 id；132,266 条音频本地存在 |
| 本地原始 test-clean/test-other | 2,620 / 2,939 行，全部音频存在，16kHz mono，音频 header 无错误 |
| 官方本地关键词标注复现 | 固定 common_words_5k 排除规则在 5,559 行全部一致 |
| 用 all_rare_words 取交集 | clean 108 行、other 161 行与已有标注不同 |
| train/test id 和 speaker 交集 | 均为 0 |
| train/test 归一化正文相同 | 7 条训练记录，全部落在 train fold；正式准备时排除 |
| train 派生空关键词样本 | 24,180 行；不需要从官方测试集借空样本 |
| 有序测试会话 | clean 87、other 90 个 chapter；5,382 对相邻编号连续 |
| 历史 GT 的上一句重复机会 | 385 / 10,522 个后续句关键词条目，约 3.66% |
| 历史 GT 的前三句重复机会 | 782 / 10,522，约 7.43% |

历史覆盖率只反映按 chapter 顺序的重复词机会，不是模型效果。所有派生标签统计都基于脚本中明确记录的英文 normalization；不将其冒充为云端原始关键词 manifest 的已有统计。

云端 `asr_train_transcribe_rare_keywords_nonempty.jsonl` 与 `test-*.biasing_100.think.jsonl` 已定位但本机无可用读取凭据，尚未读取。这里实际审计的是本地原始 TSV 和不带 `.think` 后缀的 biasing JSONL，不断言它们与云端文件逐行相同。

数据划分只是对拟议 split 函数进行只读统计，正式训练/验证 JSONL 尚未生成。音频存在性与 header 检查也不代表全部音频经过逐帧解码或人工检查。

详细设计与修改位置见 [v2 方案](../../docs/plans/asr_keywords_continuous_detailed_20261023.md)，交互流程图见 [HTML](../../paper_notes/asr_keywords_continuous_architecture.html)。

# LibriSpeech 转写与关键词联合输出：统一实验计划 v5

日期：2026-10-23。状态：**联合输出代码、配置和本地预检已完成；Brix/Blob 已恢复，A0 baseline 已提交运行；训练待两套 baseline 完成**。

唯一当前计划为本文件；v1/v2 入口均指向这里。连续 biasing 在[后续备忘](asr_keywords_biasing_followup_20261023.md)中，属于暂不开展的独立实验。实现位置见第 10 节；可运行配置、命令和当前限制以 [v2609_kw/README](../../recipe/phimm/config/v2609_kw/README.md) 为准。下面保留实验设计，最终落地与执行状态见第 14 节。

- 远程/本地分支：`boren/asr-transcription-keywords`。
- 独立 worktree：`/home/boren/.codex/worktrees/asr-transcription-keywords/verl`。
- [流程图](../../paper_notes/asr_keywords_joint_output_architecture.html)。
- [数据审计结果](../../reports/asr_keywords_plan_20261023/local_data_audit.json)与[只读审计脚本](../../reports/asr_keywords_plan_20261023/audit_local_data.py)。

## 1. 目标、已确认决定与本轮范围

本次实验定性为**关键词联合输出**：让模型对一条音频既给出原始 transcription，又给出该段音频转写中实际出现的 rare words 列表。检验 RL 是否能让原 base 学会此能力，同时保留正文识别质量。

用户已确认：保留原转写；`<KEYWORDS>` 独占一行，下一行使用英文逗号分隔；没有关键词时保留头和空行；利用已有关键词数据；RL 使用完整 ground-truth rollout；在 LibriSpeech 上训练与验证；先做详细计划，不急于实现。

本轮每条音频独立生成，不输入关键词候选、distractors 或 history。跨段预测反馈、连续识别、history TTL、rolling 策略、流式调度器及相关训练全部留待下一轮计划。

验收有三个独立维度：正文质量、关键词集合准确性、输出协议正确性。仅有格式成功率不能证明抽词成功；模型“输出能在转写中找到的词”也不代表这些词在音频中识别正确。

## 2. 已有 rare-word 配置与本计划的统一选择

| 配置 / 代码 | 现有行为 | 本轮如何使用 |
|---|---|---|
| `recipe/phimm/config/data/train_data/ls_rare05.yaml` | 原始 TSV；`common_num: 10000`、`rare_ratio: 0.5`；`task: rare_asr` | 保留为历史 rare-word 来源参考，不直接继承其抽词规则或 task |
| `recipe/phimm/data/dataset.py:add_rare_keywords` | `text_norm(english)`，去重并排除单字符词，减去 common 集合；可与 rare_file 取交集；超上限随机抽样 | 可参考数据处理入口；不能把旧函数默认参数直接用于 v1 新协议 |
| `recipe/phimm/config/data/train_data/ls_rare_nonempty_verb_langhint.yaml` | 读取已有 nonempty JSONL；`lang_asr_verb`、English、`lang_hint: true`、`prefix_prob: 0.0` | 复用任务外壳与数据接入方式；预生成标签先核验 |
| `recipe/phimm/config/data/train_data/ls_rare_nonempty_verb.yaml` | 同一 JSONL，无显式 language hint | 历史变体；本轮固定使用有 language hint 的版本作为任务基础 |
| `recipe/phimm/config/v2609_ls/remax_2609r2_ls_s400_bs128_n4_r256_g32.yaml` | 引用 nonempty 数据；`word_kw` 衡量正文中关键词的识别 | 复用 ReMax/正文 reward 结构；它没有关键词列表监督或新块解析 |
| `recipe/phimm/config/v2609_ls/gen/gen_2609v0_ls_rare_nonempty_10k_g16.yaml` | 覆盖为已有 seed42 10k manifest | 定位旧数据；与本轮 speaker split 后的新 10k 不能视为同一产物 |
| `recipe/phimm/config/data/val_data/ls_kw_verb_langhint.yaml` | 使用云端 test-clean/other `.think.jsonl` | 仅作路径参考；本轮优先已核验的本地原始 reference |

**统一决定（本计划的实施默认）：** 训练使用现有 LibriSpeech raw TSV 和固定 `common_words_5k.txt` 生成完整 rare-word 目标；测试使用已有 keyword reference 标注。理由是 common5k 规则已经逐行复现全部 5,559 条现有测试标注，可保持训练/测试定义一致。common10k+随机裁剪不是本轮协议，`all_rare_words.txt` 交集也不是本轮协议。

这是一套从现有数据派生的新 manifest，必须有新名字和来源 hash。已有 nonempty JSONL 如果核验与本规则完全一致，可在相同 split/去重后复用对应记录；不一致则作为独立 legacy 版本记录差异，不拼接混训，也不宣称其原始生成方式已经查明。

## 3. LibriSpeech 数据清单与核验状态

### 3.1 本地可用的数据与音频

| 数据 | 已读路径 | 已核验规模 | 用途 |
|---|---|---|---|
| LibriSpeech 原始训练 TSV | `/home/boren/data/LibriSpeech/asr_train_transcribe.tsv` | 280,393 行、5,466 个 speaker/chapter、0 重复 id；132,266 条音频本地存在 | 核验训练源，构造 train/dev/internal-test 和训练内 probe |
| test-clean 原始 keyword reference | `/home/boren/data/librispeech_biasing/ref/test-clean.biasing_100.jsonl` | 2,620 行；2,620 条本地音频，5.4035 h | 独立正文与 keywords 测试 |
| test-other 原始 keyword reference | `/home/boren/data/librispeech_biasing/ref/test-other.biasing_100.jsonl` | 2,939 行；2,939 条本地音频，5.3415 h | 独立正文与 keywords 测试 |
| 固定常用词表 | `/home/boren/data/librispeech_biasing/words/common_words_5k.txt` | 由审计脚本记录实际集合大小 | 确定 rare-keyword 标注协议 |

两个测试集的本地 flac 均通过 soundfile header 检查，为 16 kHz mono，无 header 读取错误。此检查不等价于逐帧解码或人工音质检查。

训练 TSV 与官方测试：id 重合 0、speaker 重合 0，但归一化 reference 文本有 7 条训练记录与测试文本相同。实施数据准备时把这 7 条从训练源排除，并给出同源文本敏感性统计，避免把“id 不重合”误写成完全无重复。训练源并非全部音频本地就绪，GPU 全量训练前必须完成远端路径/音频可读性检查。

### 3.2 已定位、尚未读取的云端候选

```text
az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe_rare_keywords_nonempty.jsonl
az://orngwus2cresco/data/boren/data/LibriSpeech/asr_train_transcribe_rare_keywords_nonempty_sample10k_seed42.jsonl
az://orngwus2cresco/data/boren/data/librispeech_biasing/ref/test-clean.biasing_100.think.jsonl
az://orngwus2cresco/data/boren/data/librispeech_biasing/ref/test-other.biasing_100.think.jsonl
```

训练配置：`recipe/phimm/config/data/train_data/ls_rare_nonempty_verb_langhint.yaml`；验证配置：`recipe/phimm/config/data/val_data/ls_kw_verb_langhint.yaml`。云端账号当前缺少本机可用读取凭据，本轮没有重试获取密钥或启动 GPU 节点。文件规模、字段和与本地文件的等同性尚未验证。

本方案可基于已有本地 raw TSV 和固定词表继续做协议/数据准备设计，不依赖声称云端文件已经可读。若后续采用云端现有 keywords，先与下面明确的规则逐行核验；不同规则应独立命名并报告，不能混为同一监督目标。

### 3.3 已验证的关键词定义

原始测试 JSONL 的 `text` 是正文；`ground_truth` 是关键词 list；`distractors` 是含干扰词的候选集。后者不作为目标。

在本地全部 5,559 行上，下面的集合规则与已有关键词标注完全一致：

```text
tokens = lowercase English words, preserving apostrophes and hyphens
target_keywords = unique(tokens(reference)) - fixed_common_words_5k
```

审计使用 `re.findall(r"[\w'\-]+", text.lower())`。这证明其在当前本地 reference 上一致，不宣称适用于所有语言或任意带标点的新文本。v1 使用上述原始正则，不额外做 Unicode apostrophe 替换、数字 verbalization 或缩写展开，不添加 `len(word)>1` 过滤；冻结 normalization 代码版本与词表 SHA。该限定服务于当前 LibriSpeech 数据，不扩展成通用多语言协议。

直接与 `all_rare_words.txt` 取交集会使 test-clean 108 行、test-other 161 行不一致。因此**本计划统一采用固定 common5k 排除规则**，不给 rare words 再添加外部词典存在性限制，也不采用 `rare_ratio=0.5` 的随机裁剪。

本轮 GT 列表使用归一化后的小写词，按参考转写第一次出现的位置排序并稳定去重；评分使用同一 normalization 的集合。正文仍保留原始转写，不为了列表改变正文大小写。空列表有真实监督，不能只用 nonempty 文件训练后忽略空输出行为。

## 4. 数据划分与“训练侧测试 / 独立测试”的边界

先将完整原始训练源按 speaker 做稳定划分，再过滤 nonempty 或抽样，确保不同阶段使用同一 split。采用 `int(SHA256("ls-keywords-v1:<speaker>").hexdigest(), 16) % 100`：0..4 dev，5..9 internal-test，其他 train。该规则已审计出各 fold 的规模，详见 JSON；排除 7 条重复文本后数字会变化。

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

## 5. 监督目标构建：单一确定性协议

标注版本命名为 `ls-common5k-v1`，输出协议命名为 `asr-keywords-v1`。英文任务、格式版本 `2609`、`lang_asr_verb`、English language hint、`prefix_prob=0.0` 为首轮默认。

目标算法（拟实现；审计脚本已有集合级核验）：

```python
# 不改 reference_text 本身；没有随机抽样，没有关键词数上限。
common = load_first_column_lowercase(common_words_5k_txt)
seen, ordered_targets = set(), []
for word in re.findall(r"[\w'\-]+", reference_text.lower()):
    if word not in common and word not in seen:
        ordered_targets.append(word)
        seen.add(word)
```

词表 SHA256：`30eadb83c338df9376e4fabd31bd1349becc06b896b81dbc5fb7dac6fca72a8e`。需将实际 source/word-list/normalization hash 一起写入数据元信息，不能只依赖路径。

对于已有 reference，先验证 list 类型和每项在正文中按 token 边界出现，再验证 canonical 集合与 common5k 规则一致；只调整顺序到首次出现顺序，不改变既有标注集合。如果出现不一致，本次数据准备失败并报告差异，不悄悄改写测试标签。

字段缺失、`null` 与 `[]` 必须不同：只有明确构建完成或读取到有效空 list，才是合法空目标；缺标注不是空目标。使用显式 `keyword_annotation_valid`，新任务缺此状态或目标未构建时 fail fast。

先做字段映射/正文清理与必要音频变换，再构建目标，最后构建 prompt/GT。本轮不启用改变参考正文的随机裁词、随机音频边缘裁切或分段增强。不能在目标生成后继续改写 `text`。不对无实体标签的 LibriSpeech 文本运行会覆盖 keywords 的 `clean_tagged_text`。

## 6. 输入 prompt 与输出格式：可直接核对的例子

### 6.1 Prompt

复用现有 `get_task_prompt(task="lang_asr_verb", version=2609, lang="English")`。本轮冻结 `rand=false`，保持原任务指令和已有音频处理约定，只追加：

```text
Also output the rare words that occur in this audio. After the complete
transcription, write <KEYWORDS> on a new line, then a comma-separated list
on the next line. List each word once, in its order of first occurrence.
If there are no keywords, leave that line empty. Do not add explanations.
```

现有 helper 的可见结果为：

```text
Transcribe the audio clip into text. Transcribe verbatim, including all filler words and disfluencies.<audio>
The language is English.
```

新指令加在这个原 prompt 后并保留原有 `<audio>`/processor 处理流程；不得额外重复插入占位符。模型不读取 common5k 全词表或当前 GT 词。“rare”是通过固定监督规则学习的概念，其泛化质量由 held-out 指标检验。

### 6.2 精确输出语法

```text
[原有语言头与模式头]
[完整原始 transcription]
<KEYWORDS>
[以英文逗号分隔的关键词，或空行]
```

`<KEYWORDS>` 为普通文本协议，不新增 tokenizer token 或 embedding；标签必须全大写并区分大小写，小写或混合大小写标签记为协议错误；只有一个块；无需 `</KEYWORDS>`；没有 JSON、Markdown code fence、项目符号或解释。正文模式继续为 `<VERBATIM>`，不新增 `transcription` tag。

生成端使用 `", ".join(target_keywords)`；解析端接受英文逗号两侧空格。当前目标是单个词，正则不会产生含英文逗号或换行的项，**v1 不引入 CSV 引号/短语语法**。例如 `"Washington, D.C."` 不作为本轮合法单项。短语与 CSV 支持留给未来协议版本，避免“有的配置 split，有的配置 CSV”的冲突。

### 6.3 例一：真实单关键词样本

来源：test-other `3764-168670-0020`。正文和标签来自本地原始 reference；此处只作格式示例，不用于训练。

```text
<src=English><tgt=English>
<VERBATIM>
asked jean valjean fauchelevent replied
<KEYWORDS>
fauchelevent
```

应解析为：

```json
{
  "transcription": "asked jean valjean fauchelevent replied",
  "predicted_keywords": ["fauchelevent"],
  "keyword_block_status": "valid_nonempty"
}
```

### 6.4 例二：真实多个关键词，按正文顺序

来源：test-clean `237-134493-0004`。原始 JSON 的标签 list 顺序为 `intermingled, mated`；目标集合不变，展示顺序按正文为 `mated, intermingled`。集合评分不惩罚仅顺序不同；生成目标固定首次出现顺序。

```text
<src=English><tgt=English>
<VERBATIM>
the air and the earth are curiously mated and intermingled as if the one were the breath of the other
<KEYWORDS>
mated, intermingled
```

### 6.5 例三：真实无关键词样本

来源：test-clean `2830-3980-0017`，已有标签为 `[]`。

```text
<src=English><tgt=English>
<VERBATIM>
when i was a young man i thought paul was making too much of his call
<KEYWORDS>

```

尾部的精确字符串为 `"\n<KEYWORDS>\n\n"`，之后是模型正常 EOS（EOS 是 token，不作为可见文本输出）。合法空列表应得到空目标的正确列表分，不能被当成漏块。raw response 必须保存 strip 前版本和 finish_reason，以识别空尾部。

### 6.6 例四：异常与评分边界

| 原始尾部 / 行为 | 状态 | 处理 |
|---|---|---|
| 正文后完全无 `<KEYWORDS>` | `missing` | 正文按旧 parser 提取；关键词任务分为 0，即使 GT 为空也不算成功 |
| `<KEYWORDS>` 与正文同一行 | `malformed` | 不当作合法块；保留可见错误，不能任意删去尾部来降低 WER |
| `fauchelevent, fauchelevent` | `malformed`，原因 duplicate | 保存原列表与去重诊断；主列表任务分为 0 |
| `fauchelevent,` | `malformed`，原因 empty_item | 不用空字符串填充列表 |
| 两个 `<KEYWORDS>` 块 | `malformed`，原因 duplicate_header | 不采取“取最后一个块”或覆盖前块 |
| 列表下一行后又出现解释文字 | `malformed`，原因 extra_lines | 不把解释悄悄丢弃；记录格式失败及可见多余内容 |
| 因 token 上限在头后结束 | `truncated` | 不当成合法空列表；主列表任务分为 0 |
| 正确语法，但词不在预测正文中 | `valid_nonempty` + consistency=false | 保留原预测并与 GT 评分，不先过滤到好看的结果 |
| 词在预测正文中但正文听错 | 语法可能有效 | WER 和 GT 列表分会反映错误，自洽不能代替正确性 |

## 7. 字段映射与 ground truth 生成：闭合的数据链路

### 7.1 字段语义

| 层 | 字段 | 类型与语义 |
|---|---|---|
| 本地测试原始 JSONL | `text` | 纯参考转写 string |
| 本地测试原始 JSONL | `ground_truth` | **关键词 list**，不直接传给 RL 正文评分 |
| 本地测试原始 JSONL | `distractors` | 候选干扰词；不进目标或输入 prompt |
| 统一 manifest | `reference_text` / `target_keywords` | 纯正文 string / 有序关键词 list |
| 现有数据处理内部 | `text` / `keywords` | 分别映射以上两个字段，兼容已有 scorer 取词 |
| RL `reward_model` | `ground_truth` | **纯正文 string**，始终供 WER/CER 使用 |
| RL `reward_model` | `gt_output` | 可接在实际 assistant prefix 后的完整联合答案 completion string |
| RL `extra_info` | `keywords`、标注状态、policy/protocol/id | 列表评分目标、数据审计和解析模式 |

测试 JSON 的 `ground_truth` list 与 RL 同名字段 string 是不同含义，必须在适配器显式转换。新任务禁止 `gt_output` 缺失时回退到纯 `ground_truth`；旧任务保持原行为。

### 7.2 GT formatter 实施规范

已新增共享 `recipe/phimm/data/keyword_output.py`，提供目标校验、关键词块 formatter 和 envelope parser，由数据 formatter 与训练/验证 scorer 实际调用。

```python
base_answer = get_task_output(
    task="lang_asr_verb", lang="English", text=reference_text, version=2609
)
full_answer = base_answer + "\n<KEYWORDS>\n" + ", ".join(target_keywords) + "\n"

# 首轮 prefix_prob=0，prefix=""，completion 等于 full_answer。
# 若将来启用 prefix，必须按真实 prefix 精确移除一次，不用 lstrip 改写正文。
assert full_answer.startswith(prefix)
gt_output = full_answer[len(prefix):]
assert prefix + gt_output == full_answer
```

对于例一，RL 中应保存：

```json
{
  "reward_model": {
    "ground_truth": "asked jean valjean fauchelevent replied",
    "gt_output": "<src=English><tgt=English>\n<VERBATIM>\nasked jean valjean fauchelevent replied\n<KEYWORDS>\nfauchelevent\n"
  },
  "extra_info": {
    "keywords": ["fauchelevent"],
    "keyword_annotation_valid": true,
    "annotation_policy": "ls-common5k-v1",
    "output_protocol": "asr-keywords-v1",
    "prefix": ""
  }
}
```

此 JSON 是字段映射示例，不是已生成的训练记录。测试的参考答案可以为评分/round-trip 离线构建，但绝不能通过 GT 注入给正式测试模型。

### 7.3 Token 与 rollout 完整性

1. 使用实际模型 tokenizer 编码 completion，显式约定特殊 token 行为，避免重复 BOS/EOS；正常文本编码优先 `add_special_tokens=False`，终止 token 按模型约定追加一次。
2. 长度必须包含正文、头、列表、换行和 EOS。提前统计 token 分布，记录最大值与分位数。
3. 本轮初始沿用基础配置的 768 response token 预算，最终在训练数据/dev 上核验后固定；确认 `data.max_response_length`、rollout response_length、validation max_tokens 和 model context limit 一致。
4. 超长训练示范在进入 PPO 分组前显式隔离/过滤并记录，不静默裁切。独立测试不因 GT 过长而排除样本；预测截断照实计入指标。
5. `parse(prefix + decode(gt_tokens))` 必须恢复同一正文、同一关键词集合和合法块，且 EOS/response_mask 正确。这是训练开始前的硬门槛。
6. prefix 属于模型输入，不能在 response loss/log-prob 中再次计入，也不能重复生成语言头。首轮 prefix 为空，但兼容性测试仍覆盖非空 prefix。

## 8. Parser 实施设计与所有评分调用点

修改前 `get_hyp_text(version=2609)` 会把合法关键词尾部混入正文；当前通过显式协议先调用 `keyword_output.parse_keyword_output`，再调用旧 segment parser。返回 `KeywordOutput(raw_response, body_raw, predicted_keywords, keyword_block_status, protocol_errors)`，正文由 `asr_keywords.parse_response` 提取。终止原因优先使用 rollout 传递的原生 vLLM finish_reason；只有旧产物缺字段时才回退到 EOS/response mask/budget 推断，并记录来源。length/timeout/abort 不计合法列表。新协议 mask 按真实返回 token 数限制，修复 PAD 与 EOS 不同且 stop 被消费时 padding 误算答案的问题。

解析步骤：

1. 通过显式 `output_protocol` 选择新 parser；旧任务走旧路径，不自动猜测所有任务均有新块。
2. 在原始文本中找**独占行、精确匹配**的 `<KEYWORDS>`。先识别 envelope，再调用已有语言/模式段 parser 解析正文，不能先清标签或 `.strip()` 后丢失结构证据。
3. 恰好一个头时分开正文与尾部；尾部只能有一个列表行及末尾空白。列表按英文逗号拆分、trim，每项验证为一个 v1 token，无空项、无重复。不使用 GT 参与解析或修复。
4. 没有头、重复头或坏位置时明确报错。非空异常尾部保存为 `unparsed_suffix`，在正文词法评分中保留其可见词；不因 parser 丢弃错误而改善 WER。合法列表才从正文中移除。
5. 合法头但语法错误、带解释的尾部不按合法列表剥离；诊断可单独展示解析候选，但主评分使用保守正文路径并记录协议失败。训练与独立评估采用相同失败规则。
6. `finish_reason=length` 或调度超时一律不是合法空结果。若解码层去掉末尾空白，只有确认正常 EOS/stop 时允许仅头结尾规范化为 `valid_empty_normalized`；无终止元信息又没有明确空行时记 `incomplete`，不奖励猜测空目标。原始合法空行可直接识别。
7. 单独检查 predicted keywords 是否在预测正文中出现，但不通过过滤预测来增加 precision；不要求输出列表等于“对预测正文再运行 common5k”的结果，模型预测要独立评估。

本轮训练、greedy baseline reward 与独立 evaluation 都通过 `main_asr_dapo` / `asr_keywords` 使用同一 envelope 结果；不使用 `main_asr_gen` 路径。现有 `word_kw`/`keyword` 是正文中关键词识别指标，继续基于纯正文；新增列表指标使用 `predicted_keywords`。禁止只修改离线 eval 而遗漏训练 reward。

若多语言/多段 components 出现，本轮数据检查应拒绝不支持的结构或先隔离，不能以英文单段 parser 静默处理。2607 本轮不扩展；旧 2607 行为回归验证。

## 9. RL、GT rollout 与 reward

### 9.1 最小训练设置与隔离

沿用现有 `v2609_ls` 的 ReMax 路径，新增独立 recipe，首轮默认 `n=8`、每个 prompt **1 条完整 GT + 7 条模型采样**。现有基础配置 `gt_rollout: false`，新 recipe 需显式开启，不能以为已自动启用。

已修复旧注入条件未排除 `do_sample=False` 的问题，增加明确 rollout purpose：只有 train sampling 才可注入，同时要求 `do_sample=True`、`validate=False`。greedy baseline、validation、standalone eval 均为 0 GT。

每组第一个 candidate 使用 GT，新增 `gt_group_id` 和组内 target 一致性校验；本地 shard 必须保留完整 n 行组，否则显式失败。GPU 分片、mask 与实际 vLLM 联动仍待远程运行验证。

复用 `gt_rollout.py` teacher-forced scoring 与 `rollout_log_probs.py` actor 温度对齐能力。GT 属于人为示范轨迹，报告其与 sampled 的不同来源；不宣称是无偏 on-policy 样本。检查 TIS/IS、KL、response_mask、EOS、非有限 log-prob、过滤器是否保留有效训练信号。GT 的 advantage 不保证每次为正，但正文和列表全对的示范应有正确 reward，且不能全被 mask/filter 掉。

### 9.2 列表计分：空目标与缺头无歧义

对有效列表，canonical set 记为 P（预测）与 G（GT）：

```text
TP = |P ∩ G|; FP = |P - G|; FN = |G - P|
非空常规情况：precision = TP/|P|，recall = TP/|G|，F1 = 2TP/(2TP+FP+FN)
P=G=空：per-sample P/R/F1/exact = 1
只有一方为空：per-sample P/R/F1/exact = 0（本协议的显式约定）
缺块、坏块、incomplete、truncated：主 kw_list_f1=0，exact=0，kw_block_fmt=0
```

有效列表顺序不同不扣集合分；重复是格式错误。诊断可保留未门控的 set overlap，命名为 `diagnostic_kw_set_f1`，不可代替主任务分。

micro 汇总累加 TP/FP/FN；全局分母为 0 时记 N/A 并单报空目标 exact。缺块/坏块的非空 GT 计 FN；空 GT 的格式失败由主 macro F1/exact、格式率、空目标协议失败率体现。空目标假阳性报告“确实输出了非空候选”的比例，另报不可解析比例，避免把缺块算作正确空预测。

### 9.3 Reward 组合与 sanity cases

保留原 ASR 正文 reward 标度；新增独立 `kw_list_f1` 和 `kw_block_fmt`。推荐 pilot 草案：

```text
R = R_existing_asr(parsed_transcription)
    + 0.5 * kw_list_f1
    + 0.1 * kw_block_fmt
```

0.5/0.1 是初始实验超参，不是已证明最优值；需要按现有 reducer 的真实标度检查后在 dev 上冻结。实现时核对 `asr_measure.py` 聚合入口；只有选用 vote reducer 才同步修改 `asr_vote.py`，不为本实验强制迁移 reward 框架。

| Case（同一参考音频） | 正文 | 列表 | 预期 |
|---|---|---|---|
| 完整正确 GT | 正确 | 正确 | 总 reward 高于同正文缺块结果 |
| 只输出正文 | 正确 | 缺头 | 正文奖励保留，新增列表/格式奖励为 0 |
| 胡列词 | 正确 | 多余项 | precision/F1 下降 |
| 漏 rare word | 正确 | 少一项 | recall/F1 下降 |
| 正文错误但复制正确列表 | 错误 | 正确 | 正文错误仍受罚；列表不能掩盖 WER |
| 空目标输出空块 | 正确 | 空 | 列表与格式得满分 |
| 空目标输出缺块 | 正确 | 缺头 | 列表与格式均 0，区别于合法空 |
| 重复头、坏列表、解释尾部 | 任意 | 不合法 | 主列表/格式为 0，错误内容不能被静默隐藏 |

在同一小批数据打印分项 reward、聚合 reward、baseline、advantage、mask 与来源标记。若列表复制收益压过正文错误代价，先调整权重再训练，不能以正文退化换取漂亮格式率。

## 10. 已落地的实现位置

| 文件 | 实际改动 |
|---|---|
| `recipe/phimm/data/keyword_output.py` | common5k 分词/校验、统一 formatter、严格 envelope parser |
| `recipe/phimm/data/prompts.py` | opt-in 指令与完整联合 GT |
| `recipe/phimm/data/dataset.py` | 目标显式校验、prefix 精确切分、GT round-trip、字段传播 |
| `recipe/phimm/reward/asr_keywords.py` | 共用解析、正文 WER/BWER/UWER 计数、列表与格式 reward |
| `recipe/phimm/reward/keyword_manager.py` | 注册 per-utterance manager，保留完整输出，使用原生终止状态并兼容旧产物回退，隔离候选 metadata 别名 |
| `recipe/phimm/reward/asr_response.py` | opt-in 解析，保留旧接口 |
| `recipe/phimm/main_asr_dapo.py` | 注册新 manager，解析结果接入验证 dumps；eval/remax 入口复用 |
| `recipe/dapo/dapo_ray_trainer.py`、`verl/trainer/ppo/ray_trainer.py` | 显式 train_sampling/baseline purpose |
| `verl/workers/rollout/gt_rollout.py`、`vllm_rollout/vllm_rollout_spmd.py` | train-only、完整分组、完整 GT 编码、EOS、预算检查；超长失败 |
| `recipe/phimm/prepare_ls_keywords.py` | 固定 hash、speaker split、文本去重、音频存在检查、独立 manifests |
| `recipe/phimm/keyword_preflight.py` | 真实 tokenizer round-trip / 长度 / hash 核验 |
| `recipe/phimm/summarize_keyword_eval.py` | 分集合累计 WER/BWER/UWER/NWER 和列表指标，拒绝重复/混 step |

旧 `asr_measure` 的正文 reward 由新 scorer 在剥离合法块后调用；本轮没有使用 vote reducer 或 `main_asr_gen`，无需改写这些独立路径。`rl_dataset.py` 既有 metadata/tokenization 链路复用，实际 dataset 配置链路已验证；GPU processor/vLLM 联动仍待验证。

## 11. 实际产物与配置

配置按用户要求放在 `recipe/phimm/config/v2609_kw/`：根目录 `base.yaml` 保存公共设置，`data/training/` 单独保存 pilot 训练数据，`data/eval/` 单独保存 dev、官方测试与 checkpoint 测试的数据配置；`baseline_eval/` 包含原 prompt / 联合 prompt 两套 base eval 设置，`training/` 包含 ReMax 10k pilot 设置，`evaluation/` 包含训练后 checkpoint eval 设置。入口通过 Hydra defaults 组合公共、数据和执行设置；A0/A1 共用同一份官方测试数据 YAML。完整树、字段说明和执行命令见 [README](../../recipe/phimm/config/v2609_kw/README.md)。

本地清单目录 `/home/boren/data/LibriSpeech/ls_joint_keywords_common5k_v1/` 已存在；`_remote/` 为云端音频引用版本，已上传到云端。配置默认云端根目录为 `az://orngwus2cresco/data/boren/data/LibriSpeech/ls_joint_keywords_common5k_v1`；远端 manifest 与音频可读性必须恢复凭据后确认。

| 正式清单 | 条数 | 备注 |
|---|---:|---|
| train | 120,024 | 仅本地已存在音频；排除 official/heldout 重复正文 |
| train_pilot10k_seed42 | 10,000 | 保留非空/空比例 |
| full dev | 6,096 | 保留；未训练 step-0 baseline 用于筛选 |
| training validation | 1,000 | dev 中 baseline 逐句 BWER 最低；仅此子集选 checkpoint |
| internal_test | 6,131 | 与训练 speaker 不交叉 |
| test_clean | 2,620 | 全部保留 |
| test_other | 2,939 | 全部保留 |
| train_probe / dev_smoke512 | 256 / 512 | 预检用途 |

原始 TSV 中 148,126 条音频本地缺失，builder 使用 `--require-local-audio` 显式排除；这不是全量 LibriSpeech 训练。最终排除 7 条 official 重复正文及 9 条 train/heldout 重复正文，第 4 节表格为排除前原始审计，不代表正式清单数量。

## 12. 实验矩阵、统计与结果报告

| Arm | 模型 | Prompt | 输入 hints | 作用 |
|---|---|---|---|---|
| A0 | 同一原 base | 原转写 | 无 | 正文基线；列表指标 N/A |
| A1 | 同一原 base | 联合输出 | 无 | 只改 prompt 的能力与正文代价 |
| A2 | 新 RL checkpoint | 与 A1 相同的联合输出 | 无 | RL 是否学会抽词与完整格式 |

全部正式评估为真实模型生成，GT 注入数为 0。固定相同音频、语言 hint、正常解码参数与 response 预算；首轮以 greedy 评估可复现性，若运行采样评估则另列解码设置。A0 的列表缺失不是“格式能力差”的公平证据，因为其未被要求输出列表。

可选单段 GT 消融：相同 prompt/reward/更新预算下关闭 GT，明确记录每步示范与采样 token 数的差异；不作为本轮最小实验必需项。它不包括任何 biasing 或跨段实验。

| 数据角色 | 用途 | 报告要求 |
|---|---|---|
| Train-probe | 训练内拟合与 GT 数据路径诊断 | 不能称为独立泛化结果 |
| Dev | checkpoint、权重、长度预算选择 | 报告筛选条件、实际音频覆盖与最优 checkpoint |
| Internal-test | 训练 corpus held-out speaker | 冻结后一次评估，按固定清单报告 |
| Official test-clean | 2,620 条 | WER/CER、列表、协议、空目标，独立一行 |
| Official test-other | 2,939 条 | 同上，不与 clean 合并掩盖变化 |

正文报告现有 p_err/WER/CER；列表报告 micro/macro P/R/F1、集合 exact、非空目标 recall、空目标假阳性；协议报告有效/缺失/坏块/截断率、重复与顺序遵循；成本报告 token/音频时长/推理时延。原正文 keyword/B-WER 可以辅助分析，明确分母，与新列表 F1 分开。

配对比较 A1-A0 的正文变化、A2-A1 的联合输出收益、A2-A0 的正文保持。按 speaker/chapter 聚类策略固定 bootstrap 实现与 seed，至少按 chapter 计算 95% CI；若存在同 speaker 多章相关性，可用 speaker 聚类作敏感性分析。模型/配置选择仅看 dev，官方测试不选阈值。

工程门槛建议在实施准备结束时冻结：协议有效率至少 99%，正文 WER 相对 A0 非劣界暂提议 +0.1 个百分点，并报告 CI；列表要求 A2 优于 A1 的 F1/exact 且空目标没有塌缩。前两者与数值阈值均为计划提议、不是用户已确认的结果；不能事后根据 official 成绩移动门槛。格式过关但 F1 无提升时，结论只能为学会格式，抽词收益未证实。

错误分析固定抽查：缺块、错 rare word、漏词、常用词误报、正文/列表自洽但听错、合法空目标、空目标误报、长答案截断。保存 raw/parsed/reference/分项分数/finish_reason，不用人工修复后的输出作主指标。

## 13. 必须覆盖的测试与执行顺序

| 阶段 | 工作与测试 | 进入下一阶段的硬条件 |
|---|---|---|
| P0 本轮 | 统一文档、真实数据和代码核验、流程图 | 已完成；代码与 CPU 预检状态见第 14 节 |
| P1 数据 | builder 重现 common5k；split/去重；原始 GT list 映射；空/缺标注；音频可读与来源 hash | 正式 manifest 存在；数值与审计差异有解释；测试不混入训练 |
| P2 协议 | 实现 formatter/parser/真实调用点；单/多词/空块/坏块；旧任务回归；prefix 与 EOS | 全部有效 GT round-trip，正文不含合法列表，新任务禁止 text fallback |
| P3 训练链路 | 实际 tokenizer/processor 小批贯通；GT purpose、group、shard/pad/mask；greedy/val/eval 零 GT；奖励排序 | GT 长度完整，reward 与 token/log-prob 对齐；正确示范有可观测信号 |
| P4 Pilot | 10k 或更小，固定 dev/probe，观察 reward/advantage/KL/长度 | 正文与列表有效学习；不因空列表投机或过滤失去信号 |
| P5 冻结测试 | 固定 checkpoint/prompt/reward/protocol，A0/A1/A2 独立生成 | internal/clean/other 分别报告；无 GT 泄漏；包含失败样例和限制 |

测试最少包含：

- 例一、例二、例三的 formatter→tokenizer→decode→parser round-trip。
- 空尾部保留、尾部空白被 decoder 去掉但正常 EOS、length 截断、缺 finish metadata 四种不同状态。
- 单行头错误、重复头、空项、重复项、额外说明、短语/CSV 引号不支持，语法错误无静默修复。
- 参考正文/关键词/完整答案不串字段；`None` 与 `[]` 不混同；已有 keywords 不被清理流程覆盖。
- 新 reward 和所有 eval 调用共享 parser；正文正确但缺块的新增任务分严格低于完整 GT；空目标的缺头不拿满分。
- GT 只在 train sampling；1/8 示范分组正确；greedy 不被注入；完整 completion 超长 fail fast。
- 2609 旧任务和 2607 旧任务的兼容；新任务中音频占位符、language hint、prefix 均不重复。

## 14. 当前证据与待执行事项

已完成独立远程分支/worktree、联合输出实现、四套配置、正式 manifests。A0/A1 与 training/dev 的实际 Hydra 数据转换链路通过；真实 tokenizer 对 pilot/dev/internal/clean/other 共 27,786 条 GT round-trip 与长度校验全部通过，最大 182 tokens（含 EOS），预算 768。证据见 `reports/asr_keywords_plan_20261023/` 的 manifest metadata、dataset config preflight 和 tokenizer preflight JSON。

本地主回归 299 passed / 2 deselected；两项排除是已有 torchcodec/FFmpeg 环境不兼容及旧 bad-format 断言不符。定向测试另覆盖 parser/GT/reward/汇总；实际 GT 辅助函数及温度对齐测试 56 passed。CPU rollout 测试通过 namespace 加载实际 helper，避开本机 torchvision GPU registry 导入错误，未模拟 helper 算法。这些不是 GPU 端到端验证。

Brix/Blob 已恢复，云端引用版 manifests 已上传；主节点已缓存全部 5,559 条测试音频，训练/dev/internal 各抽样读取通过。A0 job `raysubmit_8QYAUSqkEcEr7Ji8` 已在 `verl-n2-i10`（2×8 H100）启动，代码快照 `5307f0132`，暂无 WER/BWER 结果。父分支最新 `d34b3ba28` 已无冲突合并（merge `45c370aab`），后续 job 同步新版本。继续按 A0 → A1 → pilot → A2 执行；GT log-prob/mask、实际吞吐与模型质量仍待 GPU 验证。实时状态见 execution_status.json 和 Ray，已配置 30 分钟跟进。

评估统一使用 unit-cost jiwer 与单词边界匹配，BWER/UWER 错误分区总和等于 WER；NWER 为 UWER 别名。不能直接与旧加权 alignment/子串匹配的历史 BWER 混比。最终结果按集合累计计数，分母 0 报 N/A。

完成实验仍需：**无 GT 泄漏的实际 RL 运行，以及 LibriSpeech 正文/关键词/协议三组模型结果**。连续 biasing 留待后续，不作为本轮依赖。


### 2026-10-23 训练验证集缩小（覆盖先前 full-dev 训练验证安排）

用户要求训练期间 validation 使用 baseline BWER 最低的 1,000 条。选择源限定原有 dev 的未训练 step-0 结果，禁止使用 test-clean/test-other/internal-test。按 (BWER, WER, ID) 排序，排除无关键词分母的样本，冻结名单并记录输入/输出 SHA256 与原始计数。实现为 `recipe.phimm.select_keyword_validation`，配置为 `data/eval/ls_dev_lowbwer1000.yaml`；full dev 配置保留。训练入口加载新子集，使用独立实验名 `remax_ls_joint_pilot_dev1k` 与数据源，避免与历史 full-dev 指标混合。100 steps、save10、test20、10k train、n8 (1 GT+7 samples) 不变。该子集为容易样本，存在基线 BWER 为零的地板效应；完整 held-out 测试继续用于最终结果。

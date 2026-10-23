# ASR transcription + keywords：RL 实验方案（待讨论，未实现）

日期：2026-10-23。代码调研基线：`aa0f6d523`。

隔离分支：`boren/asr-transcription-keywords`，跟踪同名 origin 分支。
工作目录：`/home/boren/.codex/worktrees/asr-transcription-keywords/verl`。
本文件仅记录方案；不修改功能、训练配置或启动训练。

## 1. 已确认的目标与输出协议

保留原有 transcription 格式、语言头和 VERBATIM/LEXICAL 模式，在完整转写的末尾添加一次关键词块。第一阶段针对现用 2609/2611 模型格式，2607 保持原有行为；如后续需要，对 2607 单独扩展和验证。

```text
<src=English><tgt=English>
<VERBATIM>
Microsoft reported revenue growth.
<keywords>
Microsoft, revenue growth
```

用户已确认：

- `<keywords>` 独占一行，下一行使用英文逗号分隔关键词。
- 无关键词时保留头和空的关键词行，例如正文后追加 `\n<keywords>\n\n` 再结束输出；解析时须允许 tokenizer/decode/strip 去掉末尾空白，但不能把完全缺失头当成合法空列表。
- 使用现有 keywords 标注，默认只列参考转写中实际出现的词或短语。

建议细则：

- 关键词按参考转写中第一次出现的位置排序，稳定去重；同一短语重复出现只列一次。
- 输出保留参考转写的表面形式和大小写；匹配可使用现有语言相关 normalization，但不能只用不带词边界的 substring 匹配。中文等语言采用对应的字符/分词规则。
- 多语言多段 transcription 共用末尾一个 keywords 块，不给 keywords 添加 src/tgt 头。
- `<keywords>` 是普通文本协议头，不增加 tokenizer vocabulary 或模型 embedding。
- 含英文逗号的短语建议使用 CSV 引号，例如 `Microsoft, "Washington, D.C."`；使用标准 CSV writer/reader，不能直接 `split(',')`。这是提议，最终实现前仍可选择过滤此类样本以简化第一版。
- 标注缺失与人工确认空列表分开处理。缺标注不能自动视为“没有关键词”。

## 2. 代码调研结论

| 位置 | 当前行为 | 本次需要处理的原因 |
|---|---|---|
| `recipe/phimm/data/prompts.py:get_task_prompt` | 按 task 生成原 prompt，支持语言提示 | 在原 prompt 上增加结构化输出指令 |
| `prompts.py:get_task_output` | 接收 text/components/version，不接收 keywords | 当前 GT 没有关键词块 |
| `recipe/phimm/data/dataset.py:add_task_info` | 同时生成 prompt、prefix、gt_output，随后剥离已放进 assistant prefix 的头 | 必须同步构建新 prompt 和完整 GT completion，避免 prefix 重复 |
| `dataset.py:verl_format_ds` | `ground_truth=text`，`gt_output` 使用格式化结果，缺省回退 text | 必须保留纯文本 reference，同时禁止新任务静默回退成纯转写 GT |
| `dataset.py:_extra_info_value` | keywords 缺失时使用 `[""]` | 必须在任务构建前区分缺失、空标注、空字符串占位 |
| `dataset.py:process_ds` | 固定先 remove_brackets，再 clean_tagged_text | entity 标签的尖括号可能先被移除，后续无法可靠提取 |
| `dataset.py:clean_tagged_text` | 从标签抽取 keywords，并覆盖目标字段 | 原来已有 keywords 时，可能被无标签文本产生的空列表覆盖 |
| `recipe/phimm/reward/asr_response.py` | 返回 src/tgt/text segments；2609 最后一段会吸收 keywords 尾部 | 新元数据会污染 WER/CER、标点、数字等评分 |
| `recipe/phimm/reward/asr_measure.py` | word_kw/char_kw/keyword 评估正文中标注关键词的识别误差 | 这些指标不评估新输出的关键词列表，需要新增独立指标 |
| `recipe/phimm/reward/asr_vote.py` | 从 `_parse_response` 取指标，对照 greedy baseline 加权 | 新指标必须接入真实解析结果；仅添加 YAML 指标名会报错 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 非 validation 时，每 n 条取一条 gt_output 进行 teacher forcing | 已有 GT 能力可以复用，但须保护新任务格式、GT 长度和使用场景 |
| 同上 | GT 文本 tokenize 后直接截到 response_length-1，再添加 EOS | 新 keywords 位于末尾，可能被静默截掉，形成错误示范 |
| 同上 | GT 条件未排除 `do_sample=False` | ReMax greedy baseline 也可能注入答案，须隔离 |
| `verl/trainer/ppo/rollout_log_probs.py` | GT 行使用 actor 对齐温度下的 log-prob；保留 GT mask | 应复用，并验证新协议的全部 token 都覆盖 |

调研时直接调用当前 `get_hyp_text(..., version=2609)`，输入上面的新格式，返回值仍包含 `\n<keywords>\nMicrosoft, revenue growth`。这确认了解析污染问题；本轮未修改该行为。

ReMax 的 generator 输入在 trainer 中先按 n 重复；baseline 使用该 batch 的副本。除 GT 开关外，实施时还需核对 baseline 输出数量、去重复位置和原始样本 uid 对应关系，不能假定“每个 worker 的第 n 行”始终等价于一个完整 prompt 组。

## 3. 数据与目标：先建立一致的 contract

建议新增显式可选配置，例如 `add_task_info.output_keywords: true`，默认关闭，不通过隐式 task 名字匹配启用。训练、validation 和 reward 都使用同一 task contract / 样本标志。

目标构建顺序：

1. 从原始样本读取 text/components 和原始 keywords；保留标注来源与是否有标注的状态。
2. 若需要实体标签提取，必须在破坏标签的文本清理前完成；明确现有 keywords 与提取结果的合并规则，不能空覆盖。优先新增显式处理路径，避免全局改变旧 recipe 行为。
3. 完成正文清理、chunk 切分等影响 reference 的操作，再基于最终 reference 过滤关键词。chunk 只能带本 chunk 内完整出现的关键词。
4. 清除 None/空白/空占位，按语言归一化匹配，去重，按首次出现排序；输出使用最终 reference 中的表面形式。记录过滤原因和数量。
5. 用同一份 `target_keywords` 构建 `gt_output` 和 reward 的关键词列表目标。保留原始标注以便审计；原有正文 keyword reward 的标注语义不静默改变。
6. 统一生成 prompt、格式化 transcription、keywords 块；随后仅剥除确实已包含在 assistant prefix 中的前缀。
7. `reward_model.ground_truth` 始终是纯正文；`reward_model.gt_output` 是可直接接在 assistant prefix 后的完整 completion。
8. 在 extra_info 保留输出任务开关、关键词列表目标、标注有效性、协议版本等必要字段；更新 Arrow schema 和 extra_keys，避免混合数据加载时字段丢失或类型冲突。

新任务必须具备可靠标注。已标注空列表参与空列表训练；缺失标注的样本建议第一阶段保留在原转写任务，或排除在新任务之外，不能批量制造空列表负例。

数据审计按 OpenML、Earnings、LibriSpeech 和每个语言分别输出：样本数、缺标注率、真实空列表率、非空率、匹配失败率、关键词数量分布、逗号/换行/标签保留字、GT token 长度 p50/p95/p99/max。只读取 manifest/已有缓存即可先审计，不需要为此下载全部音频。

已知 recipe 的 4:1:2 混合比例只是现有实验配置，不代表关键词监督覆盖比例。未读真实 manifest 前，不宣称这些源均有完整标注。先选审计合格子集做第一轮。

## 4. Prompt 与 GT output 同步修改

保留既有任务、语言提示、音频占位符、VERBATIM/LEXICAL 指令，再追加统一指令。例如：

```text
Also output keywords from the transcription. After the complete transcription,
write <keywords> on a new line, then a comma-separated keyword list on the next
line. If there are no keywords, leave that line empty.
```

若采用 CSV 引号，补充短语内逗号的引用规则；若数据标注实际上代表 rare words 或 named entities，应按审计结果为 “keywords” 给出一致的语义定义。任意来源标注政策不一致时，仅有 “output keywords” 可能无法学成可泛化任务。

不能将 ground-truth keywords 值拼入用户 prompt，以免变成抄写目标。GT 只放进 teacher-forced response 和训练评分元数据。

`get_task_output` 接收显式 keywords/任务选项：先调用原 transcription formatter，再追加唯一的关键词块。plain ASR、lang_asr、VERBATIM、LEXICAL、multilingual components 共用追加逻辑；第一阶段 recipe 限定 2609，2607 不隐式启用。

GT 必须满足两个可验不变量：

- `parse(prefix + gt_output).transcription` 等于最终 reference 对应的原任务输出正文。
- `parse(prefix + gt_output).keywords` 等于 canonical target_keywords，格式有效。

## 5. 统一解析与防止奖励漏洞

建议新增结构化 envelope 解析入口，返回 transcription segments、keywords、头是否存在、格式是否有效、错误原因、未解析尾部。保留现有 `parse_task_output` 的 list-of-segments 接口，避免一次修改所有调用方的返回类型。

先识别末尾完整的关键词块，再把 transcription 部分交给旧 segment parser。合法块不参与语言 segment 数量、WER/CER、标点/大小写、数字指标。`get_hyp_text`、ASR measure/eval、inhouse eval 和 generation 展示走同一个规则。

严格区分：缺少关键词块、合法空块、非空列表、重复头、位置不正确、多行列表、未闭合 CSV 引号、转写出现在关键词块之后、模型额外解释、截断。缺少/错误块不能被“宽松解析成功”冒充为新任务格式正确。

只在新任务启用关键词协议；旧任务遇到类似文本继续保留旧评分语义。新任务出现坏尾部时，保留 diagnostics，采用明确失败分数/格式门控；不能通过在 `<keywords>` 后放任意幻觉正文来隐藏 ASR 插入错误。合法候选词仍须满足长度/单行约束，并用正文一致性及列表 precision 约束。

GT 过滤使用 reference；预测列表评分不能先用 reference 过滤预测，否则模型乱输出的词会消失、precision 被虚增。预测去重前应记录重复并给予格式/重复惩罚，不能靠重复词刷分。

## 6. Reward 设计

保留现有正文 word/char、插入/删除、word_kw、lang、fmt；新列表使用独立命名，避免改变已有 `keyword` 指标的含义。

建议新增：

- `kw_list_precision / recall / f1`：按归一化后的完整短语一对一匹配，顺序不影响集合评分。
- `kw_list_exact`：预测列表与目标集合完全一致，用于监控。
- `kw_block_fmt`：头、单行 CSV、重复头等格式正确性。
- `kw_in_transcript`：预测列表项目是否在预测 transcription 中出现，作为一致性指标。
- `kw_duplicate_rate`、缺块率、空块率、GT 截断/跳过数作为诊断。

空目标约定：有效空块得列表 F1=1；输出非空列表得 0；缺块得 0 且格式失败。非空目标预测为空得 0。缺标注样本不参与列表 reward 或分母。

初始 reward 提议是现有 ASR 项 + 较小的列表 F1 和块格式权重，F1 先用绝对分数，precision/recall 单独记录。权重在离线案例上校准后再确定，确保添加正确列表的 GT 相对纯转写 greedy 有正 advantage，同时随机列词的得分低。不能只靠“有 tag”给高奖，也不能只奖励 recall。

如沿用 asr_vote，扩展 `_parse_response` 的指标并验证 greedy 与 sampled 使用同一组 measure 定义；新任务格式门控应覆盖对应的新任务得分，旧任务不受影响。检查 reduce.total 与新权重一致，日志拆开正文收益和列表收益。

## 7. GT rollout：保证它真的是正确示范

复用已有 vLLM teacher-forced scoring 和 actor log-prob 温度对齐，不另起一套离线 token 拼接路径。

1. 仅训练 sampling 路径允许注入；validation、standalone eval、ReMax `do_sample=False` 的 greedy baseline 一律不注入 GT。
2. 每个原始 prompt 的 n 条训练 trajectory 中先用 1 条 GT，余下 n-1 条真实 sampling；若 n=8，初始名义占比 12.5%。保持总轨迹数不变，实际占比须在 sharding/dedup/filter 后统计。
3. 用 prompt uid/明确组索引保证分组，不依赖不经验证的 worker-local 行步长；保证 mask、reward metadata 与 log-prob 随重排对齐。
4. 新任务强制存在合规 `gt_output`，不允许回退到纯 `ground_truth`。格式不合规提前报错，标出 source/id。
5. tokenize 时明确 special-token 策略，正文+完整关键词块+EOS 必须全部容纳。超过 response budget 时提前统计并过滤/报错，或经长度审计调整预算；禁止静默截断正确示范。
6. 确认 prompt + GT + 评分所需的额外 token 不超过 engine context，assistant prefix 不重复，EOS 只出现一次且使用现有模型正确的结束方式。
7. GT mask/log-probs 覆盖全部 response token，包括关键词头、列表和 EOS；保留现有 finite 检查、actor temperature 对齐、padding/response mask 行为。
8. 检查去重、group filtering、advantage 归一化、ReMax disagreement mask 等是否使 GT 无效，分别记录 GT/sample 的 reward、advantage、有效训练 token、KL 和 log-prob。

这里的 GT 是人为给定轨迹，属于带示范的 RL，teacher forcing 评分不会使它成为 policy 自采样。它能提供正确输出的训练轨迹，但不能单凭“有 GT”保证模型会学会；需要验证 GT 获得正向训练信号以及无 GT 推理时的格式和列表质量。第一阶段沿用现有算法，若 GT 信号被 KL/advantage 压制，再讨论比例调整或辅助监督项，不在本次方案中默认加入 SFT。

## 8. 修改范围与实施顺序

| 阶段 | 预计修改位置 | 可审核产物 / 进入下一阶段条件 |
|---|---|---|
| A 数据审计 | 新增轻量审计脚本与报告 | 各源标注覆盖、清理前后示例、长度统计；明确关键词语义 |
| B 协议和目标 | prompts.py、dataset.py、必要的 rl_dataset schema | 新任务 opt-in；prompt/GT/metadata 一致；原 recipe 不变 |
| C 解析 | asr_response.py 及共享调用点 | 正文和列表分离、错误状态明确、旧接口兼容 |
| D Reward | asr_measure.py、asr_vote.py 和新任务 scorer/配置 | 正确/缺失/错误/作弊案例的 reward 与 advantage 排序符合预期 |
| E GT 安全性 | vllm_rollout_spmd.py、必要的 trainer metadata/gt_rollout 辅助函数 | train-only 注入、greedy 纯生成、完整 GT 与正确组对应 |
| F 独立实验配置 | 新 train/val 数据 recipe 与独立 experiment YAML | resolved config 审核，所有数据分量显式启用或关闭新任务，缓存版本更新 |
| G 小规模验证 | CPU integration + 后续指定 GPU 节点 smoke run | 通过后才考虑完整训练 |

修复预处理覆盖问题时优先限定新路径；GT baseline 隔离若修复到共享代码，需要增加旧任务回归验证并单独说明其全局影响。

新配置使用新的 experiment name、checkpoint/output 目录、dataset cache key。固定旧 cache_name 的数据集必须显式换名，不能重新载入含旧 prompt/gt_output 的已格式化缓存。

## 9. 测试与验收

有针对性地扩展现有 `tests/recipe/phimm/data/test_prompts.py`、`test_dataset.py`、`tests/recipe/phimm/reward/test_asr_response.py`、reward tests、`tests/workers/rollout/test_gt_rollout_on_cpu.py` 和 `tests/trainer/ppo/test_gt_rollout_log_probs.py`。

测试矩阵：

- 单/多关键词、多词短语、重复、None/空列表/缺失、逗号和引号、中英文/多语言、VERBATIM/LEXICAL、prefix_prob 0/1。
- clean_tagged_text 与既有关键词合并、标签清理先后顺序、最终 reference 过滤、混合 Arrow schema 和缓存失效。
- formatter → parser round trip；原转写附加合法块后，正文 hyp、WER/CER/lang/fmt 及数字指标不变。
- 缺头、重复头、CSV 错误、多行/额外尾部、截断和幻觉隐藏；坏格式不得取得满分。
- GT completion 完整包含两部分；超长 fail/filter 而非截断；EOS、padding、mask、全部 token log-prob 对齐。
- train sampling 每组恰好 1 GT；greedy/val/eval 为 0 GT；跨 worker、batch padding、重排、去重后关联保持正确。
- 离线 reward 验证：正确列表 > 缺列表；匹配正文的关键词 > 幻觉关键词；空目标空块 > 胡乱列词；GT 相对实际 greedy 有正向信号。

GPU smoke 需后续选定节点后进行：先小批量 rollout，人工查看 prompt、reference、prefix、GT completion、解析结果、sample/greedy/GT 标志，再跑少量 update。通过标准包括 loss/log-prob 有限、GT 有效训练 token 非零、baseline 无答案注入、validation 无 GT。训练效果不承诺固定步数达到。

验证报告按数据集分别列正文 p_err/WER/CER（继续作为主要质量目标）和列表 P/R/F1/exact、协议成功率，不合成一个掩盖正文退化的数字。关键词指标只在有可靠标注的数据上计算。已有 benchmark 可继续评估旧 prompt，同时用新 prompt 评估正文兼容性；缺关键词标注的 benchmark 不虚构列表质量。

建议实验对照：原模型+原 prompt；原模型+新 prompt；新任务 RL 不含 GT；相同条件新任务 RL 含 1/n GT。若资源有限，先完成 base+新 prompt 与含 GT 小试验，不默认启动全部对照。

## 10. 下一次讨论的决策

输出头、空列表格式、使用已有标注并按转写过滤已确认。下一步建议优先确定：

1. 先从哪些标注可靠的数据开始；是否以现有三源 recipe 为审计起点。
2. keywords 在各源中的含义是否足够一致，是否先限定 rare words / entities 某一类。
3. 是否接受 CSV 引号处理短语内逗号，以及缺标注样本继续使用原转写任务的安排。
4. 第一版是否复用 n=8、1 GT + 7 sampled；列表 reward 权重以离线结果校准。

本轮不启动训练，不修改模型/解析/GT rollout 实现。后续所有实现与实验准备继续在上述独立 worktree 和远程分支中进行。

# LibriSpeech noisy + RIR v1

生成配置：`recipe/phimm/config/data/cache/gen_ls_raw_rp_edge_nodigits_aug_audio_rir.yaml`。
训练配置：`recipe/phimm/config/data/train_data/noisy/ls_rir_verb_langhint.yaml`。

目标目录：
`az://orngwus2cresco/data/boren/data/verl/gen_cached/ls_raw_rp_edge_nodigits_aug_audio_rir_but_v1/`。
`data.jsonl` 仅在全部样本生成成功后发布；目录中只有音频或 resources 不代表完成。

## 样本和增强

以原缓存 `ls_raw_rp_edge_nodigits_aug_audio/data.jsonl` 的 20,424 个样本为集合，
读取其中 `source_audio_path` 指向的原始 LibriSpeech 语音，保留原始文本和 ID。
原 manifest 的 MD5 为 `a7601b824e4729ad24a369a39722a130`。原缓存已经进行数字过滤，
本次保持完全相同的样本集合，不重新运行上游模型挖掘或数字过滤。

顺序为 **变速 → RIR 卷积 → 加性噪声 → 必要时峰值衰减 → PCM16 WAV**。
每条均执行 1–2 倍变速、RIR 和 10–12 dB 加噪。变速同时改变音高。
RIR 和加性噪声均使用 BUT Speech@FIT ReverbDB 的真实测量资源。
原作者页面明确标注数据集采用 **CC BY 4.0**，允许商业使用，需要署名。
本次不使用 SLR28 混合真实 RIR 池，其中 RWCP 原始页面标注仅研究和开发使用。
原 AED 噪声的许可证未核实，因此以 ReverbDB 的真实环境噪声替换。
这意味着与旧缓存相比，同时改变了混响和噪声来源，不能作为只改变 RIR 的严格消融。

来源和许可证据：

- 数据集作者页：<https://speech.fit.vut.cz/software/but-speech-fit-reverb-database>
- 原始档案：<https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz>
- 作者提供的说明：<https://merlin.fit.vutbr.cz/ReverbDB/read_me.txt>
- CC BY 4.0：<https://creativecommons.org/licenses/by/4.0/>
- LibriSpeech 的 CC BY 4.0 声明：<https://www.openslr.org/12/>
- 论文：<https://arxiv.org/abs/1811.06795>，DOI `10.1109/JSTSP.2019.2917582`。

仅从档案中路径组件为 `RIR` 的 WAV 提取 RIR；从 `silence` 目录选择 100 条
真实环境噪声，跨房间分配，每个麦克风/声源布置最多一个版本。
排除元数据中标有 radio、music 或第二声源的布置，避免把播放语音当背景资源。
`resources/rirs.jsonl`、`resources/noises.jsonl` 保存原档案成员、房间、许可和 SHA256。
`resources/notices/` 保留作者页、原始说明、许可文本及归属说明。
作者网页的 CC BY 4.0 声明和说明文件自身的 Apache 2.0 头分别保留，不混淆其适用对象。

复用 `scripts/augment_audio.py` 的 RIR 实现：随机选择一个麦克风通道，
单位能量归一化，以最大脉冲位置对齐，裁剪到变速后语音长度，匹配输入 RMS。
因此最后的混响尾部会被裁去；此对齐约定不等同于估计真实直达声到达时间。
SNR 定义为混响后语音相对加性噪声的整句功率比。噪声不另做 RIR 卷积。
RT60 没有额外采样或测量；混响特征来自各真实 RIR。

RIR 随机流使用 `20261027 + 原样本索引`。原缓存每条的变速、噪声槽位、
SNR 和噪声裁剪随机流使用 `0 + 原样本索引`，与 RIR 随机流分开。
原索引取自旧 `audio_path` 文件名的八位数字前缀，数字过滤后的索引不一定连续。

## 原噪声池快照

原噪声清单 URL 是可变资源。本次根据旧 manifest 恢复其生成时使用的 100 个噪声
文件及顺序，快照写到 `resources/original_noise_paths.jsonl`。
恢复方法：对每条原索引重放 speed 概率、speed 均匀采样和 noise 概率的随机调用，
以 `randrange(100)` 得到噪声槽位，用旧 manifest 的 `noise_path` 填充。
全部 20,424 条均满足同一槽位对应同一文件、全部 100 槽位已覆盖，
且接着采样的 SNR 和原 speed 均在 1e-8 容差内匹配旧记录。
生成程序在写音频前再次校验全部原随机参数。原 100 个噪声槽位与新的
100 个 ReverbDB 噪声一一映射，保留变速和 SNR 数值及随机流，但噪声内容和
实际裁剪位置随新素材变化。`parent_noise_path` 记录旧噪声路径，仅用于追溯。

## 生成和验证

依赖：NumPy、SciPy、SoundFile、blobfile、PyYAML。

```bash
python scripts/build_ls_rir_dataset.py --work-dir /path/to/ls_rir_v1_build
```

可用 `--source-local`、`--noise-local`、`--rir-manifest-local` 和 `--rir-local-dir`
指定已验证的本地缓存；逻辑配置及数据指纹仍使用资源内容。
`--max-examples 8` 仅生成小批检查文件，不发布最终 manifest。
中断后使用相同 work-dir 和输入可从成功行恢复。
变更代码或输入会改变指纹，需使用新 work-dir。
最终 manifest 已存在时拒绝覆盖。

每条写出前校验实际 SNR、PCM16 编解码、采样率、长度和有限数值，记录 WAV SHA256。
旧模型的 `response/raw_response/wer/edge_wer` 移入 `source_generation`，
表示上游挖掘的来源记录，避免作为新混响音频的解码质量使用。
原文本保留；训练加载时重新生成 `lang_asr_verb`、English language hint、
无 prefix 的 prompt/ground truth，`data_source` 沿用 `noisy_ls` 便于现有 reward 映射使用。

最终 `summary.json` 包含完整音频小时数、RIR 使用数、SNR 误差和内容指纹。

```bash
python -m pytest tests/scripts/test_build_ls_rir_dataset.py tests/scripts/test_prepare_but_reverbdb_resources.py tests/scripts/test_augment_audio.py -q
```

## 资源清单复现

先下载上述原作者档案，校验 SHA256：
`d0d14cdd626b4b05401027c5a774b4f24369f4ea38f57480c6fc6e04e9f3f5b5`。
将其完整解压到独立目录，例如：

```bash
mkdir -p /path/to/but_extracted
rapidgzip -d -P 8 --verify -c BUT_ReverbDB_rel_19_06_RIR-Only.tgz | tar --no-same-owner -xf - -C /path/to/but_extracted
python scripts/prepare_but_reverbdb_resources.py \
  --extracted-dir /path/to/but_extracted \
  --work-dir /path/to/but_resources
```

普通 `tar -xzf` 同样可用。准备脚本检查所有被选文件的采样率、声道、有限值和正功率，
保存实际音频字节 SHA256、原档案成员路径及房间。
`selection.json` 记录跨房间的数量和被排除的噪声布置。
发布资源时按各清单的 `audio_path` 上传对应的 `rirs_final`、`noises_final` 文件，
全部成功后再发布 `rirs.jsonl` 和 `noises.jsonl`。源说明和归属文件随数据保留。

## 2026-10-27 生成结果

已在 `verl-new-n2-i0-0` 完成，退出码 0。
共 **20,424 条，46.4057 小时**；2,325 个 RIR 和 100 个环境噪声均有使用，覆盖 9 个房间。
逐条核验了全部远端 WAV 的数量和大小，64 条随机回读 SHA256、采样率、长度和信号检查通过。
正式 `create_datasets` 从远端最终 JSONL 读取并生成 20,424 条训练行；节点上 38 项测试通过。
原始样本 ID、文本、原音频路径、变速和 SNR 数值保持一致；噪声素材按许可要求替换。

- Manifest SHA256：`dd928db18d7b61ac63d989fe200293529c38a653d350539904b2ebad39eb7b61`。
- 生成指纹：`95552c8c16997e3fcba51ac9d85e287aceb0a7d976ca4ca8fd301a7861a505f7`。
- 执行代码快照：`8e3ce05029c373612080f9e01fef2c3f3bf17c5e`。
- 量化前 SNR 最大误差：`6.19e-07 dB`。
- 完整验证：目标目录下 `validation.json`；汇总：`summary.json`。
- 本机验证副本：`/home/boren/data/ls_rir_build_20261027/but_validation.json`。

运行使用独立 checkout `/root/code/verl-ls-rir-v1`，原 checkout HEAD 仍为
`e1049fbbebb5deed9e925d4b98eb7eccbc0c139c`，未改动原训练 checkout。

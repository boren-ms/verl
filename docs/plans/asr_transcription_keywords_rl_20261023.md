# ASR 转写与关键词联合输出：当前计划入口

当前唯一实验设计见 [统一计划 v4](asr_keywords_joint_output_detailed_20261023.md)。该文档包含 LibriSpeech 数据与旧 rare-word 配置对照、统一 common5k 标注、精确输出例子、GT/prefix/token 生成、parser 失败策略、RL/reward、实现位置和验收测试。

本文件旧稿已由 v4 替代，历史内容可从 Git 查看。连续 biasing 仅保存在[后续备忘](asr_keywords_biasing_followup_20261023.md)，暂不开展。

分支：`boren/asr-transcription-keywords`。独立 worktree：`/home/boren/.codex/worktrees/asr-transcription-keywords/verl`。当前仅文档与只读核验，尚未实现或训练。

"""Build the final LibriSpeech report from verified corpus metrics."""

import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
trained = json.loads((OUT / "metrics.json").read_text())
baseline0 = json.loads((ROOT / "baseline_a0/metrics.json").read_text())
baseline1 = json.loads((ROOT / "baseline_a1/metrics.json").read_text())
verification = json.loads((OUT / "verification.json").read_text())
selection = json.loads((ROOT / "checkpoint_selection.json").read_text())
names = {"ls_internal": "internal-test", "ls_clean": "test-clean", "ls_other": "test-other"}
sources = list(names)
error_keys = ["WER", "BWER", "UWER"]
list_keys = ["kw_micro_precision", "kw_micro_recall", "kw_micro_f1", "kw_exact", "kw_format_success", "kw_empty_false_positive"]

comparison = {}
for s in ["ls_clean", "ls_other"]:
    comparison[s] = {}
    for bname, baseline in [("A0_original_prompt", baseline0), ("A1_joint_prompt", baseline1)]:
        comparison[s][bname] = {
            k: {"baseline": baseline[s][k], "trained": trained[s][k],
                "delta_percentage_points": 100 * (trained[s][k] - baseline[s][k]),
                "relative_error_change_percent": 100 * (trained[s][k] / baseline[s][k] - 1)}
            for k in error_keys
        }
(OUT / "baseline_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

with (OUT / "all_metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "dataset", *trained["ls_clean"].keys()])
    writer.writeheader()
    for model, metrics in [("A0_original_prompt", baseline0), ("A1_joint_prompt", baseline1), ("A2_checkpoint100", trained)]:
        for source, row in metrics.items():
            writer.writerow({"model": model, "dataset": names[source], **row})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
x = np.arange(2)
for ax, key in zip(axes, error_keys, strict=True):
    for i, (label, metrics, color) in enumerate([
        ("A0: original prompt", baseline0, "#94a3b8"),
        ("A1: joint prompt", baseline1, "#60a5fa"),
        ("A2: trained step100", trained, "#e37b32"),
    ]):
        values = [100 * metrics[s][key] for s in ["ls_clean", "ls_other"]]
        bars = ax.bar(x + (i - 1) * .25, values, .23, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xticks(x, ["test-clean (2620)", "test-other (2939)"])
    ax.set_title(("UWER / NWER" if key == "UWER" else key) + " | lower is better")
    ax.set_ylabel("Corpus error rate (%)")
    ax.set_ylim(0, max(100 * trained[s][key] for s in ["ls_clean", "ls_other"]) * 1.24)
    ax.grid(axis="y", alpha=.15)
axes[0].legend(fontsize=9)
fig.suptitle("LibriSpeech held-out tests | transcription regression after joint keyword training", fontsize=16, fontweight="bold")
fig.savefig(OUT / "baseline_comparison.png", dpi=180)
fig.savefig(OUT / "baseline_comparison.pdf")
plt.close(fig)


def pct(v):
    return "undefined" if v is None else f"{100 * v:.3f}%"


def md_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


transcription = [[names[s], trained[s]["utterances"], *[pct(trained[s][k]) for k in error_keys]] for s in sources]
keywords = [[names[s], *[pct(trained[s][k]) for k in list_keys]] for s in sources]
baseline_rows = []
for s in ["ls_clean", "ls_other"]:
    for label, metrics in [("A0 原始提示", baseline0), ("A1 联合提示", baseline1), ("A2 训练100步", trained)]:
        baseline_rows.append([names[s], label, *[pct(metrics[s][k]) for k in error_keys]])

summary = """模型学会了输出关键词列表，但本次100步训练未改善正式测试集的转写准确率。test-clean / test-other 的 WER、BWER 均高于 A0 和 A1；非关键词 UWER 略有改善。关键词列表 micro precision 仅为33.08%–48.27%，空关键词样本的误报率为92.73%–98.46%，说明格式学习成功，但关键词选择仍明显过量。"""
report = "# LibriSpeech 转写 + <KEYWORDS> 实验最终报告\n\n" + summary + "\n\n"
report += "## 完成状态与来源\n\n"
report += "- 训练：100/100步成功，任务 `raysubmit_6PPiWDZg5QJ3zagk`。\n"
report += "- 选择：仅使用固定 dev1000 的最小 corpus WER（同分取较晚步），选中 checkpoint100，p_err=0.007720556342711737；测试集不参与选择。\n"
report += "- 导出：LoRA rank256/alpha512，176对适配器合并；781个有限值BF16张量，与匹配基座的键和形状一致，训练目标权重差异已确认。\n"
report += "- 评估：`raysubmit_xJU7mCyiJTL2XtAy` SUCCEEDED；双节点、greedy、无GT注入，仅评估 internal-test / test-clean / test-other。\n"
report += "- 输出：11,690条、11,690个全局唯一ID，与三个冻结manifest的ID/参考转写/关键词/来源哈希完全匹配；逐条验证 n_err=nb_err+nu_err、n_ref=nb_ref+nu_ref。\n"
report += "- JSONL 内 step=0 是独立评估的计数器，实际模型为训练后的 checkpoint100。原生vLLM结束原因已保留。\n"
report += "- 训练与评估均成功结束，16块GPU已释放。\n\n"
report += "## 训练模型的转写指标\n\n" + md_table(["数据集", "条数", "WER", "BWER", "UWER / NWER"], transcription)
report += "\nWER/BWER/UWER 均由语料级错误计数之和除以参考词数之和得到；NWER 是 UWER 的别名。\n\n"
report += "## 关键词列表指标\n\n" + md_table(["数据集", "micro P", "micro R", "micro F1", "exact", "format", "空目标误报率"], keywords)
report += "\nP/R/F1 为语料级 micro 聚合；exact/format 为逐条成功比例。空目标误报率的分母仅为参考关键词为空的样本。\n\n"
report += "## 与已有基线比较\n\n" + md_table(["数据集", "模型", "WER", "BWER", "UWER / NWER"], baseline_rows)
report += "\nA0/A1 均为同一匹配基座，分别使用原始转写提示与联合关键词提示；两组已有官方测试输出均为完整5559条，本次未重跑。A1 关键词列表 recall/F1/exact/format 均为0，precision未定义（未输出关键词块）。internal-test 没有已有的匹配A0/A1结果，因此不报告该集相对改善。\n\n"
for s in ["ls_clean", "ls_other"]:
    c = comparison[s]["A0_original_prompt"]
    report += f"- {names[s]} 相对 A0：WER 增加 {c['WER']['delta_percentage_points']:.3f} 个百分点（相对错误率 +{c['WER']['relative_error_change_percent']:.2f}%），BWER 增加 {c['BWER']['delta_percentage_points']:.3f} 个百分点（+{c['BWER']['relative_error_change_percent']:.2f}%）。\n"
report += "\n## 解读与限制\n\n"
report += "训练 reward 的上升主要来自关键词列表和格式学习；不能把它等同于转写质量提升。固定训练dev是按未训练模型最低BWER筛选的1000条容易样本，筛选时WER/BWER为0，因此dev上选出的最好checkpoint仍须由独立测试验证。本次测试显示关键词词错误增加，非关键词错误略降，整体WER变差。模型在参考为空的样本上仍频繁生成关键词；后续可优先分析词表标注标准与空目标/误报惩罚，再决定新实验。本报告不据此自动更改训练设置或启动额外训练。\n\n"
report += "## 产物与复现\n\n"
report += "- [完整指标](metrics.json)、[输出核验](verification.json)、[基线差异](baseline_comparison.json)、[CSV](all_metrics.csv)。\n"
report += "- [基线比较图](baseline_comparison.png)、[比较图PDF](baseline_comparison.pdf)。\n"
report += "- [训练/validation交互图](../training_charts_20261024/index.html)、[3条原始validation样本](../validation100_first3_raw.md)。\n"
report += "- [导出来源与权重SHA256](../export100_provenance.json)、[checkpoint选择依据](../checkpoint_selection.json)。\n"
report += "- [评估W&B](https://msaip.wandb.io/genai/2609-Keyword/runs/d96bbcbf363a1d7cab0cff8c5d03160a)。\n"
report += "- 训练模型：`az://orngwus2cresco/data/boren/outputs/2609-Keyword/remax_ls_joint_pilot_dev1k/global_step_100/qwen_hf`。\n"
for s in sources:
    report += f"- {names[s]} 原始输出：`{verification[s]['cloud_jsonl']}`，SHA256 `{verification[s]['file_sha256']}`。\n"
(OUT / "final_report.md").write_text(report)


def table(headers, rows):
    return "<table><thead><tr>" + "".join("<th>" + html.escape(h) + "</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LibriSpeech 联合关键词实验 · 最终报告</title><style>body{font:16px system-ui,sans-serif;line-height:1.75;background:#f5f7fc;color:#172033;margin:0}main{max-width:1300px;padding:36px;margin:auto}h1{font-size:30px}.summary{background:#fff1e7;border-left:5px solid #e37b32;padding:18px 24px}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{padding:12px;text-align:right;border-bottom:1px solid #e2e8f0}th{background:#eaf0fb}.note{color:#475569}a{color:#2563eb}img{max-width:100%;background:white;border-radius:12px}</style><main>"""
page += "<h1>LibriSpeech 转写 + &lt;KEYWORDS&gt;：最终结果</h1><p>训练100/100成功 · checkpoint100导出成功 · 评估11,690/11,690成功并核验 · GPU已释放</p>"
page += "<div class='summary'>" + summary + "</div>"
page += "<h2>训练模型的转写指标</h2>" + table(["数据集", "条数", "WER", "BWER", "UWER / NWER"], transcription)
page += "<h2>关键词列表指标</h2>" + table(["数据集", "micro P", "micro R", "micro F1", "exact", "format", "空目标误报率"], keywords)
page += "<p class='note'>列表P/R/F1为micro；exact/format为逐条成功比例。空目标误报率仅以参考关键词为空的样本为分母。</p>"
page += "<h2>与同一基座的官方测试比较</h2><img src='baseline_comparison.png' alt='Official LibriSpeech baseline comparison'>" + table(["数据集", "模型", "WER", "BWER", "UWER / NWER"], baseline_rows)
page += "<p class='note'>A0：原始提示；A1：联合提示；A2：训练checkpoint100。A1没有输出关键词块，列表recall/F1/exact/format为0，precision未定义。internal-test没有已有匹配基线，因此不声称相对改善。</p>"
page += "<h2>选择与核验</h2><p>仅按固定dev1000的最小corpus WER选择checkpoint100，测试未参与选择。全部11,690条输出ID全局唯一，与manifest的ID/参考/关键词/来源哈希匹配；每条错误统计分区成立。独立评估JSONL的step=0不代表未训练模型。</p>"
page += "<p class='note'>训练dev来自未训练模型的最容易1000条，筛选时WER/BWER为0；不能用该dev结果声称超过基座。训练reward包含每组1条GT，验证与测试不注入GT。关键词列表格式学习成功，转写WER和BWER退化；应优先检查误报与空目标处理后再设计下一轮实验。</p>"
page += "<p><a href='final_report.md'>完整报告与来源</a> · <a href='all_metrics.csv'>全部指标CSV</a> · <a href='metrics.json'>指标JSON</a> · <a href='verification.json'>输出核验</a> · <a href='baseline_comparison.pdf'>比较图PDF</a> · <a href='../training_charts_20261024/index.html'>训练和validation图表</a></p></main></html>"
(OUT / "index.html").write_text(page)
print(json.dumps(comparison, indent=2))

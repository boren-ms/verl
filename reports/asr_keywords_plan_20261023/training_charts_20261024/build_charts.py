"""Build meeting charts from saved W&B history and verified training logs."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TMP = ROOT / "tmp/verl_asr_run/keyword_joint_pipeline"
REPORT = OUT.parent
KEYS = [
    "critic/score/mean",
    "critic/baseline_score/mean",
    "reward-core/score/mean",
    "reward-core/asr_score/mean",
    "reward-core/kw_list_f1/mean",
    "reward-core/kw_block_fmt/mean",
    "actor/grad_norm",
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/lr",
    "nonfinite_prob/invalid",
]
history = json.loads((TMP / "meeting_wandb_history.json").read_text())
train = {
    int(row["step"]): {k: row.get(k) for k in KEYS}
    for row in history
    if row.get("step") is not None and row.get("critic/score/mean") is not None
}
provenance = {step: "W&B full scan_history" for step in train}
# Prefer actual successful recovery logs for steps 41-100; this also recovers
# the final step that was not flushed to the W&B history service.
remote = json.loads((TMP / "training41_100_metrics.json").read_text())
for line in remote["rows"]:
    pairs = re.findall(r"(?:^| - )([A-Za-z0-9_/@.:-]+):([-+0-9.eE]+)(?= - |$)", line)
    values = {k: float(v) for k, v in pairs}
    step = int(values["step"])
    assert "critic/score/mean" in values
    train[step] = {k: values.get(k) for k in KEYS}
    provenance[step] = "successful Ray training log: raysubmit_6PPiWDZg5QJ3zagk"
assert len(train) == 99 and set(range(1, 101)) - train.keys() == {40}
for step, row in train.items():
    assert all(np.isfinite(v) for v in row.values() if v is not None), step
    expected = row["reward-core/asr_score/mean"] + 0.5 * row["reward-core/kw_list_f1/mean"] + 0.1 * row["reward-core/kw_block_fmt/mean"]
    assert abs(expected - row["reward-core/score/mean"]) < 1e-6, step

with (OUT / "training_metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=["step", *KEYS, "source"])
    writer.writeheader()
    for step in sorted(train):
        writer.writerow(dict(step=step, **train[step], source=provenance[step]))

validation = json.loads((REPORT / "pilot_dev_validation_metrics.json").read_text())
steps = sorted(int(k) for k in validation)
val = {s: validation[str(s)]["metrics"]["ls_dev_lowbwer1000"] for s in steps}
assert steps == [20, 40, 60, 80, 100]
assert all(x["utterances"] == 1000 for x in val.values())
with (OUT / "validation_metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=["step", *val[20]])
    writer.writeheader()
    for step in steps:
        writer.writerow(dict(step=step, **val[step]))

x = np.arange(1, 101)


def series(key):
    return np.array([train.get(int(s), {}).get(key, np.nan) for s in x], dtype=float)


def smooth(y):
    # Centered five-step window; preserve the missing point at step 40.
    return np.array([
        np.nanmean(y[max(0, i - 2): min(len(y), i + 3)]) if np.isfinite(y[i]) else np.nan
        for i in range(len(y))
    ])


def val_series(key):
    return [100 * val[s][key] for s in steps]


first = {k: float(np.mean([train[s][k] for s in range(1, 11)])) for k in KEYS}
last = {k: float(np.mean([train[s][k] for s in range(91, 101)])) for k in KEYS}
summary = dict(
    training_status="SUCCEEDED 100/100", selected_checkpoint=100,
    available_training_points=99, missing_training_steps=[40],
    first_10_mean=first, last_10_mean=last,
    validation_100=val[100],
    reward_definition="ASR score + 0.5 * per-row keyword-list F1 + 0.1 * keyword-block format",
    training_population="512 rows/batch = 64 groups * (1 GT + 7 sampled); GT included in logged reward",
    validation_population="1000 frozen easy-dev utterances, greedy model generation, no GT injection",
    validation_selection="minimum corpus WER (p_err), tie later; test sets not used",
    baseline_caveat="easy dev selected using untrained BWER/WER; selected baseline WER/BWER=0; final test generalization pending",
)
(OUT / "chart_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

colors = ["#2563eb", "#d97706", "#059669", "#9333ea"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), layout="constrained")
ax = axes[0, 0]
for key, label, color in [("critic/score/mean", "Training reward (includes GT)", colors[0]), ("critic/baseline_score/mean", "Greedy ReMax baseline (model only)", colors[1])]:
    y = series(key)
    ax.plot(x, y, color=color, alpha=.2, linewidth=1)
    ax.plot(x, smooth(y), color=color, linewidth=2.3, label=label)
ax.set(title="Training reward | raw + 5-step mean", ylabel="Reward", xlabel="Optimizer step")
ax.legend(fontsize=9, loc="lower right")

ax = axes[0, 1]
for key, label, color in [("reward-core/asr_score/mean", "ASR score", colors[0]), ("reward-core/kw_list_f1/mean", "Keyword-list F1 (per-row mean)", colors[1]), ("reward-core/kw_block_fmt/mean", "Keyword-block format", colors[2])]:
    y = series(key)
    ax.plot(x, y, color=color, alpha=.15, linewidth=1)
    ax.plot(x, smooth(y), color=color, linewidth=2.3, label=label)
ax.set(title="Training reward components | includes 1/8 GT", ylabel="Score / fraction", xlabel="Optimizer step", ylim=(-.03, 1.06))
ax.legend(fontsize=9, loc="lower right")

ax = axes[1, 0]
for key, label, color in [("WER", "WER (checkpoint selection)", colors[0]), ("BWER", "BWER (keyword words)", colors[1]), ("UWER", "UWER / NWER (other words)", colors[2])]:
    ax.plot(steps, val_series(key), "o-", color=color, linewidth=2, label=label)
    ax.annotate(f"{100 * val[100][key]:.3f}%", (100, 100 * val[100][key]), xytext=(7, 0), textcoords="offset points", color=color, va="center", fontsize=10)
ax.scatter([100], [val[100]["WER"] * 100], s=160, facecolors="none", edgecolors=colors[0], linewidths=2)
ax.set(title="Validation | frozen easy dev, n=1000 | lower is better", ylabel="Corpus error rate (%)", xlabel="Checkpoint step", xlim=(15, 115), ylim=(0, 4.9))
ax.legend(fontsize=9, loc="upper right")

ax = axes[1, 1]
for key, label, color in [("kw_micro_f1", "List F1 (micro)", colors[0]), ("kw_micro_recall", "List recall (micro)", colors[1]), ("kw_micro_precision", "List precision (micro)", colors[3]), ("kw_format_success", "Format success", colors[2])]:
    ax.plot(steps, val_series(key), "o-", color=color, linewidth=2, label=label)
ax.set(title="Validation keyword output | greedy, no GT injection", ylabel="Percent (%)", xlabel="Checkpoint step", ylim=(0, 105))
ax.legend(fontsize=9, loc="lower right")
for i, ax in enumerate(axes.flat):
    ax.grid(alpha=.17)
    if i < 2:
        ax.axvline(40.5, color="#64748b", linestyle="--", alpha=.65)
        ax.text(.02, .98, "Resume from checkpoint 40", transform=ax.transAxes, va="top", fontsize=9, color="#64748b")
        ax.set_xlim(0, 102)
    else:
        ax.axvline(100, color="#64748b", linestyle=":", alpha=.4)
        ax.set_xticks(steps)
fig.suptitle("LibriSpeech joint transcription + <KEYWORDS> | training complete: 100/100", fontsize=19, fontweight="bold")
fig.supxlabel("Training: 64 groups x (1 GT + 7 samples); reward includes GT. Validation: 1000 model-only greedy outputs.\nStep 40 training reward unavailable (gap shown); all 5 validation points verified. Easy-dev baseline floor = 0; test generalization pending.", fontsize=10)
fig.savefig(OUT / "training_validation_overview.png", dpi=180)
fig.savefig(OUT / "training_validation_overview.pdf")
plt.close(fig)

interactive = make_subplots(rows=2, cols=2, subplot_titles=("Training reward (includes GT)", "Training reward components (includes GT)", "Validation corpus error rates — lower is better", "Validation keyword lists / format — higher is better"), vertical_spacing=.14)
for key, label, color in [("critic/score/mean", "Train reward", colors[0]), ("critic/baseline_score/mean", "Greedy ReMax baseline", colors[1])]:
    interactive.add_trace(go.Scatter(x=x, y=series(key), name=label, line=dict(color=color), connectgaps=False), row=1, col=1)
for key, label, color in [("reward-core/asr_score/mean", "Train ASR score", colors[0]), ("reward-core/kw_list_f1/mean", "Train list F1 (per-row)", colors[1]), ("reward-core/kw_block_fmt/mean", "Train format", colors[2])]:
    interactive.add_trace(go.Scatter(x=x, y=series(key), name=label, line=dict(color=color), connectgaps=False), row=1, col=2)
for key, label, color in [("WER", "Dev WER", colors[0]), ("BWER", "Dev BWER", colors[1]), ("UWER", "Dev UWER / NWER", colors[2])]:
    interactive.add_trace(go.Scatter(x=steps, y=val_series(key), name=label, mode="lines+markers", line=dict(color=color)), row=2, col=1)
for key, label, color in [("kw_micro_f1", "Dev list F1", colors[0]), ("kw_micro_recall", "Dev recall", colors[1]), ("kw_micro_precision", "Dev precision", colors[3]), ("kw_format_success", "Dev format", colors[2])]:
    interactive.add_trace(go.Scatter(x=steps, y=val_series(key), name=label, mode="lines+markers", line=dict(color=color)), row=2, col=2)
for col in [1, 2]:
    interactive.add_vline(x=40.5, line_dash="dash", line_color="#94a3b8", row=1, col=col)
    interactive.update_xaxes(title_text="Optimizer step", row=1, col=col)
    interactive.update_xaxes(title_text="Checkpoint step", tickvals=steps, row=2, col=col)
interactive.update_layout(height=850, template="plotly_white", hovermode="x unified", margin=dict(t=60, b=50), legend=dict(orientation="h", y=-.13), font=dict(family="Arial, sans-serif"))
interactive.update_yaxes(title_text="%", row=2, col=1)
interactive.update_yaxes(title_text="%", row=2, col=2)
plot = interactive.to_html(full_html=False, include_plotlyjs=True)
table_rows = "".join("<tr><td>" + "</td><td>".join([str(s), *[f"{100*val[s][k]:.3f}%" for k in ["WER", "BWER", "UWER", "kw_micro_precision", "kw_micro_recall", "kw_micro_f1", "kw_exact", "kw_format_success"]]]) + "</td></tr>" for s in steps)
text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LibriSpeech 训练与验证曲线</title>
<style>body{{font:16px system-ui,sans-serif;color:#172033;background:#f5f7fc;margin:0}}main{{max-width:1450px;margin:auto;padding:32px}}h1{{font-size:30px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;border-radius:12px;padding:20px;flex:1;min-width:180px}}.big{{font-size:30px;font-weight:700;color:#2563eb}}.note{{line-height:1.8;color:#475569}}.plot{{background:white;border-radius:16px;padding:16px;margin:24px 0}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:right;border-bottom:1px solid #e2e8f0}}th{{background:#eaf0fb}}a{{color:#2563eb}}</style>
<main><h1>LibriSpeech 转写 + &lt;KEYWORDS&gt;：训练与验证</h1><p class="note">100/100 步训练成功 · 仅按固定 dev1000 的 corpus WER 选择 checkpoint100 · 测试集未参与选择</p>
<div class="cards"><div class="card">训练 reward：前10步 → 后10步均值<div class="big">{first['critic/score/mean']:.3f} → {last['critic/score/mean']:.3f}</div></div><div class="card">Checkpoint100 · Dev WER<div class="big">{val[100]['WER']*100:.3f}%</div></div><div class="card">Checkpoint100 · 关键词 micro F1<div class="big">{val[100]['kw_micro_f1']*100:.2f}%</div></div><div class="card">Checkpoint100 · 格式成功率<div class="big">{val[100]['kw_format_success']*100:.1f}%</div></div></div>
<p class="note">Reward 上升主要来自关键词列表和格式的学习；ASR 分数保持较高。开发集 WER 在 20–100 步之间波动，100 步最低。关键词 micro F1 从 {val[20]['kw_micro_f1']*100:.2f}% 提升到 {val[100]['kw_micro_f1']*100:.2f}%。</p>
<div class="plot">{plot}</div>
<h2>已核验的 validation 指标</h2><table><thead><tr><th>Step</th><th>WER</th><th>BWER</th><th>UWER/NWER</th><th>List P</th><th>List R</th><th>List F1</th><th>List exact</th><th>Format</th></tr></thead><tbody>{table_rows}</tbody></table>
<p class="note">训练统计：64组 ×（1条完整 GT + 7条采样），共512条，图中 reward 和分项均包含 GT。开头 F1/format 的 12.5% 包含 GT 贡献。Greedy ReMax baseline 由模型生成，不注入 GT。训练 F1 是逐条平均，验证 F1 是 corpus micro，不能直接比较绝对值。总 reward = ASR score + 0.5 × list F1 + 0.1 × block format。</p>
<p class="note">固定开发集是按未训练模型的低 BWER 挑选的最容易1000条；筛选时基线 WER/BWER 均为0。因此这里只能说明本次训练过程和关键词输出学习情况，不能据此声称转写超过未训练基线。泛化结论需等待独立测试集结果。图中未连入不同范围的全量 dev6096。</p>
<p class="note">数据覆盖：99个可恢复的训练点；step40 的训练 reward 未留存，保留缺口，不插值。step41–100 来自成功训练任务原始日志，step1–39 来自 W&B 完整 history；validation20/40/60/80/100 均逐条核验1000个唯一ID。静态图使用居中5步均值；交互图显示原始逐步值。竖虚线表示从 checkpoint40 恢复。</p>
<p><a href="training_validation_overview.png">PNG</a> · <a href="training_validation_overview.pdf">PDF</a> · <a href="training_metrics.csv">Training CSV</a> · <a href="validation_metrics.csv">Validation CSV</a> · <a href="https://msaip.wandb.io/genai/2609-Keyword/runs/24964a75bb9508d952a03ab5e6720e5b">W&B</a></p></main></html>"""
(OUT / "index.html").write_text(text)
print(json.dumps({"reward_first10": first["critic/score/mean"], "reward_last10": last["critic/score/mean"], "files": [p.name for p in OUT.iterdir()]}, indent=2))

"""Render a pinned language-error analysis as charts, Excel and an HTML report."""
import argparse
import csv
import html
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histogram(values, counts, edges):
    result, _ = np.histogram(np.asarray(values, dtype=float), bins=edges, weights=np.asarray(counts, dtype=float))
    return result


def run(folder):
    s = json.loads((folder / 'summary.json').read_text())
    df = pd.read_csv(folder / 'language_summary.csv')
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    observed = [v for v in s['languages'].values() if v['decoded_samples']]
    cards, plot_data = [], []
    for info in observed:
        lang = info['language']
        if not info['error_samples_observed']:
            continue
        top = [json.loads(line) for line in (folder / info['top_path']).open()]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        specifications = [
            ('duration', 'audio_duration_seconds', 'Audio duration (seconds)', [0, 5, 10, 15, 20, 30, 40, 60, 120, 300, float('inf')]),
            ('word_count', 'analysis_word_count', 'Original-reference whitespace word count', [0, 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500, float('inf')]),
            ('error_count', 'n_err', 'Absolute error count (n_err)', [1, 2, 3, 4, 6, 11, 21, 51, 101, 201, 501, float('inf')]),
        ]
        for ax, (field, top_field, title, edges) in zip(axes, specifications):
            freq = pd.read_csv(folder / 'distributions' / f'{lang}_{field}_frequency.csv')
            all_counts = histogram(freq['value'], freq['samples'], edges)
            top_counts = histogram([r[top_field] for r in top], np.ones(len(top)), edges)
            assert int(sum(all_counts)) == info['error_samples_observed']
            assert int(sum(top_counts)) == len(top)
            labels = []
            for lo, hi in zip(edges, edges[1:]):
                labels.append(f'{lo:g}+' if np.isinf(hi) else f'[{lo:g},{hi:g})')
            x = np.arange(len(labels))
            ax.bar(x - .2, all_counts / sum(all_counts) * 100, .4, label='All observed errors', color='#377eb8')
            ax.bar(x + .2, top_counts / sum(top_counts) * 100, .4, label=f'Top {len(top):,}', color='#ff8c42')
            ax.set(xticks=x, xticklabels=labels, ylabel='Samples (%)', title=title)
            ax.tick_params(axis='x', labelrotation=55)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            for label, a, b in zip(labels, all_counts, top_counts):
                plot_data.append({'language': lang, 'metric': field, 'bin': label,
                                  'all_error_samples': int(a), 'top_samples': int(b),
                                  'all_error_pct': a / sum(all_counts) * 100,
                                  'top_pct': b / sum(top_counts) * 100})
        axes[0].legend(fontsize=9)
        fig.suptitle(f'{info["language_name"]} ({lang}) | observed errors: {info["error_samples_observed"]:,} | {info["status"]}', fontsize=15)
        image = f'figures/{lang}_distributions.png'
        fig.savefig(folder / image, dpi=160)
        plt.close(fig)
        table = []
        for metric, label in [('duration', '时长（秒）'), ('word_count', '原始 reference 空白词数'), ('character_count', '非空白字符数'), ('error_count', '绝对错误数 n_err'), ('scored_reference_units', '评分器 reference 单位数 n_ref')]:
            v = info['statistics'][metric]
            table.append({'指标': label, **{k: v[k] for k in ['mean', 'min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']}})
        metrics = pd.DataFrame(table)
        metrics.to_csv(folder / f'{lang}_statistics.csv', index=False, encoding='utf-8-sig')
        cards.append(f'<section><h2>{html.escape(info["language_name"])} ({lang})</h2>'
                     f'<p>已解码 {info["decoded_samples"]:,} / {info["expected_samples"]:,} 条；'
                     f'错误样本 {info["error_samples_observed"]:,} 条，合计 {info["error_duration_hours"]:,.2f} 小时。'
                     f'Top {len(top):,} 的 n_err 范围 {info["top_min_n_err"]}–{info["top_max_n_err"]}，'
                     f'合计 {info["top_duration_hours"]:,.2f} 小时。</p>'
                     f'<img src="{image}" alt="Duration, word count and error count distributions">'
                     + metrics.to_html(index=False, float_format=lambda x: f'{x:,.3f}')
                     + f'<p><a href="{info["top_path"]}">下载该语言排序 JSONL</a> · '
                     f'<a href="{lang}_statistics.csv">分位数 CSV</a></p></section>')
    pd.DataFrame(plot_data).to_csv(folder / 'histogram_bins.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(folder / 'language_error_statistics.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Languages', index=False, freeze_panes=(1, 3))
        pd.DataFrame(plot_data).to_excel(writer, sheet_name='Histogram_bins', index=False)
        pd.DataFrame([{'field': k, 'value': str(v)} for k, v in s.items() if k != 'languages']).to_excel(writer, sheet_name='Method_and_provenance', index=False)
        for info in observed:
            lang = info['language']
            if not info['top_path']:
                continue
            data = [json.loads(line) for line in (folder / info['top_path']).open()]
            columns = ['analysis_language_rank', 'audio_chunk', 'source_family', 'source_language', 'n_err', 'n_ins', 'n_del', 'n_sub', 'n_ref', 'audio_duration_seconds', 'analysis_word_count', 'analysis_character_count', 'text', 'response']
            pd.DataFrame([{k: r.get(k) for k in columns} for r in data]).to_excel(writer, sheet_name=f'{lang}_Top10000'[:31], index=False, freeze_panes=(1, 2))
        for sheet in writer.book:
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                letter = column[0].column_letter
                sheet.column_dimensions[letter].width = min(55, max(14, len(str(column[0].value)) + 2))
    with zipfile.ZipFile(folder / 'top10000_by_language.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted((folder / 'top10000').glob('*.jsonl')):
            z.write(f, f'top10000/{f.name}')
        z.write(folder / 'summary.json', 'summary.json')
    coverage = s['decoded_unique'] / s['expected_rows'] * 100
    lead = f'固定快照 {s["observed_utc"]}：已解码 {s["decoded_unique"]:,} / {s["expected_rows"]:,} 条（{coverage:.2f}%），发现 {s["error_rows_partial"]:,} 条 n_err&gt;0 样本。'
    fields = ['language', 'language_name', 'status', 'expected_samples', 'decoded_samples', 'coverage_pct', 'error_samples_observed', 'error_duration_hours', 'duration_mean', 'word_count_mean', 'error_count_mean', 'top_count']
    doc = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OSS 按语言错误样本统计</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;color:#17233a;background:#f7f9fc}h1,h2{color:#123d70}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow:auto}td,th{padding:7px 10px;border:1px solid #dce3ed;white-space:nowrap}th{background:#e8eff8}a{color:#145ac4}.notice{padding:18px;background:#fff2ce;border-left:5px solid #d79700}</style>'
    doc += '<h1>OSS 按语言错误样本统计与 Top 10,000</h1><p>' + lead + '</p>'
    status_text = ('解码覆盖已完整；Top-K 由各语言全部已核验 generation 结果计算。最终训练发布仍须完成 pipeline 的 reference/split/heldout 审计。'
                   if s['complete_generation'] else
                   '解码尚未完成。这是已解码前缀的精确统计与临时 Top-K，不能称为全量语言统计或最终全局 Top-K。NOT_DECODED 的错误数量、时长和均值为 N/A，不代表没有错误。')
    doc += '<div class="notice">' + status_text + '分析子集未用于替换训练数据。</div>'
    doc += '<p><a href="language_error_statistics.xlsx">Excel 工作簿</a> · <a href="language_summary.csv">全部语言汇总 CSV</a> · <a href="top10000_by_language.zip">各语言排序 JSONL 压缩包</a> · <a href="summary.json">完整统计与来源证据</a></p>'
    doc += ''.join(cards)
    doc += '<section><h2>全部语言覆盖情况</h2>' + df[fields].to_html(index=False, na_rep='N/A', float_format=lambda x: f'{x:,.3f}') + '</section>'
    doc += '<section><h2>统计口径</h2><ul><li>仅保留 n_err&gt;0；按绝对 n_err 降序、canonical audio_chunk 升序排序。不按错误率排序，不补足不足 10,000 条的语言。</li><li>Word count = len(原始 reference.split())；中文等无空格语言不应解释为语言学词数，另列非空白 Unicode 字符数。n_ref 是评分器规范化后的单位数，与原始 word count 分开。</li><li>时长来自实际已解码波形的 audio_duration_seconds；总时长和平均时长按错误样本逐条累加，未用源平均值估算。</li><li>每种分布均提供完整精确频数 CSV；图表采用固定区间，包含长尾；分位数采用 nearest-rank，均值使用完整精确频数。</li><li>zh/yue 的 n_err 是字符编辑数，其他语言是现有 whitespace WER 的词编辑数，不跨语言合成一个 WER。原始人工 reference 保留，hypothesis 仅供审计。</li><li>完整 reference 相等性与 heldout/parent/waveform 审计仍待原 pipeline 完成；本分析不替代最终训练发布门槛。</li></ul></section></html>'
    (folder / 'report.html').write_text(doc)
    (folder / 'README.md').write_text('# OSS 按语言错误样本统计\n\n' + lead.replace('&gt;', '>') + '\n\n请打开 [HTML 报告](report.html)。Excel 含所有语言汇总、分布区间和已观测语言 Top 10,000；JSONL 保留完整原始 reference 和解码字段。未完成语言的 Top-K 为临时结果，尚未解码的语言为 N/A。精确频数见 distributions/，来源及排序口径见 summary.json。\n')
    print(json.dumps({'report': str(folder / 'report.html'), 'workbook': str(folder / 'language_error_statistics.xlsx'), 'zip': str(folder / 'top10000_by_language.zip'), 'observed_languages': [v['language'] for v in observed]}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    run(p.parse_args().snapshot.resolve())

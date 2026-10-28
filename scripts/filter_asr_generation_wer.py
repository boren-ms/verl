#!/usr/bin/env python3
"""Validate complete generation, retain strict WER thresholds, and compare distributions.

Example: python scripts/filter_asr_generation_wer.py --input-dir az://... \
    --output-dir az://... --report-dir outputs/opensource_wer --expected-rows 1920120
Original ground-truth text and audio pointers are preserved. No upper WER cap.
"""
import argparse
from array import array
from collections import defaultdict
import json
import math
from pathlib import Path
import re

METRICS = ('wer_percent', 'audio_seconds', 'reference_word_count',
           'hypothesis_word_count', 'scoring_reference_words')
BINS = {
    'wer_percent': [0, 1, 5, 10, 20, 50, 100, 200, float('inf')],
    'audio_seconds': [0, 2, 5, 10, 20, 40, 60, 120, float('inf')],
    'reference_word_count': [0, 5, 10, 20, 50, 100, 200, 500, float('inf')],
    'hypothesis_word_count': [0, 5, 10, 20, 50, 100, 200, 500, float('inf')],
    'scoring_reference_words': [0, 5, 10, 20, 50, 100, 200, 500, float('inf')],
}


def _source_groups(row, pointer):
    """Return stable report groups without constraining valid audio layouts."""
    match = re.search(r'/Chunk/(VOiCES|VCTK|LibriTTS)/(.*?)/[^/]+\.audio:', pointer)
    if match:
        source, branch = match.groups()
        return source, f'{source}/{branch}'

    pointer_lower = pointer.lower()
    if '/speech/earningscall_2026_conversion_svad/' in pointer_lower:
        return 'earnings_fy27', 'earnings_fy27/batch1'
    if '/speech/earningscall-2026/batch2-raw/' in pointer_lower:
        return 'earnings_fy27', 'earnings_fy27/batch2'

    data_source = row.get('data_source')
    if isinstance(data_source, str) and data_source:
        return data_source, data_source
    return 'unknown', 'unknown'


def measurements(row):
    for key in ('n_err', 'n_ref'):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise ValueError(f'Invalid integer {key}: {value!r}')
    errors, refs = int(row['n_err']), int(row['n_ref'])
    if errors < 0 or refs <= 0:
        raise ValueError('Expected n_err >= 0 and n_ref > 0')
    wer = row.get('wer')
    if isinstance(wer, bool) or not isinstance(wer, (int, float)) or not math.isfinite(wer) or not math.isclose(wer, errors / refs, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError('WER disagrees with error/reference counts')
    duration = row.get('audio_duration_seconds')
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Missing/invalid decoded audio duration')
    if not isinstance(row.get('text'), str) or not isinstance(row.get('response'), str):
        raise ValueError('Missing reference or response text')
    pointer = row.get('audio_chunk') or row.get('audio_path')
    if not isinstance(pointer, str) or not pointer:
        raise ValueError('Missing unique audio pointer')
    source, branch = _source_groups(row, pointer)
    return source, branch, pointer, (
        100 * errors / refs, duration, len(row['text'].split()), len(row['response'].split()), refs)


def retained(row):
    return {'wer_gt0': row['n_err'] > 0, 'wer_gt1pct': 100 * row['n_err'] > row['n_ref']}


class Stats:
    def __init__(self):
        self.values = [array('d') for _ in METRICS]
        self.errors = self.refs = self.at_limit = 0

    def add(self, row, values):
        for column, value in zip(self.values, values):
            column.append(value)
        self.errors += row['n_err']
        self.refs += row['n_ref']
        self.at_limit += bool(row.get('response_at_limit', False))

    def summary(self, denominator):
        import numpy as np
        out = {'count': len(self.values[0]), 'retention_rate': len(self.values[0]) / denominator if denominator else None,
               'corpus_wer_percent': 100 * self.errors / self.refs if self.refs else None,
               'n_err': self.errors, 'n_ref': self.refs, 'response_at_limit_count': self.at_limit}
        for key, column in zip(METRICS, self.values):
            values = np.asarray(column)
            if not len(values):
                out[key] = None
                continue
            bins = BINS[key]
            counts, _ = np.histogram(values, bins=bins)
            out[key] = {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
                        'quantiles': dict(zip(['p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'],
                                              map(float, np.quantile(values, [.1, .25, .5, .75, .9, .95, .99])))),
                        'zero_count': int((values == 0).sum()),
                        'histogram': [{'lower_inclusive': lo, 'upper_exclusive': hi if math.isfinite(hi) else None, 'count': int(n)}
                                      for lo, hi, n in zip(bins, bins[1:], counts)]}
        out['audio_hours'] = sum(self.values[1]) / 3600
        return out


def iter_rows(paths):
    import blobfile as bf
    for path in paths:
        with bf.BlobFile(path, 'r') as stream:
            for i, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    measurements(row)
                    yield row
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f'{path}:{i}: {exc}') from exc


def run(args):
    import blobfile as bf
    import numpy as np
    paths = sorted(bf.glob(args.input_dir.rstrip('/') + '/part-*.jsonl'),
                   key=lambda p: int(re.search(r'part-(\d+)\.jsonl$', p).group(1)))
    indices = [int(re.search(r'part-(\d+)\.jsonl$', p).group(1)) for p in paths]
    if not paths or indices != list(range(len(paths))):
        raise ValueError('Expected contiguous generation part files from zero')
    if bf.exists(args.output_dir.rstrip('/') + '/COMPLETE.json'):
        raise ValueError('Output already complete; use a fresh output directory')
    stats = defaultdict(lambda: defaultdict(Stats))
    seen = set()
    for row in iter_rows(paths):
        source, branch, pointer, values = measurements(row)
        if pointer in seen:
            raise ValueError(f'Duplicate audio pointer: {pointer}')
        seen.add(pointer)
        subsets = ['original'] + [name for name, keep in retained(row).items() if keep]
        for group in dict.fromkeys(('all', source, branch)):
            for subset in subsets:
                stats[group][subset].add(row, values)
    if len(seen) != args.expected_rows:
        raise ValueError(f'Incomplete generation: {len(seen)} rows, expected {args.expected_rows}; no subsets published')
    del seen
    summary = {'expected_rows': args.expected_rows, 'input_parts': paths,
               'definitions': {'wer': 'compound-aware HF English normalized WER; percent, uncapped',
                               'reference_word_count': 'original reference whitespace words',
                               'hypothesis_word_count': 'parsed response whitespace words',
                               'scoring_reference_words': 'normalized WER denominator (floor one for empty reference)',
                               'audio_seconds': 'waveform delivered to model; max_audio_dur=null',
                               'original': 'all generated samples before filtering',
                               'wer_gt0': 'n_err > 0', 'wer_gt1pct': '100*n_err > n_ref'}, 'groups': {}}
    for group, subsets in stats.items():
        denominator = len(subsets['original'].values[0])
        summary['groups'][group] = {name: subsets[name].summary(denominator)
                                     for name in ('original', 'wer_gt0', 'wer_gt1pct')}
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    # Each source part maps to one filtered part. Original text, response and audio pointers survive.
    for index, path in enumerate(paths):
        from contextlib import ExitStack
        with ExitStack() as stack:
            outputs = {}
            for row in iter_rows([path]):
                for name, keep in retained(row).items():
                    if keep:
                        if name not in outputs:
                            directory = args.output_dir.rstrip('/') + '/' + name
                            bf.makedirs(directory)
                            outputs[name] = stack.enter_context(bf.BlobFile(f'{directory}/part-{index:03d}.jsonl', 'w'))
                        outputs[name].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, METRICS[:4]):
        for name in ('original', 'wer_gt0', 'wer_gt1pct'):
            values = np.asarray(stats['all'][name].values[METRICS.index(metric)])
            if len(values):
                # ECDF retains the uncapped long tail and compares unequal populations fairly.
                ordered = np.sort(values)
                picks = np.unique(np.linspace(0, len(ordered) - 1, min(len(ordered), 3000), dtype=int))
                ax.plot(ordered[picks], (picks + 1) / len(ordered), label=name)
        ax.set_title(metric)
        ax.set_ylabel('Cumulative fraction')
        ax.set_xscale('symlog', linthresh=1)
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / 'distributions.png', dpi=160)
    plt.close(fig)
    lines = ['# ASR WER 过滤统计', '', '原始分布是全部样本生成后的分布；过滤后分布来自同一批模型响应。原始转写继续作为训练标签。', '',
             '| 数据 | 子集 | 样本数 | 保留率 | Corpus WER | 平均样本 WER | 音频小时 |',
             '|---|---|---:|---:|---:|---:|---:|']
    report_groups = ['all'] + sorted(
        group for group in summary['groups'] if group != 'all' and '/' not in group
    )
    for group in report_groups:
        if group not in summary['groups']:
            continue
        for name, data in summary['groups'][group].items():
            wer = data['corpus_wer_percent']
            mean = data['wer_percent']['mean'] if data['wer_percent'] else None
            lines.append(f"| {group} | {name} | {data['count']:,} | {data['retention_rate']:.2%} | {wer if wer is not None else 'N/A'} | {mean if mean is not None else 'N/A'} | {data['audio_hours']:.2f} |")
    lines += ['', '![WER、时长与词数累计分布](distributions.png)', '',
              'summary.json 包含总体、数据集和 clean/simu 分支的直方图、零值数、均值及 p10/p25/p50/p75/p90/p95/p99。WER 不设上限；词数分别统计原始转写、生成转写和 WER 归一化参考词数。',
              '', f"主要过滤数据：`{args.output_dir}/wer_gt1pct/part-*.jsonl`", '',
              'response_at_limit_count 标记达到输出 token 上限的样本，需结合报告检查潜在输出截断。']
    (report_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    # JSON is a YAML subset, so no extra parser dependency is needed here.
    task_info = {'task': getattr(args, 'training_task', 'lang_asr')}
    if getattr(args, 'training_language', None):
        task_info.update(language=args.training_language,
                         prefix_prob=0.0,
                         lang_hint=getattr(args, 'training_lang_hint', False))
    post_process = {'verl_format': {'prompt_key': 'prompt',
        'extra_keys': ['wer', 'n_err', 'n_ref', 'audio_duration_seconds']}}
    if getattr(args, 'training_data_source', None):
        post_process['add_field'] = {'fields': {'data_source': args.training_data_source}}
    training_config = {
        'dataset_name': 'jsonl',
        'version': 2609,
        'jsonl_paths': [args.output_dir.rstrip('/') + '/wer_gt1pct/part-*.jsonl'],
        'add_task_info': task_info,
        'post_process': post_process,
    }
    (report_dir / 'training_data.yaml').write_text(json.dumps(training_config, indent=2) + '\n')
    for name in ('summary.json', 'report.md', 'distributions.png', 'training_data.yaml'):
        bf.copy(str(report_dir / name), args.output_dir.rstrip('/') + '/' + name, overwrite=True)
    with bf.BlobFile(args.output_dir.rstrip('/') + '/COMPLETE.json', 'w') as stream:
        json.dump({'rows': args.expected_rows, 'counts': {k: v['count'] for k, v in summary['groups']['all'].items()}}, stream)
    print(json.dumps(summary['groups']['all'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--expected-rows', type=int, required=True)
    parser.add_argument('--training-task', default='lang_asr')
    parser.add_argument('--training-language')
    parser.add_argument('--training-lang-hint', action='store_true')
    parser.add_argument('--training-data-source')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

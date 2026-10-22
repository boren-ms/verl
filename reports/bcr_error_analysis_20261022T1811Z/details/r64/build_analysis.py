#!/usr/bin/env python3
"""Build official R64-vs-Fix step440 summary from frozen logged counts."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
evidence = json.loads((OUT / 'examples_evidence.json').read_text())
diagnostic = json.loads((OUT / 'parentheses_diagnostic.json').read_text())
fix_lineage = json.loads((OUT / 'sources/fix_retained_lineage.json').read_text())['runs']['baseline']
fix440 = next(row for row in fix_lineage['validation_rows'] if row['step'] == 440)
r64_history = json.loads((OUT / 'sources/r64_metrics_history.json').read_text())
r64_segment = 'bcr64-recover390-20261022T1535Z'
r64440 = next(row for row in r64_history['segments'][r64_segment]['validation_rows'].values() if row['step'] == 440)
selected = {
    'openml_bad': ['8031031497862620363', '16940047964722835471', '9071737899953350186', '2114_1656_000013'],
    'ls_clean': ['7176-92135-0037', '8555-284447-0006', '1995-1837-0005'],
    'ls_other': ['1998-29454-0017', '2609-157645-0008', '2609-169640-0000'],
}
datasets = {}
for name, ds in evidence['per_dataset'].items():
    csv_path = next((OUT / 'detail_reports' / name).glob('*.full.csv'))
    full = pd.read_csv(csv_path, dtype={'id': str})
    info = {key: ds[key] for key in [
        'step', 'unit', 'baseline_raw_rows', 'target_raw_rows', 'baseline_units', 'target_units',
        'matched', 'baseline_only', 'target_only', 'reference_conflicts', 'metadata_conflicts',
        'logged_n_ref_differences_despite_equal_gts', 'baseline_invalid_units', 'target_invalid_units',
        'category_counts', 'recovered_subset_of_improved', 'delta_n_err', 'delta_p_err_percentage_points',
        'added_errors_in_newly_failed', 'added_errors_in_worsening_persistent', 'removed_errors_in_improved',
    ]}
    info['metrics'] = {}
    info['full_csv'] = str(csv_path)
    info['normalized_references_equal'] = int(full['ref_matches_baseline'].sum())
    info['original_raw_references_equal'] = len(full)
    info['source_files'] = {}
    for side, scalar in [('baseline', fix440['metrics']), ('target', r64440['p_err'])]:
        counts = ds[f'{side}_counts']
        computed = {label: int(full[f'{side}_{field}'].sum()) for label, field in [
            ('n_sub', 'substitutions'), ('n_del', 'deletions'), ('n_ins', 'insertions'),
            ('n_err', 'errors'), ('n_ref', 'ref_words'),
        ]}
        assert computed['n_err'] == counts['n_err']
        assert computed['n_ref'] == counts['n_ref']
        assert computed['n_sub'] + computed['n_del'] + computed['n_ins'] == computed['n_err']
        computed['p_err'] = computed['n_err'] / computed['n_ref']
        computed['p_err_percent'] = computed['p_err'] * 100
        computed['logged_validation_p_err'] = scalar[f'val-core/{name}/p_err/mean@1']
        assert abs(computed['p_err'] - computed['logged_validation_p_err']) < 1e-14
        computed['recomputed_total_matches_logged_counts_and_validation_scalar'] = True
        info['metrics'][side] = computed
        path = OUT / side / f'{name}_440.jsonl'
        info['source_files'][side] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    info['sdi_delta_target_minus_baseline'] = {
        field: info['metrics']['target'][field] - info['metrics']['baseline'][field]
        for field in ['n_sub', 'n_del', 'n_ins', 'n_err', 'n_ref']
    }
    info['examples'] = []
    for identity in selected.get(name, []):
        row = full.loc[full['id'] == identity].iloc[0]
        info['examples'].append({
            'id': identity,
            'baseline_n_err': int(row['baseline_errors']), 'target_n_err': int(row['target_errors']),
            'reference': row['raw_ref'],
            'baseline_hypothesis': row['clean_output_baseline'], 'target_hypothesis': row['clean_output_target'],
        })
    if name == 'earning_chunk_verb':
        info['parent_error_changes'] = [
            {key: row[key] for key in ['stable_identity', 'baseline_n_err', 'target_n_err', 'delta_n_err', 'segment_count']}
            for row in ds['examples']
        ]
    datasets[name] = info

table_coverage = {}
for side, table in evidence['tables'].items():
    table_coverage[side] = {key: table[key] for key in [
        'raw_flattened_observations', 'unique_observations', 'duplicate_cumulative_copies_removed',
        'step_coverage', 'selected_step', 'selected_matches', 'selected_unmatched',
        'selected_ambiguous', 'selected_dataset_coverage',
    ]}

analysis = {
    'generated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'scope': 'Frozen step440; R64 versus direct Fix; four complete matched validation corpora',
    'selected_common_validation_step': 440,
    'baseline': {
        'name': 'Fix', 'run_id': '0ca34a537b05e74628264cac2ea86a75',
        'retained_segment': fix440['segment_id'], 'source_identity': fix440['source_identity'],
        'scalar_source': fix440['source'], 'scalar_evidence_coverage': fix440['evidence_coverage'],
    },
    'target': {
        'name': 'R64', 'run_id': 'e51cc81ed31e918470dbcc8143a6b731',
        'retained_segment': r64_segment, 'job_id': r64440['job_id'], 'driver_id': r64440['driver_id'],
        'scalar_source': r64440['source_path'],
        'lineage': 'Original raysubmit_a5BkBYyJeRuPPHft retained through390; current recovery retained391+; discarded original391-396 excluded',
    },
    'metrics_policy': 'Official logged n_err/n_ref are primary. Exact scorer S/D/I recomputation reproduces all per-dataset totals and official scalars. Each side retains its own denominator. No averaging or pooling across datasets.',
    'bad_case_policy': evidence['categories'],
    'per_dataset': datasets,
    'table_coverage': table_coverage,
    'parentheses_diagnostic': {
        'path': str(OUT / 'parentheses_diagnostic.json'),
        'reproduction_script': str(OUT / 'parentheses_diagnostic.py'),
        'official_scoring_replaced': False,
        'method': diagnostic['transformation'],
        'all_393_original_row_counts_exactly_reproduced': True,
        'aggregates': diagnostic['aggregates'],
        'original_error_gap': diagnostic['original_target_minus_baseline_errors'],
        'diagnostic_error_gap': diagnostic['diagnostic_target_minus_baseline_errors'],
        'gap_change': diagnostic['gap_change'],
        'rows_with_metric_change_on_either_side': diagnostic['rows_with_metric_change_on_either_side'],
    },
    'limitations': evidence['limitations'] + [
        'Skill-generated OpenML summary.json baseline_wer incorrectly divides Fix476 by target ref5633. The full CSV has correct separate ref counts5630/5633. This analysis uses official logged counts, hence Fix8.454706927%, R648.574471862%. Comparator source is unchanged.',
        'The parentheses sensitivity analysis is a scoring diagnostic, not an alternate official result. It cannot establish an acoustic or causal training mechanism.',
        'These are single fixed-step comparisons without repeated training seeds; small differences are descriptive.',
    ],
}
(OUT / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')

lines = ['# R64 与 Fix：step 440 完整验证对比', '',
         '正式结果来自冻结的完整 dump 和对应 retained lineage；四个数据集逐一核对，没有跨数据集合并平均。所有 S/D/I 重算均准确复现 logged n_err/n_ref 和 val-core p_err。', '',
         '| 验证集 | 单元 | Fix p_err | R64 p_err | Δ pp | Fix S/D/I | R64 S/D/I |',
         '|---|---:|---:|---:|---:|---:|---:|']
for name, ds in datasets.items():
    b, t = ds['metrics']['baseline'], ds['metrics']['target']
    units = '6 parents / 341 chunks' if name == 'earning_chunk_verb' else str(ds['matched'])
    sdi = lambda m: '/'.join(str(m[k]) for k in ['n_sub', 'n_del', 'n_ins'])
    lines.append(f"| {name} | {units} | {b['p_err_percent']:.6f}% | {t['p_err_percent']:.6f}% | {ds['delta_p_err_percentage_points']:+.6f} | {sdi(b)} | {sdi(t)} |")
lines += ['',
    'R64 在 Earnings、OpenML、LS clean 的错误数分别增加 7、7、6，在 LS other 减少 14。Earnings 的两条变差 parent 合计增加 33 个错误，四条改善 parent 合计减少 26；其替换错误增加 21，删除和插入分别减少 8、6。LS clean 主要表现为替换增加 11，LS other 则替换和插入分别减少 8、11，删除增加 5。', '',
    '| 验证集 | 改善 | 新增错误样本 | 原坏样本变差 | 原坏样本等率 | 一直正确 |',
    '|---|---:|---:|---:|---:|---:|']
for name, ds in datasets.items():
    c = ds['category_counts']
    lines.append('| ' + name + ' | ' + ' | '.join(str(c[k]) for k in ['improved', 'newly_failed', 'worsening_persistent', 'persistent', 'unchanged_correct']) + ' |')
lines += ['',
    'OpenML 的括号诊断对两侧的 raw reference 和 clean_output 一致地只删除 ASCII `(`、`)`，保留内部文字，再走同一语言分派及 normalizer。393 条逐条原分数均已复现。Fix 476/5630 → 464/5630；R64 483/5633 → 466/5632，因此错误数差距 +7 → +2，净差距中 5 个错误对格式敏感。三个例子 `8031031497862620363`、`16940047964722835471`、`9071737899953350186` 分别从差距 +4/+2/+2 变为 0，但同时存在 Fix 的括号型收益被抵消，不能只加总目标模型的个例收益。正式结果仍采用原计分。', '',
    '内容变化的反例也同时存在：OpenML `2114_1656_000013` 恢复开头 “qui me menait”，错误 4→0；LS clean `7176-92135-0037` 把 “a meal” 写成 “Amiel”，0→2，而 `1995-1837-0005` 把 “inhuman” 还原为 “and human”，2→0；LS other `2609-169640-0000` 恢复 “the sea is full of them”，7→3，而 `1998-29454-0017` 的 “ain’t / she’ll” 变为 “an’t / you’ll”，0→3。', '',
    '覆盖：所有 5958 个验证单元完整配对，Earnings 的 341 chunks / 6 parents 均完整。两侧 W&B step440 各 20 条低分表样本均唯一回连到完整 dump；表不含 Earnings，不能据表推断各集错误率。OpenML 相同 raw references 中有两条经联合归一化后分母不同，保留实际 5630/5633。', '',
    '注意：技能生成的 OpenML summary.json 的 baseline_wer 使用了 target 分母；本报告和 analysis.json 使用正确的逐侧官方分母。详细 transcript HTML 不含该总指标头。单 step、单 seed 只能描述当前差异，尚不能归因到 BCR 的因果机制。', '',
    '机器可读证据：`analysis.json`、`examples_evidence.json`、`matched_validation_cases.jsonl`；诊断：`parentheses_diagnostic.json` 和可重跑脚本 `parentheses_diagnostic.py`。', '']
(OUT / 'report.md').write_text('\n'.join(lines))
print(json.dumps({'analysis': str(OUT / 'analysis.json'), 'report': str(OUT / 'report.md'), 'all_four_official_scalar_checks_passed': True}))

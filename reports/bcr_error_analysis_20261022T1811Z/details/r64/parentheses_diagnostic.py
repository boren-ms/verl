#!/usr/bin/env python3
"""Diagnostic only: remove ASCII parentheses, preserve their contents, rescore.

The official untouched scores remain the primary reported results. This reuses
the exact scoring functions called by the asr-detail-compare skill with auto
language dispatch, on all 393 matched OpenML rows on both sides.
"""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path('/home/boren/code/verl')
OUT = Path(__file__).resolve().parent
SCORER = ROOT / '.github/skills/asr-detail-compare/scripts/compare_result_details.py'
spec = importlib.util.spec_from_file_location('r64_comparison_scorer', SCORER)
scorer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scorer
spec.loader.exec_module(scorer)
TRANSLATION = str.maketrans('', '', '()')
FIELDS = ['ref_words', 'errors', 'substitutions', 'deletions', 'insertions']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    return scorer.compute_metrics(
        pd.DataFrame(rows), 'gts', 'clean_output', 'score', 'auto', '', 'language'
    ).rename(columns=lambda c: c.removeprefix('score_'))


def total(frame):
    result = {key: int(frame[key].sum()) for key in FIELDS}
    result['p_err'] = result['errors'] / result['ref_words']
    return result


sources = {}
sides = {}
for side, expected in [('baseline', (476, 5630)), ('target', (483, 5633))]:
    path = OUT / side / 'openml_bad_440.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 393
    assert len({row['id'] for row in rows}) == len(rows)
    original = metrics(rows)
    assert (int(original['errors'].sum()), int(original['ref_words'].sum())) == expected
    assert all(int(original.iloc[i]['errors']) == row['n_err'] for i, row in enumerate(rows))
    assert all(int(original.iloc[i]['ref_words']) == row['n_ref'] for i, row in enumerate(rows))
    transformed = [dict(row, gts=str(row['gts']).translate(TRANSLATION),
                        clean_output=str(row['clean_output']).translate(TRANSLATION)) for row in rows]
    diagnostic = metrics(transformed)
    sides[side] = {
        'rows': rows, 'transformed': transformed,
        'original': original, 'diagnostic': diagnostic,
        'index': {row['id']: i for i, row in enumerate(rows)},
    }
    sources[side] = {'path': str(path), 'sha256': sha256(path), 'rows': len(rows)}

assert set(sides['baseline']['index']) == set(sides['target']['index'])
assert all(sides['baseline']['rows'][i]['gts'] ==
           sides['target']['rows'][sides['target']['index'][key]]['gts']
           for key, i in sides['baseline']['index'].items())

cases = []
for key, base_index in sides['baseline']['index'].items():
    case = {'id': key, 'language': sides['baseline']['rows'][base_index]['language']}
    any_changed = False
    any_ascii_parentheses = False
    for side, data in sides.items():
        index = data['index'][key]
        row, changed = data['rows'][index], data['transformed'][index]
        original = {field: int(data['original'].iloc[index][field]) for field in FIELDS}
        diagnostic = {field: int(data['diagnostic'].iloc[index][field]) for field in FIELDS}
        norm_original = scorer.normalize_asr_pair(row['gts'], row['clean_output'], pd.Series(row), 'auto')
        norm_diagnostic = scorer.normalize_asr_pair(changed['gts'], changed['clean_output'], pd.Series(changed), 'auto')
        text_affected = row['gts'] != changed['gts'] or row['clean_output'] != changed['clean_output']
        case[side] = {
            'original': original, 'diagnostic': diagnostic,
            'error_change': diagnostic['errors'] - original['errors'],
            'ref_words_change': diagnostic['ref_words'] - original['ref_words'],
            'ascii_parentheses_present': text_affected,
            'reference': row['gts'], 'hypothesis': row['clean_output'],
            'transformed_reference': changed['gts'], 'transformed_hypothesis': changed['clean_output'],
            'original_normalized_reference': norm_original[0],
            'original_normalized_hypothesis': norm_original[1],
            'diagnostic_normalized_reference': norm_diagnostic[0],
            'diagnostic_normalized_hypothesis': norm_diagnostic[1],
        }
        any_changed |= original != diagnostic
        any_ascii_parentheses |= text_affected
    case['original_error_gap'] = case['target']['original']['errors'] - case['baseline']['original']['errors']
    case['diagnostic_error_gap'] = case['target']['diagnostic']['errors'] - case['baseline']['diagnostic']['errors']
    case['gap_change'] = case['diagnostic_error_gap'] - case['original_error_gap']
    case['metric_changed'] = any_changed
    if any_ascii_parentheses or any_changed:
        cases.append(case)

aggregates = {
    side: {'original': total(data['original']), 'diagnostic': total(data['diagnostic'])}
    for side, data in sides.items()
}
original_gap = aggregates['target']['original']['errors'] - aggregates['baseline']['original']['errors']
diagnostic_gap = aggregates['target']['diagnostic']['errors'] - aggregates['baseline']['diagnostic']['errors']
result = {
    'generated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'dataset': 'openml_bad', 'step': 440,
    'baseline_name': 'Fix', 'target_name': 'R64',
    'official_scoring_replaced': False,
    'transformation': "str.translate(str.maketrans('', '', '()')) on both raw gts and clean_output on both sides; all content inside parentheses is preserved",
    'scorer': {'path': str(SCORER), 'sha256': sha256(SCORER), 'normalizer': 'auto', 'language_column': 'language', 'method': 'compute_metrics, exact same English/OpenASR dispatch as full report'},
    'sources': sources,
    'complete_matched_rows': 393,
    'all_original_row_errors_and_reference_counts_match_logged': True,
    'aggregates': aggregates,
    'original_target_minus_baseline_errors': original_gap,
    'diagnostic_target_minus_baseline_errors': diagnostic_gap,
    'gap_change': diagnostic_gap - original_gap,
    'rows_with_ascii_parentheses_on_either_side': len(cases),
    'rows_with_metric_change_on_either_side': sum(c['metric_changed'] for c in cases),
    'cases': sorted(cases, key=lambda c: (c['gap_change'], c['id'])),
}
path = OUT / 'parentheses_diagnostic.json'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, ensure_ascii=False, indent=2))

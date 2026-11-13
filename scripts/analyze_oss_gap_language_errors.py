"""Incremental, coverage-pinned language statistics and analytical error Top-K.

Reads generated JSONL only; never decodes audio or publishes training data.
Each immutable snapshot keeps exact frequency tables, deterministic ranked raw
records and an incremental ledger. Partial coverage is explicitly provisional.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import heapq
import json
import math
from pathlib import Path
import re

import blobfile as bf


class Candidate:
    def __init__(self, record):
        self.record = record
        self.key = (-int(record['n_err']), record['audio_chunk'])

    def __lt__(self, other):
        # A min-heap with the worst retained candidate at its root.
        return self.key > other.key


def offer(heap, record, k):
    candidate = Candidate(record)
    if len(heap) < k:
        heapq.heappush(heap, candidate)
    elif candidate.key < heap[0].key:
        heapq.heapreplace(heap, candidate)


def describe(counter):
    values = sorted((float(k), n) for k, n in counter.items())
    total = sum(n for _, n in values)
    if not total:
        return {k: None for k in ['mean', 'min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']}
    result = {'mean': math.fsum(v * n for v, n in values) / total,
              'min': values[0][0], 'max': values[-1][0]}
    for label, p in [('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9), ('p95', .95), ('p99', .99)]:
        target, running = math.ceil(p * total), 0
        for value, n in values:
            running += n
            if running >= target:
                result[label] = value
                break
    return result


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args):
    case = args.case_dir.resolve()
    state = json.loads((case / 'state.json').read_text())
    coverage_path = args.coverage.resolve() if args.coverage else max(case.glob('coverage_*.json'), key=lambda p: p.stat().st_mtime)
    coverage = json.loads(coverage_path.read_text())
    raw_sources = (case / 'source_chunks.json').read_bytes()
    source_sha = hashlib.sha256(raw_sources).hexdigest()
    assert source_sha == state['source_chunks_sha256']
    sources = {r['audio_key']: r for r in json.loads(raw_sources)}
    universe = collections.Counter()
    names = {}
    for source in sources.values():
        universe[source['language']] += source['count']
        names[source['language']] = source['display_language']
    assert sum(universe.values()) == coverage['expected_rows']
    parts = coverage['parts']
    assert [int(re.search(r'part-(\d+)\.jsonl$', p['part']).group(1)) for p in parts] == list(range(len(parts)))
    output_root = (args.output or case / 'language_error_analysis').resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tag = datetime.datetime.fromisoformat(coverage['observed_utc']).strftime('%Y%m%dT%H%M%SZ')
    dest = output_root / f'{tag}_{coverage["verified_unique_rows"]}'
    if (dest / 'summary.json').exists():
        print(json.dumps({'status': 'EXISTS', 'snapshot': str(dest)}), flush=True)
        return
    if dest.exists():
        raise FileExistsError(f'Incomplete prior analysis; inspect before retry: {dest}')
    dest.mkdir()
    pointer = output_root / 'latest.json'
    ledger = {'version': 1, 'source_sha256': source_sha, 'model': state['selection_model'],
              'top_k': args.top_k, 'parts': {}, 'masks': {}, 'languages': {}}
    heaps = collections.defaultdict(list)
    if pointer.exists():
        previous = Path(json.loads(pointer.read_text())['snapshot'])
        ledger = json.loads((previous / 'ledger.json').read_text())
        assert ledger['version'] == 1 and ledger['source_sha256'] == source_sha
        assert ledger['model'] == state['selection_model'] and ledger['top_k'] == args.top_k
        assert set(ledger['parts']) <= {p['part'] for p in parts}
        for f in (previous / 'top10000').glob('*.jsonl'):
            for line in f.open():
                r = json.loads(line)
                offer(heaps[r['source_language']], r, args.top_k)
    masks = {a: int(v, 16) for a, v in ledger['masks'].items()}
    for audio, bits in masks.items():
        assert audio in sources and bits >= 0 and bits.bit_length() <= sources[audio]['count']
    stats = ledger['languages']
    for lang in universe:
        stats.setdefault(lang, {'decoded': 0, 'errors': 0, 'families': {},
                                'duration': {}, 'word_count': {}, 'character_count': {},
                                'error_count': {}, 'scored_reference_units': {}})
        for field in ['families', 'duration', 'word_count', 'character_count', 'error_count', 'scored_reference_units']:
            stats[lang][field] = collections.Counter(stats[lang][field])
    reused = len(ledger['parts'])
    for index, part in enumerate(parts):
        path = part['part']
        st = bf.stat(path)
        signature = {'size': st.size, 'version': getattr(st, 'version', None), 'mtime': st.mtime}
        assert signature == part['signature'], ('Frozen part changed', path)
        if path in ledger['parts']:
            assert ledger['parts'][path] == part
            if signature['version']:
                continue
            with bf.BlobFile(path, 'rb') as f:
                assert hashlib.sha256(f.read()).hexdigest() == part['sha256']
            continue
        digest, n = hashlib.sha256(), 0
        with bf.BlobFile(path, 'rb') as f:
            for line in f:
                digest.update(line)
                r = json.loads(line)
                audio, count, idx = r['audio_chunk'].rsplit(':', 2)
                count, idx = int(count), int(idx)
                source = sources[audio]
                assert count == source['count'] and 0 <= idx < count
                assert r['audio_chunk'] == f'{audio}:{count}:{idx}'
                bit = 1 << idx
                assert not masks.get(audio, 0) & bit, ('Duplicate ID', r['audio_chunk'])
                masks[audio] = masks.get(audio, 0) | bit
                lang = source['language']
                assert r['source_language'] == lang and r['language'] == source['display_language']
                assert r['source_family'] == source['family']
                assert r['selection_model'] == state['selection_model'] and r['selection_scorer'] == 'oss_gap_error_mining_v1'
                assert r['selection_metric'] == source['selection_metric']
                for key in ['n_err', 'n_ref', 'n_ins', 'n_del', 'n_sub']:
                    v = r[key]
                    assert not isinstance(v, bool) and isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 and int(v) == v
                assert r['n_err'] == r['n_ins'] + r['n_del'] + r['n_sub']
                assert isinstance(r['text'], str)
                item = stats[lang]
                item['decoded'] += 1
                n += 1
                if r['n_err'] <= 0:
                    continue
                duration = r['audio_duration_seconds']
                assert not isinstance(duration, bool) and isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
                words = len(r['text'].split())
                characters = sum(not c.isspace() for c in r['text'])
                item['errors'] += 1
                item['families'][source['family']] += 1
                for field, value in [('duration', duration), ('word_count', words), ('character_count', characters),
                                     ('error_count', int(r['n_err'])), ('scored_reference_units', int(r['n_ref']))]:
                    item[field][str(value)] += 1
                # Avoid keeping unneeded records in memory; compare before copying.
                key = (-int(r['n_err']), r['audio_chunk'])
                heap = heaps[lang]
                if len(heap) < args.top_k or key < heap[0].key:
                    r.update(analysis_word_count=words, analysis_character_count=characters)
                    offer(heap, r, args.top_k)
        assert n == part['rows'] and digest.hexdigest() == part['sha256']
        assert bf.stat(path).size == st.size
        ledger['parts'][path] = part
        print(f'part {index + 1}/{len(parts)} verified; decoded={sum(x["decoded"] for x in stats.values())}', flush=True)
    assert sum(x['decoded'] for x in stats.values()) == coverage['verified_unique_rows'] == sum(v.bit_count() for v in masks.values())
    assert sum(x['errors'] for x in stats.values()) == coverage['n_err_positive_partial']
    (dest / 'top10000').mkdir()
    (dest / 'distributions').mkdir()
    rows = []
    summary = {'case_id': state['case_id'], 'observed_utc': coverage['observed_utc'],
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'coverage_evidence': str(coverage_path), 'source_sha256': source_sha,
               'selection_model': state['selection_model'], 'scorer': 'oss_gap_error_mining_v1',
               'expected_rows': coverage['expected_rows'], 'decoded_unique': coverage['verified_unique_rows'],
               'error_rows_partial': coverage['n_err_positive_partial'],
               'complete_generation': coverage['verified_unique_rows'] == coverage['expected_rows'],
               'training_publication': False, 'top_k': args.top_k,
               'ranking': 'absolute n_err descending; canonical audio_chunk ascending; only n_err>0',
               'word_count_method': 'len(original human reference text.split()); whitespace tokens; not linguistic words for unspaced languages',
               'character_count_method': 'non-whitespace Unicode code points in original reference',
               'duration_method': 'audio_duration_seconds measured from actual decoded waveform; seconds; no audio re-download',
               'quantile_method': 'exact frequency tables, nearest-rank ceil(p*N)',
               'metric_caveat': 'zh/yue CER character edits, others whitespace WER word edits; do not aggregate into one WER',
               'audit_limits': 'Original reference equality, waveform/parent alias and full heldout clearance pending pipeline final audit; analytical subsets only',
               'parts_reused': reused, 'new_parts_read': len(parts) - reused, 'languages': {}}
    for lang in sorted(universe):
        item = stats[lang]
        complete = item['decoded'] == universe[lang]
        status = 'COMPLETE_LANGUAGE' if complete else ('PARTIAL_LANGUAGE' if item['decoded'] else 'NOT_DECODED')
        descriptions = {field: describe(item[field]) for field in ['duration', 'word_count', 'character_count', 'error_count', 'scored_reference_units']}
        error_hours = math.fsum(float(v) * n for v, n in item['duration'].items()) / 3600 if item['decoded'] else None
        selected = sorted(heaps[lang], key=lambda x: x.key)
        assert len(selected) == min(args.top_k, item['errors'])
        top_path, top_hash = None, None
        if selected:
            top_path = f'top10000/{lang}.jsonl'
            h = hashlib.sha256()
            with (dest / top_path).open('wb') as f:
                for rank, candidate in enumerate(selected, 1):
                    r = {**candidate.record, 'analysis_language_rank': rank,
                         'analysis_selection_status': 'LANGUAGE_TOPK_COMPLETE' if complete else 'PROVISIONAL_PREFIX_TOPK'}
                    payload = (json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n').encode()
                    f.write(payload)
                    h.update(payload)
            top_hash = h.hexdigest()
        detail = {'language': lang, 'language_name': names[lang], 'status': status,
                  'expected_samples': universe[lang], 'decoded_samples': item['decoded'],
                  'coverage_pct': item['decoded'] / universe[lang] * 100,
                  'error_samples_observed': item['errors'] if item['decoded'] else None,
                  'error_rate_observed_pct': item['errors'] / item['decoded'] * 100 if item['decoded'] else None,
                  'error_duration_hours': error_hours, 'top_count': len(selected),
                  'top_path': top_path, 'top_sha256': top_hash,
                  'top_min_n_err': int(selected[-1].record['n_err']) if selected else None,
                  'top_max_n_err': int(selected[0].record['n_err']) if selected else None,
                  'top_duration_hours': math.fsum(x.record['audio_duration_seconds'] for x in selected) / 3600 if selected else None,
                  'families': dict(item['families']), 'statistics': descriptions}
        summary['languages'][lang] = detail
        row = {k: detail[k] for k in ['language', 'language_name', 'status', 'expected_samples', 'decoded_samples',
                                      'coverage_pct', 'error_samples_observed', 'error_rate_observed_pct', 'error_duration_hours', 'top_count', 'top_min_n_err', 'top_max_n_err']}
        for field, desc in descriptions.items():
            row.update({field + '_' + k: v for k, v in desc.items()})
        rows.append(row)
        if item['errors']:
            for field in descriptions:
                values = sorted(item[field].items(), key=lambda x: float(x[0]))
                write_csv(dest / 'distributions' / f'{lang}_{field}_frequency.csv',
                          [{'value': v, 'samples': n, 'fraction': n / item['errors']} for v, n in values],
                          ['value', 'samples', 'fraction'])
    write_csv(dest / 'language_summary.csv', rows, list(rows[0]))
    ledger['masks'] = {a: hex(v) for a, v in masks.items()}
    (dest / 'ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, separators=(',', ':')))
    (dest / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    tmp = pointer.with_suffix('.tmp')
    tmp.write_text(json.dumps({'snapshot': str(dest), 'coverage_evidence': str(coverage_path)}, indent=2))
    tmp.replace(pointer)
    print(json.dumps({'status': 'COMPLETE_ANALYSIS_SNAPSHOT', 'snapshot': str(dest),
                      'decoded': summary['decoded_unique'], 'errors': summary['error_rows_partial'],
                      'languages_with_results': [k for k, v in summary['languages'].items() if v['decoded_samples']]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir', type=Path, default=Path('tmp/run_openasr_exp/oss_gap_2611r2_errors_20261112'))
    p.add_argument('--coverage', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--top-k', type=int, default=10000)
    a = p.parse_args()
    if a.top_k <= 0:
        p.error('--top-k must be positive')
    run(a)

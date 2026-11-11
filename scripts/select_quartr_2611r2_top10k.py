#!/usr/bin/env python3
"""Freeze the global top 10k by absolute canonical error count, without decoding."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import blobfile as bf
import yaml

from scripts.filter_quartr_2611r2_errors import SOURCES


def sha(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.report_dir)
    out = root / 'top10k'
    out.mkdir(exist_ok=True)
    parent = json.loads((root / 'summary.json').read_text())
    originals, scores, inputs = {}, [], {}
    for source in SOURCES:
        train_bytes = (root / f'{source}.jsonl').read_bytes()
        audit_bytes = (root / f'{source}.audit.jsonl').read_bytes()
        info = parent['sources'][source]
        assert sha(train_bytes) == info['sha256'], source
        assert sha(audit_bytes) == info['audit_sha256'], source
        rows = [json.loads(line) for line in train_bytes.splitlines()]
        audits = [json.loads(line) for line in audit_bytes.splitlines()]
        assert len(rows) == info['selected'] and len(audits) == info['total']
        raw = {row['audio_path']: row for row in rows}
        assert len(raw) == len(rows)
        eligible = set()
        for row in audits:
            assert row['source'] == source
            assert row['n_err'] == row['n_ins'] + row['n_del'] + row['n_sub']
            if row['n_err'] > 0:
                pointer = row['audio_path']
                assert pointer in raw and raw[pointer]['id'] == row['id']
                assert pointer not in originals
                originals[pointer] = raw[pointer]
                eligible.add(pointer)
                scores.append(row)
        assert eligible == set(raw)
        inputs[source] = {'training_sha256': sha(train_bytes), 'audit_sha256': sha(audit_bytes)}
    assert len(scores) == 16235 and sum(x['total'] for x in parent['sources'].values()) == 25240
    key = lambda row: (-row['n_err'], row['source'], str(row['id']), row['audio_path'])
    scores.sort(key=key)
    selected = scores[:10000]
    assert len(selected) == 10000 and len({r['audio_path'] for r in selected}) == 10000
    assert key(selected[-1]) < key(scores[10000])
    artifacts, source_counts = {}, Counter(r['source'] for r in selected)
    ranked = [dict(row, global_rank=i) for i, row in enumerate(selected, 1)]
    artifacts['ranking.audit.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ranked).encode()
    source_info = {}
    for source in SOURCES:
        chosen = [r for r in selected if r['source'] == source]
        content = ''.join(json.dumps(originals[r['audio_path']], ensure_ascii=False) + '\n' for r in chosen).encode()
        artifacts[f'filtered/{source}.jsonl'] = content
        source_info[source] = {'selected': len(chosen), 'sha256': sha(content),
            'hours': sum(r['audio_duration_seconds'] for r in chosen) / 3600,
            'n_err': sum(r['n_err'] for r in chosen),
            'n_ref': sum(r['n_ref'] for r in chosen),
            'response_at_limit': sum(bool(r['response_at_limit']) for r in chosen)}
        spec_path = Path(f'recipe/phimm/config/data/train_data/{source}_2611r2_errors.yaml')
        specs = yaml.safe_load(spec_path.read_text())
        specs[0]['jsonl_paths'] = args.output.rstrip('/') + f'/filtered/{source}.jsonl'
        new_path = spec_path.with_name(f'{source}_2611r2_errors_top10k.yaml')
        text = (f'# Global top 10,000 by absolute n_err; {len(chosen):,} rows from {source}.\n'
                f'# SHA256: {sha(content)}; original references and audio selectors preserved.\n'
                + yaml.safe_dump(specs, sort_keys=False))
        new_path.write_text(text)
        artifacts[f'configs/{new_path.name}'] = text.encode()
    parent_path = Path('recipe/phimm/config/v2611_dev/earning/remax_earning_base4c_errors.yaml')
    recipe = parent_path.read_text()
    start = recipe.index('# Training dataset:')
    end = recipe.index('data:\n', start)
    comments = '# Training dataset: global top 10,000 by descending absolute canonical n_err.\n'
    comments += '# Ties: source name, string sample ID, absolute audio path (ascending).\n'
    comments += '# Per-source files follow global rank; training remains shuffled.\n'
    for source in SOURCES:
        comments += f'# {source}: {source_counts[source]:,} samples ({source_counts[source]/100:.2f}%).\n'
    recipe = recipe[:start] + comments + recipe[end:]
    recipe = recipe.replace('# Full25240-row coverage and canonical rescoring verified; original training settings retained.',
        '# Ranked from complete 25,240-row decoding; original training settings retained.')
    for source in SOURCES:
        recipe = recipe.replace(f'/data/train_data/{source}_2611r2_errors@',
                                f'/data/train_data/{source}_2611r2_errors_top10k@')
    recipe = recipe.replace('experiment_name: remax_earning_base4c_errors',
                            'experiment_name: remax_earning_base4c_errors_top10k')
    recipe_path = parent_path.with_name('remax_earning_base4c_errors_top10k.yaml')
    recipe_path.write_text(recipe)
    artifacts[f'configs/{recipe_path.name}'] = recipe.encode()
    cutoff = selected[-1]['n_err']
    summary = {'selection': 'global top 10000 by absolute n_err, no source quotas',
        'tie_break': ['source ASC', 'string id ASC', 'audio_path ASC'],
        'decoded_universe': 25240, 'eligible_errors': len(scores), 'selected': len(selected),
        'minimum_n_err': cutoff, 'maximum_n_err': selected[0]['n_err'],
        'above_cutoff': sum(r['n_err'] > cutoff for r in scores),
        'at_cutoff_total': sum(r['n_err'] == cutoff for r in scores),
        'at_cutoff_selected': sum(r['n_err'] == cutoff for r in selected),
        'source_counts': dict(source_counts), 'sources': source_info, 'input_hashes': inputs,
        'recipe': str(recipe_path), 'training_shuffle': True,
        'membership_sha256': sha('\n'.join(r['audio_path'] for r in selected).encode()),
        'artifacts_sha256': {path: sha(content) for path, content in artifacts.items()},
        'limitations': parent['limitations']}
    artifacts['selection-summary.json'] = (json.dumps(summary, indent=2) + '\n').encode()
    for relative, content in artifacts.items():
        local = out / relative
        local.parent.mkdir(exist_ok=True, parents=True)
        local.write_bytes(content)
        uri = args.output.rstrip('/') + '/' + relative
        with bf.BlobFile(uri, 'wb') as stream:
            stream.write(content)
        with bf.BlobFile(uri, 'rb') as stream:
            assert sha(stream.read()) == sha(content), uri
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

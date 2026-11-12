#!/usr/bin/env python3
"""Freeze four separate WER>0 source subsets after complete 2611r2 generation.

Original JSON records and label fields survive; only audio_path becomes absolute.
Hypotheses and scores belong to audit files, never to the training labels.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import blobfile as bf
import yaml

SOURCES = ('earn_fy27', 'quartr_1st23', 'quartr_2nd48', 'quartr_3rd38')


def sha(content):
    return hashlib.sha256(content).hexdigest()


def read_rows(path):
    with bf.BlobFile(str(path), 'rb') as stream:
        content = stream.read()
    return [json.loads(line) for line in content.splitlines()], sha(content)


def freeze(args):
    from recipe.phimm.data.dataset import create_audio_dataset
    from recipe.phimm.reward.asr_eval import measure_openasr_en_wer

    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((report / 'source-inventory.json').read_text())
    expected = {}
    original_rows = {name: [] for name in SOURCES}
    for item in inventory:
        rows, digest = read_rows(item['uri'])
        if digest != item['sha256'] or len(rows) != item['count']:
            raise ValueError(f"Source changed: {item['uri']}")
        spec = copy.deepcopy(item['spec'])
        # Exercise the native source preprocessing; do not reimplement tag cleaning.
        spec.pop('post_process', None)
        spec.pop('add_task_info', None)
        canonical = create_audio_dataset(**{**spec, 'version': 2609})
        if len(canonical) != len(rows):
            raise ValueError('Native source preprocessing changed row count')
        for index, (raw, clean) in enumerate(zip(rows, canonical)):
            pointer = clean['audio_path']
            if pointer in expected:
                raise ValueError(f'Duplicate source audio identity: {pointer}')
            output = dict(raw, audio_path=pointer)
            expected[pointer] = dict(source=item['source'], part=item['part'],
                                     source_row=index, raw=output, text=clean['text'])
            original_rows[item['source']].append(pointer)

    paths = sorted(bf.glob(args.generation.rstrip('/') + '/part-*.jsonl'),
                   key=lambda p: int(re.search(r'part-(\d+)\.jsonl$', p).group(1)))
    indices = [int(re.search(r'part-(\d+)\.jsonl$', p).group(1)) for p in paths]
    if not paths or indices != list(range(len(paths))):
        raise ValueError('Generation parts are missing or noncontiguous')
    scores = {}
    part_hashes = {}
    for path in paths:
        rows, digest = read_rows(path)
        part_hashes[path] = digest
        for row in rows:
            pointer = row.get('audio_path')
            if pointer not in expected or pointer in scores:
                raise ValueError(f'Unexpected or duplicated generated audio: {pointer}')
            src = expected[pointer]
            if row['text'] != src['text']:
                raise ValueError(f'Generation reference differs from native source: {pointer}')
            score = measure_openasr_en_wer(row['response'], src['text'])
            if any(row[key] != score[key] for key in ('n_err', 'n_ref', 'wer')):
                raise ValueError(f'Canonical rescore mismatch: {pointer}')
            scores[pointer] = dict(row, **score)
    if scores.keys() != expected.keys():
        raise ValueError(f'Incomplete generation: {len(scores)}/{len(expected)}; no subsets published')

    summary = {'expected_count': len(expected), 'decoded_count': len(scores),
               'selection': 'n_err > 0, no upper WER cap, retain token-limit rows',
               'scorer': 'measure_openasr_en_wer, HF English normalization, merge_compounds=True',
               'generation_parts_sha256': part_hashes, 'sources': {},
               'limitations': ['No audio-content or cross-corpus speaker leakage proof.',
                              'Source human labels preserved, including known review/trim caveats.',
                              'Mining only; no refinement training or measured fix.']}
    pending = {}
    for source, pointers in original_rows.items():
        keep = [p for p in pointers if scores[p]['n_err'] > 0]
        if not keep:
            raise ValueError(f'No errors in {source}; an empty DataSpec needs explicit handling')
        train_content = ''.join(json.dumps(expected[p]['raw'], ensure_ascii=False) + '\n' for p in keep).encode()
        audit_content = ''.join(json.dumps(dict(scores[p], source=source,
                            source_part=expected[p]['part'], source_row=expected[p]['source_row'],
                            selected=scores[p]['n_err'] > 0), ensure_ascii=False) + '\n'
                            for p in pointers).encode()
        out = args.output.rstrip('/') + f'/filtered/{source}.jsonl'
        audit = args.output.rstrip('/') + f'/audit/{source}.jsonl'
        pending[out] = train_content
        pending[audit] = audit_content
        summary['sources'][source] = {
            'total': len(pointers), 'selected': len(keep), 'retention_rate': len(keep)/len(pointers),
            'n_err': sum(scores[p]['n_err'] for p in pointers),
            'n_ref': sum(scores[p]['n_ref'] for p in pointers),
            'audio_hours': sum(scores[p]['audio_duration_seconds'] for p in pointers)/3600,
            'selected_audio_hours': sum(scores[p]['audio_duration_seconds'] for p in keep)/3600,
            'response_at_limit': sum(bool(scores[p]['response_at_limit']) for p in pointers),
            'training_jsonl': out, 'sha256': sha(train_content), 'audit': audit,
            'audit_sha256': sha(audit_content),
            'membership_sha256': sha('\n'.join(keep).encode())}

    # Publish only after all joins and canonical rescoring pass.
    for uri, content in pending.items():
        bf.makedirs(bf.dirname(uri))
        with bf.BlobFile(uri, 'wb') as stream:
            stream.write(content)
        with bf.BlobFile(uri, 'rb') as stream:
            if sha(stream.read()) != sha(content):
                raise ValueError(f'Output read-back hash mismatch: {uri}')
        (report / bf.basename(uri).replace('.jsonl', '.audit.jsonl') if '/audit/' in uri
         else report / bf.basename(uri)).write_bytes(content)
    summary['loader_checks'] = {}
    for source in SOURCES:
        path = Path(f'recipe/phimm/config/data/train_data/{source}_2611r2_errors.yaml')
        specs = yaml.safe_load(path.read_text())
        ds = create_audio_dataset(**{**specs[0], 'version': 2609})
        if len(ds) != summary['sources'][source]['selected']:
            raise ValueError(f'Filtered loader count mismatch for {source}')
        kept_pointers = [p for p in original_rows[source] if scores[p]['n_err'] > 0]
        for row, pointer in zip(ds, kept_pointers):
            if row['audio_path'] != pointer or row['reward_model']['ground_truth'] != expected[pointer]['text']:
                raise ValueError(f'Filtered loader changed audio or reference: {pointer}')
        # Test real audio decoding at both ends, preserving the original selector.
        from recipe.phimm.utils.audio import load_audio
        audio_checks = []
        for index in dict.fromkeys((0, len(ds)-1)):
            waveform, sample_rate = load_audio(dict(ds[index]), None)
            if len(waveform) == 0 or sample_rate <= 0:
                raise ValueError('Filtered loader returned empty audio')
            audio_checks.append({'index': index, 'samples': len(waveform), 'sample_rate': sample_rate})
        summary['loader_checks'][source] = {'count': len(ds), 'samples': audio_checks}
        with bf.BlobFile(args.output.rstrip('/') + f'/configs/{path.name}', 'w') as stream:
            stream.write(path.read_text())

    (report / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Quartr 2611r2 error mining', '',
             'Selection: canonical English compound-aware `n_err > 0`. Original human labels retained.', '',
             '| Source | Decoded | Kept | Retention | Corpus WER | Kept hours | Token limit |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for source, data in summary['sources'].items():
        lines.append(f"| {source} | {data['total']} | {data['selected']} | {data['retention_rate']:.2%} | {data['n_err']/data['n_ref']:.3%} | {data['selected_audio_hours']:.2f} | {data['response_at_limit']} |")
    lines += ['', 'Original references and source metadata are preserved; generated responses are stored separately in audit JSONL.',
              'Known source trim/review caveats remain. No refinement was trained and no measured fix is claimed.',
              'Source membership and audio-pointer uniqueness were checked. Audio-content and speaker leakage were not established.']
    (report / 'report.md').write_text('\n'.join(lines)+'\n')
    for name in ('summary.json', 'report.md'):
        with bf.BlobFile(args.output.rstrip('/') + '/' + name, 'w') as stream:
            stream.write((report/name).read_text())
    with bf.BlobFile(args.output.rstrip('/') + '/COMPLETE.json', 'w') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-dir', required=True)
    freeze(parser.parse_args())

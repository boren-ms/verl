"""Prepare private per-language Top-K training candidates from verified analysis.

This command never launches training or certifies incomplete generation as final.
The reference is the frozen effective Main Base3a configuration, with only
training data and unique output identities overridden.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    case = args.case.resolve()
    snapshot = args.snapshot.resolve()
    summary = json.loads((snapshot / 'summary.json').read_text())
    verified = json.loads((snapshot / 'verification.json').read_text())
    state = json.loads((case / 'state.json').read_text())
    assert verified['status'] == 'PASSED' and verified['decoded'] == summary['decoded_unique']
    assert summary['source_sha256'] == state['source_chunks_sha256']
    assert summary['selection_model'] == state['selection_model'] and summary['top_k'] == 10000
    raw_sources = (case / 'source_chunks.json').read_bytes()
    assert hashlib.sha256(raw_sources).hexdigest() == summary['source_sha256']
    sources = {r['audio_key']: r for r in json.loads(raw_sources)}
    dest = case / 'base3a_top10k_candidates' / snapshot.name
    if dest.exists():
        raise FileExistsError(f'Candidate already exists; reconcile instead of overwriting: {dest}')
    dest.mkdir(parents=True)
    blob = state['output_root'].rstrip('/') + '/private_top10k_per_lang/' + snapshot.name
    files, languages, total = [], {}, 0
    for lang, info in sorted(summary['languages'].items()):
        if not info['top_path']:
            continue
        top = snapshot / info['top_path']
        assert sha(top) == info['top_sha256']
        records = [json.loads(line) for line in top.open()]
        assert len(records) == min(10000, info['error_samples_observed'])
        keys = [(-r['n_err'], r['audio_chunk']) for r in records]
        assert keys == sorted(keys) and len({r['audio_chunk'] for r in records}) == len(records)
        path = dest / f'train-{lang}.jsonl'
        with path.open('w') as f:
            for rank, r in enumerate(records, 1):
                assert r['n_err'] > 0 and r['source_language'] == lang
                audio, count, idx = r['audio_chunk'].rsplit(':', 2)
                source = sources[audio]
                assert source['language'] == lang and int(count) == source['count'] and 0 <= int(idx) < int(count)
                # All text comes from the original generation reference, never hypothesis.
                row = {'id': r['audio_chunk'], 'audio_chunk': r['audio_chunk'], 'text': r['text'],
                       'language': r['language'], 'source_language': lang, 'source_family': source['family'],
                       'source_split_status': source['split_status'], 'selection_language_rank': rank,
                       'selection_n_err': r['n_err'], 'selection_n_ref': r['n_ref'],
                       'selection_metric': r['selection_metric'], 'selection_model': r['selection_model'],
                       'audio_duration_seconds': r['audio_duration_seconds'],
                       'selection_status': 'PROVISIONAL_NOT_AUDITED_FOR_FINAL_TRAINING'}
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        files.append(path)
        total += len(records)
        languages[lang] = {'rows': len(records), 'file': path.name, 'sha256': sha(path),
                           'coverage_status': info['status'], 'min_n_err': info['top_min_n_err'],
                           'max_n_err': info['top_max_n_err'], 'hours': info['top_duration_hours']}
    assert total > 0
    extra = ['source_language', 'source_family', 'source_split_status', 'selection_language_rank',
             'selection_n_err', 'selection_n_ref', 'selection_metric', 'selection_model', 'selection_status',
             'audio_duration_seconds']
    def spec(paths):
        return [{'dataset_name': 'jsonl', 'jsonl_paths': paths, 'num_proc': 8,
                 'add_task_info': {'task': 'lang_asr_verb', 'prefix_prob': 0.0, 'lang_hint': True},
                 'post_process': {'add_field': {'fields': {'data_source': 'openml'}},
                                  'verl_format': {'prompt_key': 'prompt', 'extra_keys': extra}}}]
    data = spec([blob + '/' + p.name for p in files])
    banner = '# PRIVATE PROVISIONAL CANDIDATE. NOT FOR TRAINING LAUNCH. Full coverage/reference/split/heldout audit pending.\n'
    (dest / 'training_data.yaml').write_text(banner + yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    for lang, info in languages.items():
        (dest / f'training_data_{lang}.yaml').write_text(banner + yaml.safe_dump(spec([blob + '/' + info['file']]), allow_unicode=True, sort_keys=False))
    config_root = args.repo.resolve() / 'recipe/phimm/config'
    data_name = 'oss_gap_top10k_per_lang_2611r2_PROVISIONAL'
    data_path = config_root / 'data/train_data/oss' / (data_name + '.yaml')
    data_path.write_text((dest / 'training_data.yaml').read_text())
    experiment = 'remax_main_base3a_2611r2_oss_gap_top10k_per_lang_' + snapshot.name
    recipe = {'hydra': {'searchpath': ['file://recipe/phimm/config', 'file://verl/trainer/config']},
              'defaults': ['references/base3a_2611r2_effective_20261113@_here_',
                           f'/data/train_data/oss/{data_name}@_oss_gap_top10k_train', '_self_'],
              'data': {'train_data': ['${_oss_gap_top10k_train}'],
                       'mix': [{'shuffle': True, 'datasource': [{'name': 'openml', 'sample': total, 'shuffle': True}]}]},
              'trainer': {'experiment_name': experiment}}
    recipe_path = config_root / 'v2611_dev/main/remax_main_base3a_2611r2_oss_gap_top10k_per_lang_PROVISIONAL.yaml'
    recipe_path.write_text(banner + yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False))
    (dest / 'training_recipe.yaml').write_text(recipe_path.read_text())
    gate = {'status': 'PRIVATE_PROVISIONAL_PREPARED_NOT_LAUNCHABLE', 'training_published': False,
            'training_submitted': False, 'analysis_snapshot': str(snapshot),
            'selection': 'per ISO language n_err>0, absolute n_err DESC, canonical audio_chunk ASC; min(10000,N)',
            'source_sha256': summary['source_sha256'], 'selection_model': summary['selection_model'],
            'complete_generation': summary['complete_generation'], 'languages': languages,
            'unobserved_languages': [k for k,v in summary['languages'].items() if not v['decoded_samples']],
            'rows': total, 'native_verified_rows': None, 'candidate_derived_steps_if_native_count_unchanged': 2 * (total // 128),
            'final_absolute_optimizer_target': None, 'candidate_blob_root': blob,
            'data_config': str(data_path), 'recipe_config': str(recipe_path),
            'parent_reference': 'recipe/phimm/config/v2611_dev/main/remax_main_base3a_2611r2.yaml from main workspace',
            'reference_preserved': True, 'validation_changed': False,
            'blocking_gates': ['complete generation and unique identity coverage',
                               'full original reference/source/split/heldout audit',
                               'select TopK AFTER eligibility exclusions, refill quotas from eligible full error pool',
                               'final native loader/audio/reward verification, data-only effective diff and fixed final optimizer target'],
            'reward_route': 'openml is the existing unchanged Base3a librispeech reward alias; actual JSONL contains only the new OSS candidates'}
    (dest / 'PUBLICATION_BLOCKED.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2))
    print(json.dumps({'candidate_dir': str(dest), 'rows': total, 'languages': list(languages),
                      'data_config': str(data_path), 'recipe_config': str(recipe_path), 'blob_root': blob}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, default=Path('tmp/run_openasr_exp/oss_gap_2611r2_errors_20261112'))
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    run(p.parse_args())

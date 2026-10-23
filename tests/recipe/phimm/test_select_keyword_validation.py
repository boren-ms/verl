import json

import pytest

from recipe.phimm.select_keyword_validation import select


def fixture(tmp_path):
    originals, results = [], []
    # Equal BWER is resolved by WER, then ID; zero-keyword rows are ineligible.
    for uid, ne, nb, nr, nbr in [('z', 1, 0, 10, 2), ('b', 0, 0, 10, 2),
                                  ('a', 0, 0, 10, 2), ('c', 1, 1, 10, 3), ('empty', 0, 0, 10, 0)]:
        original = dict(id=uid, text='reference', keywords=['word'] if nbr else [],
                        split='dev', source_manifest_sha='manifest', annotation_policy_sha='policy')
        originals.append(original)
        results.append(dict(original, gts=original['text'], data_source='ls_dev', step=0,
                            n_err=ne, n_ref=nr, nb_err=nb, nb_ref=nbr, nu_err=ne-nb, nu_ref=nr-nbr))
    paths = [tmp_path / 'manifest.jsonl', tmp_path / 'baseline.jsonl']
    for path, rows in zip(paths, [originals, results], strict=True):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return paths, originals, results


def test_ranking_excludes_undefined_and_preserves_records(tmp_path):
    paths, originals, _ = fixture(tmp_path)
    selected, report = select(*paths, count=3, expected_count=5)
    assert [r['id'] for r in selected] == ['a', 'b', 'z']
    assert selected[0] == originals[2]
    assert report['excluded_undefined_bwer'] == 1
    assert report['baseline_corpus']['BWER'] == 0
    assert report['baseline_corpus']['WER'] == pytest.approx(1 / 30)
    with pytest.raises(ValueError, match='eligible'):
        select(*paths, count=5, expected_count=5)


@pytest.mark.parametrize('change,error', [
    ({'step': 1}, 'step-0'), ({'split': 'test-clean'}, 'dev only'),
    ({'data_source': 'ls_clean'}, 'step-0'), ({'gts': 'wrong'}, 'reference'),
    ({'keywords': []}, 'keyword'), ({'nb_ref': 3}, 'partition'),
    ({'nb_err': -1}, 'counts'), ({'source_manifest_sha': 'other'}, 'mismatch'),
])
def test_rejects_contamination_or_mismatched_results(tmp_path, change, error):
    paths, _, results = fixture(tmp_path)
    results[0].update(change)
    paths[1].write_text(''.join(json.dumps(r) + '\n' for r in results))
    with pytest.raises(ValueError, match=error):
        select(*paths, count=3, expected_count=5)


def test_rejects_incomplete_and_duplicate_baseline(tmp_path):
    paths, _, results = fixture(tmp_path)
    paths[1].write_text(''.join(json.dumps(r) + '\n' for r in results[:-1]))
    with pytest.raises(ValueError, match='entire'):
        select(*paths, count=3, expected_count=5)
    paths[1].write_text(''.join(json.dumps(r) + '\n' for r in [*results, results[0]]))
    with pytest.raises(ValueError, match='duplicate'):
        select(*paths, count=3, expected_count=5)


def test_training_uses_small_dev_and_official_tests_stay_separate():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).parents[3]
    config_root = root / 'recipe/phimm/config'
    with initialize_config_dir(config_dir=str(config_root / 'v2609_kw/training'), version_base=None):
        cfg = compose(config_name='remax_ls_joint_pilot', overrides=[
            f'hydra.searchpath=[file://{config_root},file://{root}/verl/trainer/config]',
        ])
    assert len(cfg.data.val_data) == 1
    source = cfg.data.val_data[0].post_process.add_field.fields.data_source
    assert source == 'ls_dev_lowbwer1000'
    assert cfg.val_reward.reward_function_by_data_source[source] == 'ls_keywords'
    assert cfg.data.val_data[0].jsonl_paths.endswith('/dev_lowbwer1000_step0_v1.jsonl')
    assert cfg.trainer.experiment_name == 'remax_ls_joint_pilot_dev1k'
    assert (cfg.trainer.total_training_steps, cfg.trainer.save_freq, cfg.trainer.test_freq) == (100, 10, 20)
    assert cfg.actor_rollout_ref.rollout.gt_rollout and cfg.actor_rollout_ref.rollout.n == 8

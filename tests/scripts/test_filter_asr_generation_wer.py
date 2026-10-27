import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

spec = importlib.util.spec_from_file_location('filter_wer', Path(__file__).parents[2] / 'scripts/filter_asr_generation_wer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(index, errors, refs):
    return dict(n_err=errors, n_ref=refs, wer=errors / refs,
                audio_duration_seconds=3.0, text='hello world', response='hello',
                audio_chunk=f'az://account/data/Chunk/VCTK/simu/features/chunk.audio:400:{index}')


def test_strict_boundaries_and_insertions():
    assert module.retained(row(0, 0, 100)) == dict(wer_gt0=False, wer_gt1pct=False)
    assert module.retained(row(0, 1, 100)) == dict(wer_gt0=True, wer_gt1pct=False)
    assert module.retained(row(0, 2, 100)) == dict(wer_gt0=True, wer_gt1pct=True)
    assert module.measurements(row(0, 3, 1))[-1][0] == 300


@pytest.mark.parametrize('bad', [dict(n_err=-1), dict(n_ref=0), dict(wer=float('nan')), dict(wer=.5), dict(audio_duration_seconds=None)])
def test_bad_measurements_are_rejected(bad):
    item = row(0, 1, 100)
    item.update(bad)
    with pytest.raises(ValueError):
        module.measurements(item)


def test_complete_outputs_and_no_publication_for_partial_run(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    rows = [row(0, 0, 100), row(1, 1, 100), row(2, 2, 100), row(3, 3, 1)]
    (source / 'part-000.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    args = SimpleNamespace(input_dir=str(source), output_dir=str(tmp_path / 'filtered'),
                           report_dir=str(tmp_path / 'report'), expected_rows=5)
    with pytest.raises(ValueError, match='Incomplete generation'):
        module.run(args)
    assert not (tmp_path / 'filtered').exists()
    args.expected_rows = 4
    module.run(args)
    summary = json.loads((tmp_path / 'report/summary.json').read_text())['groups']['all']
    assert summary['original']['count'] == 4
    assert summary['wer_gt0']['count'] == 3
    assert summary['wer_gt1pct']['count'] == 2
    assert summary['wer_gt1pct']['corpus_wer_percent'] == pytest.approx(500 / 101)
    saved = [json.loads(s) for s in (tmp_path / 'filtered/wer_gt1pct/part-000.jsonl').read_text().splitlines()]
    assert saved == rows[2:]

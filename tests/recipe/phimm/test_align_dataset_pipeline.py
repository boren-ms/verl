"""Regression coverage for alignment errors and bounded audio prefetch."""
from types import SimpleNamespace

from recipe.phimm import align_dataset as align


def test_failed_individual_alignment_is_not_silently_saved(monkeypatch):
    class FakeAligner:
        def align(self, audio, text, language):
            if isinstance(text, list) and len(text) > 1:
                raise ValueError('bad batch')
            if text == 'broken':
                raise RuntimeError('invalid audio')
            return [[SimpleNamespace(text='ok',start_time=0.1,end_time=0.3)]]
    monkeypatch.setattr(align, '_get_aligner', lambda *args: FakeAligner())
    monkeypatch.setattr('recipe.phimm.utils.audio.load_audios', lambda batch: ['a','b'])
    result=align._align_batch({'text':['ok','broken']},0,'English','fake',1)
    assert result['n_words']==[1,0]
    assert result['alignment_error'][0]==''
    assert 'invalid audio' in result['alignment_error'][1]


def test_parallel_prefetch_deduplicates_and_propagates_errors(monkeypatch):
    calls=[]
    def cache(src):
        calls.append(src)
        if src.endswith('bad.wav'): raise RuntimeError('storage unavailable')
        return src
    monkeypatch.setattr('verl.audio_cache.cache_audio_source',cache)
    monkeypatch.setattr(align, '_prefetch_pool', None)
    base='az://orngwus2cresco/data/a.wav'
    align._prefetch_batch({'audio_path':[base,base,'/local.wav']},2)
    assert calls==[base]
    import pytest
    with pytest.raises(RuntimeError,match='storage unavailable'):
        align._prefetch_batch({'audio_path':['az://orngwus2cresco/data/bad.wav']},2)
    align._prefetch_pool.shutdown()
    align._prefetch_pool=None

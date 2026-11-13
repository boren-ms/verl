import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / 'audio_subset.py'
spec = importlib.util.spec_from_file_location('audio_subset', SCRIPT)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def run(*args, check=True):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str,args)], text=True, capture_output=True, check=check)


def write(path, rows):
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def sample(uid, audio, text='one two three four five six'):
    return {'id':uid,'text':text,'audio_path':str(audio),'word_timestamps':[
        {'text':w,'start':.1+i*.2,'end':.2+i*.2} for i,w in enumerate(text.split())]}


def test_sharding_is_deterministic_disjoint_and_complete(tmp_path):
    rows=[sample(f'id-{i}',tmp_path/'audio.wav') for i in range(31)]
    source=tmp_path/'in.jsonl';write(source,rows)
    for name in ('a','b'): run('shard','--input',source,'--output-dir',tmp_path/name,'--count',3)
    grouped=[]
    for i in range(3):
        a=(tmp_path/'a'/f'shard-{i:04d}.jsonl').read_text();b=(tmp_path/'b'/f'shard-{i:04d}.jsonl').read_text(); assert a==b
        grouped.extend(x['id'] for x in map(json.loads,a.splitlines()))
    assert set(grouped)=={x['id'] for x in rows} and len(grouped)==len(rows)
    assert json.loads((tmp_path/'a'/'COMPLETE.json').read_text())['total']==31


def test_clip_resumes_after_missing_audio_and_rejects_bad_alignment(tmp_path):
    wav=tmp_path/'a.wav'; later=tmp_path/'later.wav'
    data=np.arange(32000,dtype=np.float32)/32000-.5; sf.write(wav,data,16000,subtype='PCM_16')
    bad=sample('bad',wav);bad['word_timestamps'][0]['text']='wrong'
    rows=[sample('good',wav),bad,sample('later',later)]
    source=tmp_path/'in.jsonl';write(source,rows);out=tmp_path/'out'
    result=run('crop','--input',source,'--output-dir',out,'--workers',2,check=False)
    assert result.returncode != 0 and not (out/'COMPLETE.json').exists()
    sf.write(later,data,16000,subtype='PCM_16')
    run('crop','--input',source,'--output-dir',out,'--workers',2,'--resume')
    good=read(out/'data.jsonl');rejects=read(out/'rejected.jsonl')
    assert {x['id'] for x in good}=={'good','later'} and [x['id'] for x in rejects]==['bad']
    for row in good:
        clip,sr=sf.read(out/'audio'/row['clip_filename']); original,_=sf.read(wav)
        assert sr==16000 and len(clip)==int(row['source_end']*sr)-int(row['source_start']*sr)
        np.testing.assert_array_equal(clip,original[int(row['source_start']*sr):int(row['source_end']*sr)])
        assert row['text']==' '.join(rows[0]['text'].split()[:len(row['text'].split())])
    assert run('crop','--input',source,'--output-dir',out,'--resume',check=False).returncode != 0


def test_zero_duration_and_short_transcripts():
    r=sample('short','none','one two'); assert mod.plan_prefix(r,2,5,42)[0]=='one two'
    r['word_timestamps'][1]['start']=r['word_timestamps'][1]['end']
    assert mod.plan_prefix(r,2,5,42)[3][-1]['start']==mod.plan_prefix(r,2,5,42)[3][-1]['end']
    r['word_timestamps'][0]['text']='different'
    with pytest.raises(ValueError,match='differs'): mod.plan_prefix(r)


def test_merge_rejects_duplicate_ids(tmp_path):
    for name in ('x','y'):
        p=tmp_path/name;p.mkdir();write(p/'data.jsonl',[{'id':'same'}]);write(p/'rejected.jsonl',[]);(p/'run.json').write_text('{"input_sha256":"x"}');(p/'COMPLETE.json').write_text('{"written_total":1,"rejected_total":0}')
    result=run('merge','--inputs',tmp_path/'x',tmp_path/'y','--output-dir',tmp_path/'merged',check=False)
    assert result.returncode and not (tmp_path/'merged'/'COMPLETE.json').exists()


def test_two_completed_shards_merge_and_review(tmp_path):
    wav=tmp_path/'a.wav';sf.write(wav,np.zeros(32000),16000)
    rows=[sample(f'id-{i}',wav) for i in range(8)]
    source=tmp_path/'in.jsonl';write(source,rows)
    run('shard','--input',source,'--output-dir',tmp_path/'shards','--count',2)
    dirs=[]
    for i in range(2):
        shard=tmp_path/'shards'/f'shard-{i:04d}.jsonl'; out=tmp_path/f'crop-{i}'; dirs.append(out)
        run('crop','--input',shard,'--output-dir',out,'--audio-base','az://example/audio','--workers',2)
    merged=tmp_path/'merged';run('merge','--inputs',*dirs,'--output-dir',merged)
    records=read(merged/'data.jsonl');assert len(records)==8 and len({r['id'] for r in records})==8
    assert all(r['audio_path'].startswith('az://example/audio/') for r in records)
    review=tmp_path/'review.jsonl';run('review','--input',merged/'data.jsonl','--output',review,'--count',4)
    assert len(read(review))==4


def test_zero_duration_exclusion(tmp_path):
    wav=tmp_path/'a.wav';sf.write(wav,np.zeros(32000),16000)
    row=sample('zero',wav,'one two');row['word_timestamps'][1]['start']=row['word_timestamps'][1]['end']
    source=tmp_path/'in.jsonl';write(source,[row]);out=tmp_path/'out'
    run('crop','--input',source,'--output-dir',out,'--zero-duration','exclude')
    assert read(out/'data.jsonl')==[] and read(out/'rejected.jsonl')[0]['reason']=='zero-duration selected word'
    assert json.loads((out/'COMPLETE.json').read_text())['rejected_total']==1


def test_select_middle_word_window():
    row=sample('middle','none')
    text,start,end,times=mod.plan_prefix(row,2,2,42,start_word=2)
    assert text=='three four' and start==pytest.approx(.5) and end==pytest.approx(.8)
    assert [w['text'] for w in times]==['three','four']
    with pytest.raises(ValueError,match='outside'):
        mod.plan_prefix(row,2,5,42,start_word=6)

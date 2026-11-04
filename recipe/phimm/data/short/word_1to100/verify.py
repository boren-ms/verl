#!/usr/bin/env python3
"""Verify rendered files against their immutable plans and source PCM samples."""
import argparse, collections, concurrent.futures, hashlib, json
from pathlib import Path
import numpy as np
import soundfile as sf
from verl.audio_cache import cache_audio_source


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plans',required=True);ap.add_argument('--rendered',required=True);ap.add_argument('--report',required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    assert a.workers>0
    plans=Path(a.plans);root=Path(a.rendered);report=Path(a.report)
    if report.exists(): raise FileExistsError(report)
    meta=json.loads((root/'COMPLETE.json').read_text());limit=meta['limit_per_subset'];jobs=[];out={};all_ids=set();version_sources={}
    for name,expected_hash in sorted(meta['plan_hashes'].items()):
        path=plans/name;assert digest(path)==expected_hash;subset=path.stem;directory=root/subset
        expected={}
        with path.open() as f:
            for i,line in enumerate(f):
                if limit is not None and i>=limit:break
                row=json.loads(line);assert row['id'] not in expected;expected[row['id']]=row
        actual={}
        with (directory/'data.jsonl').open() as f:
            for line in f:
                x=json.loads(line);uid=x['id'];assert uid in expected and uid not in actual and uid not in all_ids;actual[uid]=x;all_ids.add(uid)
        rejects={}
        with (directory/'rejected.jsonl').open() as f:
            for line in f:
                x=json.loads(line);uid=x['id'];assert uid in expected and uid not in rejects and uid not in actual and x['plan']==expected[uid];rejects[uid]=x
        assert set(expected)==set(actual)|set(rejects)
        count=meta['counts'][subset];assert count.get('planned',0)==len(expected) and count.get('written',0)==len(actual) and count.get('rejected',0)==len(rejects)
        files={f.name for f in (directory/'audio').iterdir()};assert files=={uid+'.wav' for uid in actual}
        used=set()
        for uid,x in actual.items():
            plan=expected[uid];assert x['text']==plan['text'] and x['word_count']==plan['target_words'] and x['seed']==plan['seed']
            assert x['audio_path']==meta['remote_root'].rstrip('/')+'/'+subset+'/audio/'+uid+'.wav'
            assert len(x['source_segments'])==len(plan['segments'])
            for s,p in zip(x['source_segments'],plan['segments']):
                assert all(s[k]==v for k,v in p.items())
                if subset!='train': assert s['id'] not in used
                used.add(s['id'])
            jobs.append((subset,uid,x,directory/'audio'/(uid+'.wav')))
        version_sources[subset]=used;out[subset]={'planned':len(expected),'written':len(actual),'rejected':len(rejects),'source_count':len(used),'word_counts':{},'duration_max':0.0,'duration_over_40_seconds':0}
    def verify(job):
        subset,uid,x,path=job;assert digest(path)==x['sha256'];wave,sr=sf.read(path,dtype='float32');assert sr==16000 and wave.ndim==1 and np.isfinite(wave).all() and len(wave)>0
        assert len(wave)==round(x['duration']*sr);offset=0
        for seg in x['source_segments']:
            src=Path(cache_audio_source(seg['audio_path']));info=sf.info(src);assert info.samplerate==sr and info.channels==1 and info.frames==seg['source_frames'] and seg['sample_rate']==sr
            start=round(float(seg['start'])*sr);end=round(float(seg['end'])*sr);assert 0<=start<end<=info.frames and start==seg['source_start_sample'] and end==seg['source_end_sample_exclusive']
            assert seg['output_start_sample']==offset and seg['output_end_sample_exclusive']==offset+end-start
            data,rate=sf.read(src,start=start,stop=end,dtype='float32');assert rate==sr and len(data)==end-start and np.isfinite(data).all()
            assert np.max(np.abs(wave[offset:offset+len(data)]-data))<=1/32768
            offset+=len(data)
        assert offset==len(wave)
        return subset,x['word_count'],len(wave)/sr
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        for i,(subset,words,duration) in enumerate(executor.map(verify,jobs),1):
            d=out[subset];d['word_counts'][words]=d['word_counts'].get(words,0)+1;d['duration_max']=max(d['duration_max'],duration);d['duration_over_40_seconds']+=duration>40
            if i%100==0:print(json.dumps({'verified':i,'total':len(jobs)}),flush=True)
    overlaps={}
    for split in ('test-clean','test-other'):
        names=sorted(n for n in version_sources if n.startswith(split+'_'))
        overlaps[split]={f'{x}__{y}':len(version_sources[x]&version_sources[y]) for i,x in enumerate(names) for y in names[i+1:]}
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x') as f:json.dump({'source_pcm_checked':True,'audio_files_checked':len(jobs),'subsets':out,'cross_version_overlap':overlaps,'renderer_complete_sha256':digest(root/'COMPLETE.json')},f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'verified_audio_files':len(jobs),'report':str(report)}),flush=True)
if __name__=='__main__':main()

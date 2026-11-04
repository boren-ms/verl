#!/usr/bin/env python3
"""Independently verify planned word windows and source reuse; does not validate audio."""
import argparse,collections,hashlib,json,re
from pathlib import Path
FILES={'train':['train-0','train-1'],'test-clean':['test-clean'],'test-other':['test-other-0','test-other-1']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--aligned-dir',required=True);p.add_argument('--plans',required=True);a=p.parse_args();aligned=Path(a.aligned_dir);plans=Path(a.plans);summary=json.loads((plans/'summary.json').read_text());sources={};ids={}
 for split,names in FILES.items():
  for name in names:
   path=aligned/(name+'.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['inputs'][split]['sha256'][path.name]
   for line in path.open():
    x=json.loads(line);uid=x['id'].removesuffix('.flac');assert uid not in sources;w=x['word_timestamps'];w=json.loads(w) if isinstance(w,str) else w;sources[uid]=(split,x,w)
 versions={};result={};total=0
 for name in summary['subsets']:
  split='train' if name=='train' else name.rsplit('_',2)[0];lo,hi=(1,100) if split=='train' else map(int,name.rsplit('_',2)[1:]);seen=set();rows=0;long=0;maxdur=0;word_counts=collections.Counter();rejects=0
  for line in (plans/(name+'.rejected.jsonl')).open():rejects+=1
  for line in (plans/(name+'.jsonl')).open():
   row=json.loads(line);rows+=1; assert row['id'] not in ids;ids[row['id']]=1;segs=row['segments'];assert segs; local=set(); texts=[];n=0;dur=0
   for s in segs:
    sid=s['id'];assert sid not in local;local.add(sid)
    if split!='train':assert sid not in seen
    source_split,original,words=sources[sid];assert source_split==split and original['audio_path']==s['audio_path'] and original['text']==s['source_text']
    selected=words[s['start_word_index']:s['end_word_index_exclusive']];assert selected and selected==s['word_timestamps'];assert s['text']==' '.join(w['text'] for w in selected) and s['start']==float(selected[0]['start']) and s['end']==float(selected[-1]['end'])
    assert s['source_split'] in (('train-clean-100','train-clean-360','train-other-500') if split=='train' else (split,));texts.append(s['text']);n+=len(selected);dur+=s['end']-s['start']
   seen.update(local);assert n==row['target_words'] and lo<=n<=hi and row['text']==' '.join(texts) and abs(dur-row['estimated_duration'])<1e-8
   word_counts[n]+=1;long+=dur>40;maxdur=max(maxdur,dur)
  assert rows+rejects==summary['subsets'][name]['requested'];versions[name]=seen;total+=rows;result[name]={'planned':rows,'rejected':rejects,'source_count':len(seen),'word_distribution':dict(word_counts),'estimated_duration_max':maxdur,'estimated_over_40_seconds':long}
 for split in ('test-clean','test-other'):
  for key,expected in summary[split+'_version_overlap'].items():
   x,y=key.split('__');assert len(versions[x]&versions[y])==expected
 print(json.dumps({'total_planned':total,'audio_rendered':False,'results':result},indent=2,sort_keys=True))
if __name__=='__main__':main()

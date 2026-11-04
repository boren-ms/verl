#!/usr/bin/env python3
"""Deterministic candidate plans; this does not create audio or final data."""
import argparse,bisect,json,math,random,re,collections,hashlib
from pathlib import Path
R=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261104);p.add_argument('--per-train-bin',type=int,default=500);p.add_argument('--per-test-target',type=int,default=200);a=p.parse_args(); rng=random.Random(a.seed)
def tok(s):return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*",s.lower())
def load(split,files):
 out=[];counts=collections.Counter(); ids=set()
 for name in files:
  for line in (R/'aligned'/f'{name}.jsonl').open():
   x=json.loads(line);uid=x['id'].removesuffix('.flac');src=x['audio_path'];ss=src.split('/LibriSpeech/')[1].split('/')[0]
   if uid in ids: raise ValueError('duplicate id '+uid)
   ids.add(uid)
   if ss not in (('train-clean-100','train-clean-360','train-other-500') if split=='train' else (split,)):raise ValueError('split mismatch')
   try:
    words=x['word_timestamps'];words=json.loads(words) if isinstance(words,str) else words
    if not words or tok(' '.join(t['text'] for t in words))!=tok(x['text']): raise ValueError('text')
    prev=0
    for w in words:
     start,end=float(w['start']),float(w['end'])
     if not math.isfinite(start) or not math.isfinite(end) or start<prev or end<=start:raise ValueError('timestamp')
     prev=end
    x['word_timestamps']=words;out.append(x);counts['accepted']+=1
   except Exception as e:counts['excluded_'+str(e)]+=1
 return out,ids,counts
sources={}; sets={};aud={}
for split,files in {'train':['train-0','train-1'],'test-clean':['test-clean'],'test-other':['test-other-0','test-other-1']}.items():sources[split],sets[split],aud[split]=load(split,files)
assert not (sets['train'] & sets['test-clean'] or sets['train'] & sets['test-other'] or sets['test-clean'] & sets['test-other'])
def window(x,target):
 w=x['word_timestamps']; i=rng.randrange(len(w)); t=float(w[i]['start']); ends=[float(z['end']) for z in w];j=bisect.bisect_left(ends,t+target,lo=i);js=[z for z in (j-1,j) if i<=z<len(w)]
 if not js:return None
 j=min(js,key=lambda k:abs((ends[k]-t)-target)); dur=ends[j]-t
 return {'id':x['id'],'source_split':x['audio_path'].split('/LibriSpeech/')[1].split('/')[0],'audio_path':x['audio_path'],'source_text':x['text'],'start_word_index':i,'end_word_index_exclusive':j+1,'start':t,'end':ends[j],'text':' '.join(z['text'] for z in w[i:j+1]),'duration':dur}
def candidate(pool,target):
 best=None
 for _ in range(32):
  x=rng.choice(pool);q=window(x,target)
  if not q:continue
  if best is None or abs(q['duration']-target)<abs(best['duration']-target):best=q
  if abs(best['duration']-target)<0.04:break
 return best
def attempt(pool,target):
 q=candidate(pool,target);best=([q],abs(q['duration']-target)) if q and q['duration']<30 else None
 if best and best[1]<=0.12:return best
 for _ in range(12):
  mid=rng.uniform(.25,.75)*target;x=candidate(pool,mid);y=candidate(pool,target-x['duration']) if x else None
  if not x or not y or x['id']==y['id']:continue
  d=x['duration']+y['duration'];err=abs(d-target)
  if d<30 and (best is None or err<best[1]):best=([x,y],err)
  if best and best[1]<.04:break
 return best
out=R/'plans';out.mkdir(exist_ok=True);summary={}
for split,pool in sources.items():
 targets=[(i+.5,a.per_train_bin) for i in range(1,30)] if split=='train' else [(t,a.per_test_target) for t in (5,10,20)]
 for target,n in targets:
  name=f'train_{int(target)}' if split=='train' else f'{split}_{target}s';counts=collections.Counter();errors=[]
  with (out/(name+'.jsonl')).open('x') as f:
   for idx in range(n):
    result=attempt(pool,target)
    if not result:counts['no_plan']+=1;continue
    segs,err=result;dur=sum(x['duration'] for x in segs)
    if err>.5:counts['outside_half_second']+=1;continue
    counts['planned']+=1;counts[f'{len(segs)}_segments']+=1;errors.append(err)
    f.write(json.dumps({'id':f'{name}_{idx:05d}','target':target,'estimated_duration':dur,'absolute_error':err,'segments':segs,'text':' '.join(z['text'] for z in segs)},ensure_ascii=False)+'\n')
  summary[name]={'requested':n,**counts,'max_error':max(errors,default=None),'mean_absolute_error':sum(errors)/len(errors) if errors else None}
report={'seed':a.seed,'policy':'full alignment with strictly positive nonoverlapping words, sampled with replacement; target tolerance 0.5 sec; at most 2 distinct source utterances; candidate plans only, audio not yet verified','inputs':{k:dict(v) for k,v in aud.items()},'subsets':summary}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

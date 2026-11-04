#!/usr/bin/env python3
"""Render candidate manifests into physical WAVs, independently of training."""
import argparse,collections,concurrent.futures,hashlib,json,math,os,re,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from verl.audio_cache import cache_audio_source
p=argparse.ArgumentParser();p.add_argument('--plans',required=True);p.add_argument('--output',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit-per-file',type=int);p.add_argument('--only-id');a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
files=sorted(Path(a.plans).glob('*.jsonl')); assert len(files)==35,len(files)
manifest_files={}; counts={}; reject_files={}
def get_subset(path):
 return 'train' if path.stem.startswith('train_') else path.stem
jobs=[]
for src in files:
 subset=get_subset(src);dest=root/subset
 if subset not in manifest_files:
  (dest/'audio').mkdir(parents=True);manifest_files[subset]=(dest/'data.jsonl').open('x');reject_files[subset]=(dest/'rejected.jsonl').open('x');counts[subset]={'input':0,'written':0,'rejected':0}
 for i,line in enumerate(src.open()):
  if a.limit_per_file is not None and i>=a.limit_per_file:break
  row=json.loads(line)
  if a.only_id and row['id']!=a.only_id:continue
  jobs.append((subset,row))
assert len({r['id'] for _,r in jobs})==len(jobs)
if a.only_id: assert len(jobs)==1, len(jobs)
class SampleReject(ValueError):
 pass
def render(job):
 subset,row=job;ident=row['id'];assert re.fullmatch(r'[A-Za-z0-9_-]+',ident)
 segs=row['segments'];assert 1<=len(segs)<=2 and len({s['id'] for s in segs})==len(segs)
 audio=[];rate=None;details=[]
 for seg in segs:
  ss=seg['source_split'];assert ss in (('train-clean-100','train-clean-360','train-other-500') if subset=='train' else (subset.rsplit('_',1)[0],))
  path=cache_audio_source(seg['audio_path']);info=sf.info(path);sr=info.samplerate
  if rate is None:rate=sr
  if sr!=rate or info.channels!=1: raise SampleReject(f'unsupported source audio rate/channels: {path} {sr} {info.channels}')
  start=round(float(seg['start'])*sr);end=round(float(seg['end'])*sr)
  if not 0<=start<end<=info.frames:raise SampleReject(f'out of range audio bounds: {ident} {path} start={start} end={end} frames={info.frames} sr={sr}')
  d,fs=sf.read(path,start=start,stop=end,dtype='float32');assert fs==sr and len(d)==end-start and np.isfinite(d).all()
  audio.append(d);details.append({**seg,'start_sample':start,'end_sample':end,'sample_rate':sr,'source_frames':info.frames})
 wave=np.concatenate(audio);duration=len(wave)/rate
 if not 0<duration<30 or abs(duration-row['target'])>.5001:raise SampleReject(f'bad duration {ident}: {duration}')
 dst=root/subset/'audio'/f'{ident}.wav';sf.write(dst,wave,rate,format='WAV',subtype='PCM_16')
 actual=sf.info(dst)
 if actual.frames!=len(wave) or actual.samplerate!=rate:raise ValueError('output integrity')
 return subset,{'id':ident,'text':row['text'],'audio_path':f"{a.remote_root.rstrip('/')}/{subset}/audio/{ident}.wav",'duration':duration,'target_duration':row['target'],'absolute_error':abs(duration-row['target']),'source_segments':details,'seed':20261104,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
def render_with_rejection(job):
 try:
  subset,row=render(job)
  return subset,row,None
 except SampleReject as e:
  return job[0],None,{'id':job[1]['id'],'reason':str(e),'plan':job[1]}
def bounded_map(pool, items):
 it=iter(items);pending=collections.deque()
 for _ in range(a.workers):
  try:pending.append(pool.submit(render_with_rejection,next(it)))
  except StopIteration:break
 while pending:
  result=pending.popleft().result()
  try:pending.append(pool.submit(render_with_rejection,next(it)))
  except StopIteration:pass
  yield result
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,(subset,row,rejection) in enumerate(bounded_map(pool,jobs),1):
   counts[subset]['input']+=1
   if rejection:
    counts[subset]['rejected']+=1;reject_files[subset].write(json.dumps(rejection,ensure_ascii=False)+'\n')
   else:
    counts[subset]['written']+=1;manifest_files[subset].write(json.dumps(row,ensure_ascii=False)+'\n')
   if i%100==0:print(json.dumps({'done':i,'total':len(jobs),'subsets':counts}),flush=True)
except Exception as e:
 print(json.dumps({'error':repr(e),'counts':counts}),file=sys.stderr,flush=True);raise
finally:
 for f in (*manifest_files.values(),*reject_files.values()):f.close()
(root/'COMPLETE.json').write_text(json.dumps({'counts':counts,'total':len(jobs),'input_plan_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}},indent=2)+'\n')
print(json.dumps({'complete':counts}),flush=True)

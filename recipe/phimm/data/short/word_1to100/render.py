#!/usr/bin/env python3
"""Render verified word-window plans as PCM WAVs. Existing destinations are refused."""
import argparse,collections,concurrent.futures,hashlib,json,math,re
from pathlib import Path
import numpy as np
import soundfile as sf
from verl.audio_cache import cache_audio_source

class InvalidSourceBounds(Exception): pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--plans',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--limit-per-subset',type=int);a=p.parse_args(); assert a.workers>0
 plans=Path(a.plans);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);summary=json.loads((plans/'summary.json').read_text());jobs=[];handles={};counts={}
 for name in sorted(summary['subsets']):
  directory=out/name;(directory/'audio').mkdir(parents=True);handles[name]=((directory/'data.jsonl').open('x'),(directory/'rejected.jsonl').open('x'));counts[name]=collections.Counter()
  for i,line in enumerate((plans/(name+'.jsonl')).open()):
   if a.limit_per_subset is not None and i>=a.limit_per_subset:break
   jobs.append((name,json.loads(line)))
 def render(job):
  name,row=job;ident=row['id'];assert re.fullmatch(r'[A-Za-z0-9_-]+',ident);pieces=[];provenance=[];offset=0
  for seg in row['segments']:
   path=cache_audio_source(seg['audio_path']);info=sf.info(path);sr=info.samplerate
   if sr!=16000 or info.channels!=1: raise InvalidSourceBounds(f'unsupported source format: {sr} Hz, {info.channels} channels')
   start=round(float(seg['start'])*sr);end=round(float(seg['end'])*sr)
   if not 0<=start<end<=info.frames: raise InvalidSourceBounds(f'out of range {seg["id"]}: {start}:{end} of {info.frames} samples')
   sound,rate=sf.read(path,start=start,stop=end,dtype='float32');assert rate==sr and len(sound)==end-start and np.isfinite(sound).all()
   pieces.append(sound);provenance.append({**seg,'source_start_sample':start,'source_end_sample_exclusive':end,'source_frames':info.frames,'sample_rate':sr,'output_start_sample':offset,'output_end_sample_exclusive':offset+len(sound)});offset+=len(sound)
  wave=np.concatenate(pieces);dst=out/name/'audio'/f'{ident}.wav';sf.write(dst,wave,16000,format='WAV',subtype='PCM_16');info=sf.info(dst);assert info.frames==len(wave) and info.samplerate==16000 and info.channels==1
  return {'id':ident,'text':row['text'],'audio_path':f'{a.remote_root.rstrip("/")}/{name}/audio/{ident}.wav','word_count':row['target_words'],'duration':len(wave)/16000,'sample_rate':16000,'source_segments':provenance,'seed':row['seed'],'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
 def run(job):
  try:return job[0],render(job),None
  except InvalidSourceBounds as e:return job[0],None,{'id':job[1]['id'],'reason':str(e),'plan':job[1]}
 def bounded(executor):
  it=iter(jobs);queue=collections.deque()
  for _ in range(a.workers):
   try:queue.append(executor.submit(run,next(it)))
   except StopIteration:break
  while queue:
   item=queue.popleft().result()
   try:queue.append(executor.submit(run,next(it)))
   except StopIteration:pass
   yield item
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
   for i,(name,row,rejection) in enumerate(bounded(executor),1):
    counts[name]['planned']+=1
    if rejection:counts[name]['rejected']+=1;handles[name][1].write(json.dumps(rejection,ensure_ascii=False)+'\n')
    else:counts[name]['written']+=1;handles[name][0].write(json.dumps(row,ensure_ascii=False)+'\n')
    if i%100==0:print(json.dumps({'done':i,'total':len(jobs)}),flush=True)
 finally:
  for pair in handles.values():
   for f in pair:f.close()
 report={'counts':{k:dict(v) for k,v in counts.items()},'plan_hashes':{f'{k}.jsonl':hashlib.sha256((plans/(k+'.jsonl')).read_bytes()).hexdigest() for k in summary['subsets']},'remote_root':a.remote_root,'limit_per_subset':a.limit_per_subset}
 (out/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'complete':len(jobs)}),flush=True)
if __name__=='__main__':main()

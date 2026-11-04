#!/usr/bin/env python3
"""Measure signal jumps at segment joins and select deterministic listening samples."""
import collections,concurrent.futures,hashlib,json,random
from pathlib import Path
import numpy as np
import soundfile as sf
r=Path('/dev/shm/librispeech_word_1to100_20261104');root=r/'merged_a1';review=r/'human_review_a1.jsonl';report=r/'boundary_audit_a1.json';assert not review.exists() and not report.exists();complete=json.loads((root/'COMPLETE.json').read_text());jobs=[];selected=[];rng=random.Random(20261104)
for name in sorted(complete['counts']):
 rows=[json.loads(line) for line in (root/name/'data.jsonl').open()];assert rows
 longest=max(rows,key=lambda x:x['duration']);normal=rng.choice(rows)
 if normal['id']==longest['id'] and len(rows)>1:normal=next(x for x in rows if x['id']!=longest['id'])
 for kind,x in [('random',normal),('longest',longest)]:selected.append({'selection':kind,'dataset':name,**x})
 for row in rows:jobs.append((name,row))
def audit(job):
 name,row=job;wav,sr=sf.read(root/name/'audio'/(row['id']+'.wav'),dtype='float32');assert sr==16000;segs=row['source_segments'];jumps=[];edge_rms=[]
 for s in segs:
  a=s['output_start_sample'];b=s['output_end_sample_exclusive'];n=min(160,(b-a)//2)
  if n:edge_rms += [float(np.sqrt(np.mean(wav[a:a+n]**2))),float(np.sqrt(np.mean(wav[b-n:b]**2)))]
 for s in segs[1:]:
  i=s['output_start_sample'];jumps.append(abs(float(wav[i])-float(wav[i-1])))
 return name,len(segs),sum(x>0.1 for x in jumps),max(jumps,default=0),sum(x>0.05 for x in edge_rms),len(edge_rms)
counts=collections.defaultdict(collections.Counter);maxjump=collections.defaultdict(float)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for i,(name,segments,jumps,mx,loud,edges) in enumerate(ex.map(audit,jobs),1):
  c=counts[name];c['samples']+=1;c['segments']+=segments;c['joins']+=max(0,segments-1);c['joins_jump_over_0_1']+=jumps;c['edges_rms_over_0_05']+=loud;c['edges_measured']+=edges;maxjump[name]=max(mx,maxjump[name])
  if i%1000==0:print(json.dumps({'checked':i,'total':len(jobs)}),flush=True)
with review.open('x') as out:
 for x in selected:out.write(json.dumps(x,ensure_ascii=False)+'\n')
data={'seed':20261104,'methods':{'join_jump':'absolute sample difference across segment join, flag >0.1 full scale','edge_rms':'RMS over first/last 10 ms of each segment, flag >0.05 full scale','limits':'Acoustic flags are triage signals only; neither speech detection nor proof of phonetic completeness.'},'subsets':{k:{**dict(v),'max_join_jump':maxjump[k]} for k,v in sorted(counts.items())},'review_rows':len(selected),'review_manifest_sha256':hashlib.sha256(review.read_bytes()).hexdigest()}
report.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');print(json.dumps({'checked':len(jobs),'review_rows':len(selected),'report':str(report)}),flush=True)

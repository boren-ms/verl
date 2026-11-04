#!/usr/bin/env python3
"""Read and hash all unique final audio referenced by the listening manifests."""
import concurrent.futures,hashlib,json,pathlib,subprocess,tempfile
r=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');paths=[r/'human_review_a1.jsonl',r/'boundary_flagged_review_a1.jsonl'];samples={}
for p in paths:
 for line in p.open():
  x=json.loads(line);uid=x['id'];assert uid not in samples or samples[uid]['sha256']==x['sha256'];samples[uid]=x
out=r/'review_readback_a1.json';assert not out.exists()
with tempfile.TemporaryDirectory(dir='/dev/shm',prefix='word100_review_') as t:
 def check(item):
  uid,row=item;dst=pathlib.Path(t)/(uid+'.wav');subprocess.run(['bbb','cp',row['audio_path'],str(dst)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);sha=hashlib.sha256(dst.read_bytes()).hexdigest();assert sha==row['sha256'],uid;return {'id':uid,'sha256':sha,'audio_path':row['audio_path']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check,samples.items()))
out.write_text(json.dumps({'unique_audio_readbacks':len(checked),'manifest_rows':sum(sum(1 for _ in p.open()) for p in paths),'items':checked},indent=2)+'\n');print(json.dumps({'unique_audio_readbacks':len(checked),'output':str(out)}),flush=True)

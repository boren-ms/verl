#!/usr/bin/env python3
"""Read back every manifest, inventory every WAV, and hash one WAV per nonempty split."""
import argparse,concurrent.futures,hashlib,json,subprocess,tempfile
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--local-root',required=True);p.add_argument('--remote-root',required=True);p.add_argument('--report',required=True);a=p.parse_args();local=Path(a.local_root);remote=a.remote_root.rstrip('/');report=Path(a.report);assert not report.exists();complete=json.loads((local/'COMPLETE.json').read_text());tasks=[];results={}
 with tempfile.TemporaryDirectory(prefix='word100_remote_audit_',dir='/dev/shm') as tmp:
  tmp=Path(tmp)
  for subset in sorted(complete['counts']):
   folder=local/subset;rows=[json.loads(line) for line in (folder/'data.jsonl').open()];expected={x['id']+'.wav' for x in rows};assert len(expected)==len(rows)
   if rows:
    listing=subprocess.run(['bbb','ls',f'{remote}/{subset}/audio/'],capture_output=True,text=True,check=True).stdout.splitlines();actual={x.rsplit('/',1)[-1] for x in listing if x.startswith('az://') and x.endswith('.wav')};assert actual==expected,(subset,len(expected-actual),len(actual-expected))
    sample=rows[0];tasks.append((f'{remote}/{subset}/audio/{sample["id"]}.wav',folder/'audio'/(sample['id']+'.wav'),sample['sha256']))
   for name in ('data.jsonl','rejected.jsonl'):tasks.append((f'{remote}/{subset}/{name}',folder/name,digest(folder/name)))
   results[subset]={'records':len(rows),'audio_inventory_exact':True,'audio_sample_readback_sha256':rows[0]['sha256'] if rows else None}
  tasks.append((f'{remote}/COMPLETE.json',local/'COMPLETE.json',digest(local/'COMPLETE.json')))
  def check(task):
   src,original,expected=task;dst=tmp/(hashlib.sha256(src.encode()).hexdigest()+'.blob');subprocess.run(['bbb','cp',src,str(dst)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True);actual=digest(dst);assert actual==expected,(src,expected,actual);return src
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,tasks))
 with report.open('x') as out:json.dump({'stage_root':remote,'subsets':results,'readback_objects':len(tasks),'all_audio_inventory_exact':True},out,indent=2,sort_keys=True);out.write('\n')
 print(json.dumps({'inventory_verified':sum(x['records'] for x in results.values()),'readbacks':len(tasks),'report':str(report)}))
if __name__=='__main__':main()

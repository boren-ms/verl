#!/usr/bin/env python3
"""Read back published manifests, inventory all audio, and load each dataset."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import yaml
from recipe.phimm.data.dataset import create_audio_dataset
from recipe.phimm.utils.audio import load_raw_audio
p=argparse.ArgumentParser();p.add_argument('--remote-root',required=True);p.add_argument('--local-root',required=True);p.add_argument('--repo-root',required=True);a=p.parse_args()
root=Path(a.local_root);(root/'manifests').mkdir(parents=True,exist_ok=True);(root/'evidence').mkdir(parents=True,exist_ok=True);(root/'readback').mkdir(parents=True,exist_ok=True)
def cp(src,dst):subprocess.run(['bbb','cp',src,str(dst)],check=True,stdout=subprocess.DEVNULL)
remote=a.remote_root.rstrip('/');verification=root/'evidence'/'verification.json';cp(remote+'/verification.json',verification);v=json.loads(verification.read_text()); report={}
repo=Path(a.repo_root);train=yaml.safe_load((repo/'recipe/phimm/config/data/train_data/concat/librispeech_1to30.yaml').read_text());vals=yaml.safe_load((repo/'recipe/phimm/config/data/val_data/concat/librispeech_1to30.yaml').read_text());configs={'train':train};configs.update({c['jsonl_paths'].split('/')[-2]:c for c in vals})
for subset,cfg in configs.items():
 manifest=root/'manifests'/f'{subset}.jsonl';cp(f'{remote}/{subset}/data.jsonl',manifest);assert hashlib.sha256(manifest.read_bytes()).hexdigest()==v['subsets'][subset]['manifest_sha256']
 rows=[json.loads(line) for line in manifest.open()]; expected={x['audio_path'] for x in rows};assert len(expected)==len(rows)==v['subsets'][subset]['count']
 listing=subprocess.run(['bbb','ls',f'{remote}/{subset}/audio/'],check=True,capture_output=True,text=True).stdout
 found={x.strip() for x in listing.splitlines() if x.strip().startswith('az://') and x.strip().endswith('.wav')};assert found==expected,(subset,len(expected-found),len(found-expected))
 ds=create_audio_dataset(**cfg);assert len(ds)==len(rows);x=ds[0];audio,sr=load_raw_audio(x);assert sr==16000 and len(audio)>0 and x['reward_model']['ground_truth']
 report[subset]={'records':len(rows),'inventory_exact':True,'manifest_sha256':v['subsets'][subset]['manifest_sha256'],'loader_sample_id':x['extra_info']['id'],'loader_audio_frames':len(audio)};print(json.dumps({subset:report[subset]}),flush=True)
review=[json.loads(line) for line in (root/'review'/'all.jsonl').open()];assert len(review)==14
byid={}
for subset in configs:
 for line in (root/'manifests'/f'{subset}.jsonl').open():
  x=json.loads(line);byid[x['id']]=x
for x in review:
 final=byid[x['id']];assert final['sha256']==x['sha256']
 dest=root/'readback'/f"{x['id']}.wav";cp(final['audio_path'],dest);assert hashlib.sha256(dest.read_bytes()).hexdigest()==x['sha256']
(root/'evidence'/'publication_audit.json').write_text(json.dumps({'subsets':report,'final_review_audio_hashes_verified':14},indent=2)+'\n');print('14 final audio readbacks verified',flush=True)

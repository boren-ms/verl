#!/usr/bin/env python3
"""Publish and independently read back provenance into a new dataset prefix."""
import collections,hashlib,json,pathlib,subprocess,tempfile
r=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');dst='az://orngwus2cresco/data/boren/data/LibriSpeech/word_1to100_20261104/provenance';stage='az://orngwus2cresco/data/boren/data/verl/staging/word_1to100_20261104_a1';old='az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1/aligned';s=json.loads((r/'plans/summary.json').read_text());a=json.loads((r/'merged_a1.audit.json').read_text());p=r/'provenance_manifest_a1.json'
assert not p.exists() and a['audio_files_checked']==11982
hashes={k:v for split in s['inputs'].values() for k,v in split['sha256'].items()}
metadata={'seed':s['seed'],'selection_policy':s['policy'],'input_audit':s['inputs'],'alignment':{'model':'Qwen/Qwen3-ForcedAligner-0.6B','revision':'c7cbfc2048c462b0d63a45797104fc9db3ad62b7','weights_sha256':'47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7','source_uri':old,'warning':'Estimated timestamps, not proof of phonetic completeness or VAD.'},'parts':{'local_partial_original_rows':1568,'remote_remaining_rows':10414,'training_rejected':18,'reject_reason':'estimated word boundary outside source audio','local_archive_sha256':'71985836a5a3957c730436eb311affdb51cdac103645c71c59e0ba4196160bcc'},'actual_subsets':a['subsets'],'cross_version_overlap':a['cross_version_overlap'],'remote_final_audio_hash_scope':'all local wavs compared to source PCM and hashed; remote full inventory and one per subset read back'}
p.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
def sha(x):
 h=hashlib.sha256()
 with x.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
files=[(str(r/'merged_a1.audit.json'),'merged_a1.audit.json',sha(r/'merged_a1.audit.json')),(str(r/'partial_verify_a1.audit.json'),'partial_verify_a1.audit.json',sha(r/'partial_verify_a1.audit.json')),(str(r/'remaining_render_a1.audit.json'),'remaining_render_a1.audit.json',sha(r/'remaining_render_a1.audit.json')),(str(r/'final_upload_audit_a1.json'),'final_upload_audit_a1.json',sha(r/'final_upload_audit_a1.json')),(str(p),'manifest.json',sha(p)),(stage+'/full_original_plans_a1.tar.gz','full_original_plans_a1.tar.gz','cc7c080394fd3cc39ef4a88fcf5123cf78f12cd1ff7e3803d8592f602af46851')]
for name,h in hashes.items():files.append((f'{old}/{name}',f'aligned/{name}',h))
with tempfile.TemporaryDirectory(prefix='word100_provenance_',dir='/dev/shm') as tmp:
 for i,(src,name,expected) in enumerate(files):
  target=f'{dst}/{name}';subprocess.run(['bbb','cp',src,target],check=True,stdout=subprocess.DEVNULL);local=pathlib.Path(tmp)/str(i);subprocess.run(['bbb','cp',target,str(local)],check=True,stdout=subprocess.DEVNULL);assert sha(local)==expected,(name,sha(local),expected);print(json.dumps({'done':i+1,'total':len(files),'file':name}),flush=True)
print(json.dumps({'phase':'provenance_published','files':len(files),'uri':dst}),flush=True)

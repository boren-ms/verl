#!/usr/bin/env python3
"""Merge two verified disjoint parts against original plans without copying WAV bytes."""
import collections,hashlib,json,os,pathlib,subprocess,sys
r=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');plans=r/'plans';parts=[r/'local_partial_a1',r/'remaining_render_a1'];audits=[r/'partial_verify_a1.audit.json',r/'remaining_render_a1.audit.json'];want=[1568,10414]
for audit,count in zip(audits,want):
 d=json.loads(audit.read_text());assert d['audio_files_checked']==count and d['source_pcm_checked']
stage=json.loads((r/'staging_audit_a1.json').read_text());assert stage['all_audio_inventory_exact'] and sum(x['records'] for x in stage['subsets'].values())==10414
summary=json.loads((plans/'summary.json').read_text());output=r/'merged_a1';output.mkdir(exist_ok=False);counts={};hashes={};total=0
for name in sorted(summary['subsets']):
 path=plans/(name+'.jsonl');plan_ids={json.loads(x)['id'] for x in path.open()};hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();dest=output/name;(dest/'audio').mkdir(parents=True);seen=set();count=collections.Counter()
 with (dest/'data.jsonl').open('x') as data,(dest/'rejected.jsonl').open('x') as reject:
  for part in parts:
   folder=part/name
   if not folder.exists():continue
   for line in (folder/'data.jsonl').open():
    row=json.loads(line);uid=row['id'];assert uid in plan_ids and uid not in seen;seen.add(uid);source=folder/'audio'/(uid+'.wav');assert source.is_file();os.link(source,dest/'audio'/(uid+'.wav'));data.write(line);count['written']+=1
   for line in (folder/'rejected.jsonl').open():
    row=json.loads(line);uid=row['id'];assert uid in plan_ids and uid not in seen;seen.add(uid);reject.write(line);count['rejected']+=1
 assert seen==plan_ids,(name,len(plan_ids-seen));count['planned']=len(seen);counts[name]=dict(count);total+=len(seen)
assert total==12000 and counts['train']['written']==9982 and counts['train']['rejected']==18
complete={'counts':counts,'plan_hashes':hashes,'remote_root':'az://orngwus2cresco/data/boren/data/LibriSpeech/word_1to100_20261104','limit_per_subset':None,'provenance':{'local_partial_archive_sha256':'71985836a5a3957c730436eb311affdb51cdac103645c71c59e0ba4196160bcc','partial_verify_sha256':hashlib.sha256(audits[0].read_bytes()).hexdigest(),'remaining_verify_sha256':hashlib.sha256(audits[1].read_bytes()).hexdigest(),'remaining_stage_audit_sha256':hashlib.sha256((r/'staging_audit_a1.json').read_bytes()).hexdigest()}}
(output/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps({'phase':'merged','planned':total,'audio':sum(x.get('written',0) for x in counts.values()),'rejects':sum(x.get('rejected',0) for x in counts.values())}),flush=True)
env=os.environ.copy();env.update(PYTHONPATH=f'{r}/python_deps:/root/code/verl',VERL_AUDIO_CACHE_ROOT=f'{r}/source_cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VERL_BLOB_READ_RETRY_LIMIT='2')
p=subprocess.run([sys.executable,'-m','recipe.phimm.data.short.word_1to100.verify','--plans',str(plans),'--rendered',str(output),'--report',str(r/'merged_a1.audit.json'),'--workers','8'],env=env)
if p.returncode:sys.exit(p.returncode)
print(json.dumps({'phase':'verified','audit':str(r/'merged_a1.audit.json')}),flush=True)

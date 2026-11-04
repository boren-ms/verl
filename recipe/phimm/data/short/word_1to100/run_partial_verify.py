#!/usr/bin/env python3
"""Prefetch and verify the separately migrated, locally rendered partial dataset."""
import hashlib,json,os,pathlib,subprocess,sys
r=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');plans=r/'partial_verify_input_a1/partial_plans';partial=r/'local_partial_a1';assert hashlib.sha256((r/'staging/local_partial_a2_c1.tar').read_bytes()).hexdigest()=='71985836a5a3957c730436eb311affdb51cdac103645c71c59e0ba4196160bcc';sources=r/'partial_sources_a1.jsonl';seen=set()
with sources.open('x') as out:
 for path in sorted(plans.glob('*.jsonl')):
  for line in path.open():
   for seg in json.loads(line)['segments']:
    x=seg['audio_path']
    if x not in seen:out.write(json.dumps({'audio_path':x})+'\n');seen.add(x)
env=os.environ.copy();env.update(PYTHONPATH=f'{r}/python_deps:/root/code/verl',VERL_AUDIO_CACHE_ROOT=f'{r}/source_cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VERL_BLOB_READ_RETRY_LIMIT='2')
print(json.dumps({'phase':'prefetch','unique_sources':len(seen)}),flush=True)
a=subprocess.run([sys.executable,'-m','recipe.phimm.data.short.word_1to100.prefetch_async','--input',str(sources),'--results',str(r/'partial_prefetch_a1.results.jsonl'),'--report',str(r/'partial_prefetch_a1.report.json'),'--workers','8'],env=env)
if a.returncode:print(json.dumps({'phase':'prefetch_failed','exit':a.returncode}),flush=True);sys.exit(a.returncode)
print(json.dumps({'phase':'verify'}),flush=True)
b=subprocess.run([sys.executable,'-m','recipe.phimm.data.short.word_1to100.verify','--plans',str(plans),'--rendered',str(partial),'--report',str(r/'partial_verify_a1.audit.json'),'--workers','8'],env=env)
if b.returncode:sys.exit(b.returncode)
print(json.dumps({'phase':'verified','audit':str(r/'partial_verify_a1.audit.json')}),flush=True)

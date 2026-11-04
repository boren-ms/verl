#!/usr/bin/env python3
"""Run the uniquely identified remote remainder and independently verify it."""
import collections,hashlib,json,os,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');archive=root/'staging/remaining_plans_a1.tar.gz';expected='96f5581723988b5ad4368998d12196a8e2cfce7053d1e95ec88c611453f2382c'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
plans=root/'remaining_plans_a1';assert not plans.exists()
with tarfile.open(archive,'r:gz') as tar:
 for member in tar.getmembers():
  if not (member.name in ('remaining_plans_a1','staging_meta') or member.name.startswith('remaining_plans_a1/') or member.name.startswith('staging_meta/')) or member.issym() or member.islnk() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts:raise ValueError('unsafe archive member')
 tar.extractall(root,filter='data')
manifest=root/'remaining_sources_a1.jsonl';seen=set()
with manifest.open('x') as out:
 for path in sorted(plans.glob('*.jsonl'),reverse=True):
  with path.open() as f:
   for line in f:
    for seg in json.loads(line)['segments']:
     source=seg['audio_path']
     if source not in seen:out.write(json.dumps({'audio_path':source})+'\n');seen.add(source)
print(json.dumps({'phase':'prepared','unique_sources':len(seen)}),flush=True)
env=os.environ.copy();env.update(PYTHONPATH=f'{root}/python_deps:/root/code/verl',VERL_AUDIO_CACHE_ROOT=f'{root}/source_cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VERL_BLOB_READ_RETRY_LIMIT='2')
prefetch=subprocess.Popen([sys.executable,'-m','recipe.phimm.data.short.word_1to100.prefetch_async','--input',str(manifest),'--results',str(root/'remaining_prefetch_a1.results.jsonl'),'--report',str(root/'remaining_prefetch_a1.report.json'),'--workers','8'],env=env)
print(json.dumps({'phase':'rendering','prefetch_pid':prefetch.pid}),flush=True)
try:
 render=subprocess.run([sys.executable,'-m','recipe.phimm.data.short.word_1to100.render','--plans',str(plans),'--output-dir',str(root/'remaining_render_a1'),'--remote-root','az://orngwus2cresco/data/boren/data/LibriSpeech/word_1to100_20261104','--workers','8'],env=env)
 if render.returncode:
  print(json.dumps({'phase':'render_failed','exit':render.returncode}),flush=True);prefetch.terminate();prefetch.wait();sys.exit(render.returncode)
 print(json.dumps({'phase':'render_complete','prefetch_return':prefetch.poll()}),flush=True)
 # The renderer has fetched every required source. Stop a still-running prefetch
 # belonging to this job; its independent result is not needed for verification.
 if prefetch.poll() is None:prefetch.terminate();prefetch.wait();print(json.dumps({'phase':'prefetch_stopped_after_render','exit':prefetch.returncode}),flush=True)
 elif prefetch.returncode:print(json.dumps({'phase':'prefetch_failed','exit':prefetch.returncode}),flush=True)
 verify=subprocess.run([sys.executable,'-m','recipe.phimm.data.short.word_1to100.verify','--plans',str(plans),'--rendered',str(root/'remaining_render_a1'),'--report',str(root/'remaining_render_a1.audit.json'),'--workers','8'],env=env)
 if verify.returncode:print(json.dumps({'phase':'verify_failed','exit':verify.returncode}),flush=True);sys.exit(verify.returncode)
 print(json.dumps({'phase':'verified','audit':str(root/'remaining_render_a1.audit.json')}),flush=True)
finally:
 if prefetch.poll() is None:prefetch.terminate();prefetch.wait()

import json,glob,collections,math
from pathlib import Path
r=Path(__file__).parent
sources={}
for path in (r/'aligned').glob('*.jsonl'):
 for line in path.open():
  x=json.loads(line);w=x['word_timestamps'];x['word_timestamps']=json.loads(w) if isinstance(w,str) else w
  key=(x['audio_path'],x['id']);assert key not in sources;sources[key]=x
counts=collections.Counter(); used=collections.defaultdict(set)
for path in (r/'plans').glob('*.jsonl'):
 for line in path.open():
  x=json.loads(line);segs=x['segments'];assert 1<=len(segs)<=2 and len({s['id'] for s in segs})==len(segs)
  assert x['text']==' '.join(s['text'] for s in segs)
  assert 0<sum(s['end']-s['start'] for s in segs)<30
  assert abs(sum(s['end']-s['start'] for s in segs)-x['target'])<=.500001
  for seg in segs:
   src=sources[(seg['audio_path'],seg['id'])]; w=src['word_timestamps'];i=seg['start_word_index'];j=seg['end_word_index_exclusive'];assert 0<=i<j<=len(w)
   assert seg['start']==w[i]['start'] and seg['end']==w[j-1]['end'];assert seg['text']==' '.join(z['text'] for z in w[i:j]);assert seg['source_text']==src['text']
   for z in w[i:j]:assert z['end']>z['start']
   used[path.stem.split('_')[0]].add(seg['id']);counts['segments']+=1
  counts['samples']+=1
out={'counts':dict(counts),'distinct_source_ids':{k:len(v) for k,v in used.items()}}
(r/'evidence'/'plan_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

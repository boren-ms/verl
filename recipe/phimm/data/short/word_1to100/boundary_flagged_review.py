#!/usr/bin/env python3
"""Select the highest-discontinuity example per dataset for listening triage."""
import json,pathlib
import soundfile as sf
r=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');root=r/'merged_a1';meta=json.loads((root/'COMPLETE.json').read_text());out=r/'boundary_flagged_review_a1.jsonl';assert not out.exists();flags={}
with out.open('x') as f:
 for name in sorted(meta['counts']):
  best=None;count=0
  for line in (root/name/'data.jsonl').open():
   x=json.loads(line);path=root/name/'audio'/(x['id']+'.wav');peak=0.0
   with sf.SoundFile(path) as wav:
    for seg in x['source_segments'][1:]:
     i=seg['output_start_sample'];wav.seek(i-1);pair=wav.read(2,dtype='float32');assert len(pair)==2;peak=max(peak,abs(float(pair[1])-float(pair[0])))
   if peak>0.1:
    count+=1
    if best is None or peak>best[0]:best=(peak,x)
  flags[name]={'flagged_samples':count,'selected_max_jump':best[0] if best else None}
  if best:f.write(json.dumps({'selection':'max_join_jump_over_0_1','dataset':name,'max_join_jump':best[0],**best[1]},ensure_ascii=False)+'\n')
  print(json.dumps({name:flags[name]}),flush=True)
(r/'boundary_flagged_summary_a1.json').write_text(json.dumps({'threshold':0.1,'subsets':flags},indent=2,sort_keys=True)+'\n')

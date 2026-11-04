import json,glob,collections,re,math,hashlib
from pathlib import Path
root=Path(__file__).parent
stats={};ids={};starts={};durations={}
for split,patterns in {'train':['train-0','train-1'],'test-clean':['test-clean'],'test-other':['test-other-0','test-other-1']}.items():
 c=collections.Counter(); seen=set(); lens=[];paths=set()
 for name in patterns:
  p=root/'aligned'/f'{name}.jsonl'
  for line in p.open():
   x=json.loads(line);c['rows']+=1;id=x['id'].removesuffix('.flac');src=x['audio_path']; sp=src.split('/LibriSpeech/')[1].split('/')[0];c['split_'+sp]+=1
   if id in seen: c['duplicate_id']+=1
   seen.add(id);paths.add(src)
   if (split=='train' and sp not in ('train-clean-100','train-clean-360','train-other-500')) or (split!='train' and sp!=split): c['wrong_split']+=1
   try:
    w=x['word_timestamps'];w=json.loads(w) if isinstance(w,str) else w
    tok=lambda s:re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*",s.lower())
    if not w or tok(' '.join(t['text'] for t in w))!=tok(x['text']): raise ValueError('text_mismatch_or_empty')
    prev=0
    for t in w:
     a,b=float(t['start']),float(t['end'])
     if not math.isfinite(a) or not math.isfinite(b) or a<prev or b<=a: raise ValueError('invalid_zero_or_overlap')
     prev=b
    lens.append(float(w[-1]['end'])-float(w[0]['start']));c['valid_full_alignment']+=1
   except Exception as e:c[str(e)]+=1
 stats[split]=dict(c);ids[split]=seen;starts[split]=paths;durations[split]=lens
stats['overlap']={'train_test_clean_ids':len(ids['train']&ids['test-clean']),'train_test_other_ids':len(ids['train']&ids['test-other']),'test_clean_other_ids':len(ids['test-clean']&ids['test-other'])}
stats['alignment_input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'aligned').glob('*.jsonl')}
(root/'evidence'/'input_audit.json').write_text(json.dumps(stats,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,5))
for split,d in durations.items():ax.hist(d,bins=range(0,41),histtype='step',density=True,label=f'{split} (n={len(d)})')
ax.set(xlabel='First to last estimated word boundary (s)',ylabel='Fraction per 1-second bin',title='Source alignment lengths — before creating output clips');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'evidence'/'source_lengths.png',dpi=150)
print(json.dumps(stats,indent=2))

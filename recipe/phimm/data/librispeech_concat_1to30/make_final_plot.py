import json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
r=json.loads((p/'evidence'/'verification.json').read_text())['subsets']
fig,axs=plt.subplots(1,2,figsize=(13,5));bins=list(range(1,30));hist=r['train']['histogram_1s'];counts=[hist.get(str(i),0) for i in bins];axs[0].bar(bins,counts,color='#2563eb');axs[0].axhline(500,color='#dc2626',ls='--',label='Target: 500 per 1-second bin');axs[0].set(xlabel='Actual output duration bin (seconds)',ylabel='Number of WAVs',title='Train: verified actual vs target');axs[0].legend()
labels=[];values=[];colors=[];targets=[]
for j,split in enumerate(('clean','other')):
 for t in (5,10,20):
  path=p/'manifests'/f'test-{split}_{t}s.jsonl';v=[json.loads(line)['duration'] for line in path.open()]
  assert len(v)==r[f'test-{split}_{t}s']['count'];values.append([d-t for d in v]);labels.append(f'{split} {t}s');targets.append(t);colors.append(('#2563eb','#d97706')[j])
bp=axs[1].boxplot(values,positions=range(1,7),widths=.55,patch_artist=True,showfliers=False)
for box,color in zip(bp['boxes'],colors):box.set_facecolor(color);box.set_alpha(.6)
axs[1].axhline(0,color='#dc2626',label='Target duration');axs[1].axhline(.5,color='#888',ls='--',lw=1,label='Tolerance ±0.5 s');axs[1].axhline(-.5,color='#888',ls='--',lw=1);axs[1].set_xticks(range(1,7),labels,rotation=35);axs[1].set(ylabel='Actual duration minus target (seconds)',title='Verified held-out duration error');axs[1].legend();fig.tight_layout();fig.savefig(p/'evidence'/'actual_distribution.png',dpi=180)
with (p/'evidence'/'actual_train_histogram.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['duration_bin_start','actual_count','target_count']);w.writerows((i,c,500) for i,c in zip(bins,counts))

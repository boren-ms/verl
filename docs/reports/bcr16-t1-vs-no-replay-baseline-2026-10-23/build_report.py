import json,math,textwrap,hashlib,collections,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
P=Path(__file__).resolve().parent;F=P/'figures';F.mkdir(exist_ok=True)
E=json.loads((P/'evidence.json').read_text());H={l:{int(s):m for s,m in h.items()} for l,h in E['histories'].items()};DS=['openml_bad','earning_chunk_verb','ls_clean','ls_other'];N={'openml_bad':'OpenML bad','earning_chunk_verb':'Earnings (6 parents)','ls_clean':'LibriSpeech clean','ls_other':'LibriSpeech other'};L={'baseline':'Baseline: L2 T1','target':'BCR16 T1'};C={'baseline':'#3c6ca8','target':'#d36332'};STEP=E['matched_validation_step'];manifest=[];narr=[]
font='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
if Path(font).exists():font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'font.size':16,'axes.titlesize':19,'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':15,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
pdf=PdfPages(P/'report.pdf');page=0

def finish(fig,title,stem,caption,keys=None,steps=None):
 global page
 page+=1;fig.set_size_inches(16,9);fig.suptitle(title,x=.055,ha='left',y=.975,fontsize=27,fontweight='bold');fig.text(.055,.027,caption,fontsize=12,va='bottom');fig.text(.95,.027,str(page),ha='right',fontsize=12)
 pdf.savefig(fig);fig.savefig(F/(stem+'.png'),dpi=135);fig.savefig(F/(stem+'.svg'));fig.savefig(F/(stem+'.pdf'));plt.close(fig);manifest.append({'figure':stem,'exact_keys':keys or [],'steps':steps,'run_ids':{l:E['run_inventory'][l]['id'] for l in L},'transform':'p_err multiplied by100; reward unsmoothed raw means; window means arithmetic across exact shared optimizer steps','source':'evidence.json; sources/*_wandb.json; sources/matched_dumps.json'});narr.append('## '+title+'\n\n'+caption+'\n\n![figure](figures/'+stem+'.png)\n')

def textslide(title,blocks,stem):
 fig=plt.figure();y=.84
 for head,body in blocks:
  fig.text(.06,y,head,fontsize=22,fontweight='bold',va='top');y-=.047
  lines=textwrap.wrap(body,112);fig.text(.06,y,'\n'.join(lines),fontsize=18,va='top',linespacing=1.3);y-=.032*len(lines)+.038
 finish(fig,title,stem,'Pinned source audit and collected W&B configs; live stop verified by exact Ray submission.')

textslide('Replay changes sampling and reward scale',[
 ('Control and experiment','Baseline L2_t1 (d807d91d): temperature1, no replay,128 fresh prompts. BCR16_T1 (73acea80):112 fresh + up to16 replay,400-prompt buffer,worst greedy word score,seed42. Replay regenerates outputs using current policy.'),
 ('Reward and shared settings','Reward divisor4 →3; same component weights and clip[-2,2]. This scales the same weighted sum by4/3. Both: ReMax,Qwen9B,LoRA320,LR3e-6 constant,8 rollouts,16 GPUs,two epochs; validation temperature0/n1.'),
 ('Provenance and limits','Baseline03680b38 →target1d72d5a. Scorer,normalizer,actor loss and core advantage code identical; replay integration changes trainer input sampling. Multiple recoveries reset replay; equal steps have different fresh-data exposure.'),
 ('Training stopped','User stop confirmed by Ray: recovery750/driver10000000 STOPPED,0 allocated GPUs. Monitor deleted. Historical last confirmed optimizer755; recovery750 stopped during initial validation with no new optimizer update.')],'01_config')

for i,pair in enumerate([DS[:2],DS[2:]]):
 fig,axes=plt.subplots(1,2,figsize=(16,9));fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.25,wspace=.28)
 for ax,ds in zip(axes,pair):
  for l in L:
   pts=E['observed_initial_latest_best'][ds][l]['points'];ax.plot([x['step'] for x in pts],[100*x['value'] for x in pts],'-o',ms=4,color=C[l],label=L[l]);o=E['observed_initial_latest_best'][ds][l];txt=' / '.join(f"{k}: {100*o[k]['value']:.3f}% @{o[k]['step']}" for k in ['initial','latest','best']);ax.text(0,-.23 if l=='baseline' else -.33,txt,transform=ax.transAxes,fontsize=12,color=C[l])
  ax.set_title(N[ds]);ax.set_xlabel('Actual optimizer / validation step');ax.set_ylabel('p_err (%) — lower is better');ax.legend();ax.axvline(600,color='#888',ls=':',lw=1)
 finish(fig,'Validation trajectories — preserve dataset tradeoffs',f'02_validation_{i}','Observed points only; lines are guides. Target latest740 vs baseline600 is an unequal-budget comparison. Initial/latest/best shown separately.',[f'val-core/{x}/p_err/mean@1' for x in pair],{'matched':600,'baseline_latest':600,'target_latest':740})
fig,ax=plt.subplots();fig.subplots_adjust(top=.83,bottom=.21,left=.28,right=.95);vals=[E['validation_comparison'][ds]['delta_pp'] for ds in DS];ax.barh(range(4),vals,color=['#d36332' if v>0 else '#3c6ca8' for v in vals]);ax.set_yticks(range(4),[N[ds] for ds in DS]);ax.invert_yaxis();ax.axvline(0,color='#444');ax.set_xlabel('BCR16 T1 − baseline p_err (percentage points); negative improves')
for i,(ds,v) in enumerate(zip(DS,vals)):
 d=E['validation_comparison'][ds];ax.text(v+(.002 if v>=0 else -.002),i,f"{v:+.4f} pp\n{100*d['baseline']:.4f}% → {100*d['target']:.4f}%",ha='left' if v>=0 else 'right',va='center',fontsize=17)
ax.set_xlim(-.16,.085)
finish(fig,'At step600: LS gains; Earnings slightly regresses','03_matched','Exact shared step600. No cross-dataset averaging; single-run descriptive comparison with reward-scale and exposure confounds.',[f'val-core/{x}/p_err/mean@1' for x in DS],[600])
selected={'reward-core':['char','word','word_kw','score'],'reward-delta':['char','word','word_del','word_ins'],'reward-vote':['char','word','lang','fmt']}
for family,components in selected.items():
 keys=[family+'/'+x+'/mean' for x in components];fig,axs=plt.subplots(2,2);fig.subplots_adjust(top=.83,bottom=.17,left=.08,right=.97,hspace=.46,wspace=.25)
 for ax,key in zip(axs.flat,keys):
  for l in L:
   pts=[(s,m[key]) for s,m in sorted(H[l].items()) if key in m and 0<s<=600];ax.plot([s for s,v in pts],[v for s,v in pts],color=C[l],lw=.7,alpha=.75,label=L[l])
  ax.set_title(key,fontsize=17);ax.set_xlabel('Actual optimizer step');ax.set_ylabel('Logged component mean');ax.legend(fontsize=12)
 finish(fig,f'{family}: raw batch means through step600',family+'_traces','No smoothing/interpolation. Logged history includes abandoned pre-recovery updates; all22 keys retained. Batches differ by replay.',keys,[1,600])
 fig,axs=plt.subplots(2,2);fig.subplots_adjust(top=.83,bottom=.18,left=.08,right=.97,hspace=.48,wspace=.25)
 for ax,key in zip(axs.flat,keys):
  w=E['reward_window'][key];ax.bar([0,1],[w['baseline'],w['target']],color=[C['baseline'],C['target']]);ax.set_xticks([0,1],['L2 T1','BCR16 T1']);ax.set_title(key,fontsize=17);ax.set_ylabel('Arithmetic mean');ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
  for x,l in enumerate(L):ax.text(x,.90,f"{w[l]:.6g}",transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=14,bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=3))
 finish(fig,f'{family}: exact shared window591–600',family+'_means','10 observations per run per component; arithmetic means. Core score has a different divisor and is not a like-for-like quality metric.',keys,list(range(591,601)))
fig,axes=plt.subplots(1,2);fig.subplots_adjust(top=.82,bottom=.23,left=.08,right=.97,wspace=.3)
for ax,pair in zip(axes,[DS[:2],DS[2:]]):
 labels=[];im=[];bad=[]
 for ds in pair:
  q=E['examples'][ds];labels.append(N[ds]+'\nmatched '+str(q['matched']));im.append(q['counts'].get('improved',0));bad.append(q['counts'].get('newly_failed',0)+q['counts'].get('worsened_persistent',0))
 x=np.arange(2);ax.bar(x-.18,im,.36,label='Fewer errors',color=C['baseline']);ax.bar(x+.18,bad,.36,label='More errors',color=C['target']);ax.set_xticks(x,labels);ax.set_ylabel('Matched scored units');ax.legend(fontsize=13)
 for i in range(2):ax.text(i-.18,im[i]+.5,str(im[i]),ha='center');ax.text(i+.18,bad[i]+.5,str(bad[i]),ha='center')
finish(fig,'Matched examples quantify improvements and regressions','07_patterns','Step600 full dumps; unchanged stable IDs and references. Earnings grouped341 chunks into6 parents; OpenML393,LS clean2620,LS other2939.',[],[600])

examples=[]
for ds in ['ls_clean','ls_other','openml_bad']:
 pairs=E['examples'][ds]['all_pairs'];examples.extend((ds,q) for q in [min(pairs,key=lambda q:q['delta_errors']),max(pairs,key=lambda q:q['delta_errors'])])
# Short excerpts explicitly retain source identity, full originals in evidence.
def excerpt(q,field,limit=190):
 s=q[field]
 if q['id']=='3729-6852-0046':return '… '+s[-175:]
 if len(s)<=limit:return s
 if q['id']=='6070-86745-0002':return s[:120]+' …'
 return s[:limit]+' …'
for i in range(3):
 fig=plt.figure();y=.84
 for ds,q in examples[i*2:i*2+2]:
  head=f"{N[ds]} | ID {q['id']} | errors {q['baseline_errors']} → {q['target_errors']}"
  fig.text(.055,y,head,fontsize=20,fontweight='bold',va='top');y-=.065
  for label,field in [('REF','reference'),('L2 T1','baseline_output'),('BCR16','target_output')]:
   lines=textwrap.wrap(label+': '+excerpt(q,field),112);fig.text(.055,y,'\n'.join(lines),fontsize=16,va='top',linespacing=1.2);y-=.029*len(lines)+.013
  fig.text(.055,y,f"Source: matched_dumps.json / {ds}; baseline line{q['baseline_line']}, target line{q['target_line']}; step600",fontsize=12,color='#555');y-=.075
 finish(fig,'Stable-ID cases: content and scoring-sensitive forms',f'08_examples_{i}','Excerpts are illustrative; complete outputs, references and logged error counts retained in evidence.json. No acoustic cause inferred.',[],[600])
textslide('Earnings: six parents, small net change hides mixed effects',[
 ('Full matched scope','All341 chunks form6 scored parent recordings,each counted once. Two parents improve,four regress; net+3 word errors. Official p_err4.4560% →4.4755%(+0.0195pp).'),
 ('Largest parent regression','Stable parent c6efdf26-e19a-4e33-8bbc-c40283fea3a3.mp3:65 →74 errors(+9),step600. Example: baseline “open the call to Q&A” vs target “open a call Q&A”; these are excerpts from one parent,not nine independently diagnosed causes.'),
 ('Largest parent improvement','Parent03c13d67-c016-4a2b-91d8-e013d5d6bd2b.mp3 improves by11 errors. Mixed parent-level directions make a blanket Earnings regression claim too broad.'),
 ('Source and limit','Complete parent references,chunk line lists,raw outputs and grouped clean outputs are in evidence.json →examples →earning_chunk_verb →all_pairs. Text inspection alone cannot identify acoustic causes.')],'08_earnings')
textslide('Check scoring-sensitive changes before tuning rewards',[
 ('Evidence','At step600: LS clean−24 errors,LS other−42,Earnings+3,OpenML−2 net errors. OpenML has2 samples with changed n_ref,so its corpus ratio needs both denominators. Table snapshots match20/20 examples per run; the full dumps cover datasets absent from selected tables.'),
 ('Bounded next check','Pinned scorer replay reproduces all12 selected cases. In OpenML ID11746087412974645092,parenthesized(2006) disappears from the normalized hypothesis: errors2 →6. Audit bracket/number normalization first; retain official scores and test a symmetric diagnostic convention on both runs.'),
 ('Success criteria','Keep official per-dataset p_err unchanged in the report. Any later one-factor trial should fix divisor/exposure confounds,reduce verified scoring-pattern errors at matched budget,and guard LS clean,LS other,OpenML and Earnings separately. No new training was launched.')],'09_next')
pdf.close();E['figures']=manifest;E['report_pdf_pages']=page;E['scoring_diagnostics']=json.loads((P/'sources/scoring_diagnostics.json').read_text());E['scorer_replay_all12_match_logged_counts']=all(q['logged_n_err']==q['target_pinned_scorer_replay']['n_err'] and q['logged_n_ref']==q['target_pinned_scorer_replay']['n_ref'] for q in E['scoring_diagnostics']['cases'])
# Null sparse entries are missing observations, not failures or zeros.
E['null_sparse_observations']=sum(1 for x in E['invalid_observations'] if x['reason']=='not finite valid numeric value');E['coverage_note']='Sparse null metrics were excluded; original nullable source records retained.'
(P/'evidence.json').write_text(json.dumps(E,ensure_ascii=False,indent=2,allow_nan=False))
md=['# BCR16 T1 versus its no-replay T1 baseline','Training STOPPED; monitor deleted; exact driver10000000 has0 GPU allocation.','## Shared validation step600','| Dataset | Baseline p_err | BCR16 T1 p_err | Delta pp |','|---|---:|---:|---:|']
for ds in DS:
 d=E['validation_comparison'][ds];md.append(f"| {N[ds]} | {100*d['baseline']:.4f}% | {100*d['target']:.4f}% | {d['delta_pp']:+.4f} |")
md+=['','## Run identities']
for l in L:
 r=E['run_inventory'][l];md.append(f"- {L[l]}: [{r['display_name']}]({r['url']}), ID `{r['id']}`. W&B tracking state `{r['wandb_tracking_state']}`; exact Ray state takes precedence. Complete unfiltered scan: {r['raw_history_rows']} rows.")
md+=['','## Initial, latest and independently best validation','|Dataset|Run|Initial|Latest|Best|','|---|---|---:|---:|---:|']
for ds in DS:
 for l in L:
  o=E['observed_initial_latest_best'][ds][l];md.append('|'+N[ds]+'|'+L[l]+'|'+'|'.join(f"{100*o[k]['value']:.4f}% @{o[k]['step']}" for k in ['initial','latest','best'])+'|')
md+=['','## Reward means: exact shared steps591–600','Raw measures and deltas have the same definitions; core/score includes divisor4 vs3 and is not directly comparable. Different replay batches limit interpretation.','|Exact key|Baseline mean|BCR16 mean|n per run|','|---|---:|---:|---:|']
for k,w in E['reward_window'].items():md.append(f"|`{k}`|{w['baseline']:.8f}|{w['target']:.8f}|{w['n']}|")
md+=['','## Full matched example scope','|Dataset|Matched units|Improved|More errors|Net added errors|Changed n_ref|','|---|---:|---:|---:|---:|---:|']
for ds in DS:
 q=E['examples'][ds];md.append(f"|{N[ds]}|{q['matched']}|{q['counts'].get('improved',0)}|{q['counts'].get('newly_failed',0)+q['counts'].get('worsened_persistent',0)}|{q['delta_errors']:+}|{q['denominator_changed_units']}|")
md+=['','## Verified scoring diagnostic','Pinned scorer replay reproduced12/12 selected case error counts. OpenML ID11746087412974645092 loses the parenthesized year2006 during normalization,adding4 errors(2→6). This is one verified scoring-sensitive case,not a population prevalence estimate. See sources/scoring_diagnostics.json.','','## Limits']+['- '+x for x in E['limits']]+['','## Figures and slide narrative']+narr
(P/'report.md').write_text('\n'.join(md));(P/'slides.md').write_text('\n'.join(narr));print(json.dumps({'pdf_pages':page,'figures':len(manifest)}))

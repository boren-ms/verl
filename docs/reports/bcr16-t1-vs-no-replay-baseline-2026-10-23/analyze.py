import json,math,re,datetime,hashlib,collections,difflib
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'sources'
labels=['baseline','target'];names={'ls_clean':'LibriSpeech clean','ls_other':'LibriSpeech other','earning_chunk_verb':'Earnings (6 parents)','openml_bad':'OpenML bad'}
runs={l:json.loads((S/(l+'_wandb.json')).read_text()) for l in labels};dumps=json.loads((S/'matched_dumps.json').read_text());hist={};conflicts=[];invalid=[];axes=[]
for label,run in runs.items():
 h={}
 for i,row in enumerate(run['history']):
  step=row.get('step');glob=row.get('training/global_step')
  if step is not None and glob is not None and step!=glob:axes.append({'run':label,'row':i,'step':step,'training/global_step':glob})
  if step is None:step=glob
  if step is None:
   if row.get('_step')==0 and any(k.startswith('val-') for k in row['metrics']):step=0
   else:continue
  step=int(step)
  for k,v in row['metrics'].items():
   if not isinstance(v,(int,float)) or not math.isfinite(v) or ('/p_err/' in k and v<0):invalid.append({'run':label,'step':step,'key':k,'reason':'not finite valid numeric value'});continue
   old=h.setdefault(step,{}).get(k)
   if old is not None and abs(v-old)>1e-12:conflicts.append({'run':label,'step':step,'key':k,'previous':old,'later':v,'policy':'later history observation shown, repeated values retained here'})
   h[step][k]=v
 hist[label]=h
keys=sorted(set(k for h in hist.values() for m in h.values() for k in m if re.match(r'^val-(core|aux)/.+/p_err/mean@1$',k)))
datasets=sorted(set(k.split('/')[1] for k in keys));aliases=[];shared={};validation={};observed={}
for ds in datasets:
 k=f'val-core/{ds}/p_err/mean@1';alias=k.replace('val-core','val-aux');shared[ds]=sorted(set(s for s,m in hist['baseline'].items() if k in m)&set(s for s,m in hist['target'].items() if k in m));observed[ds]={}
 for l in labels:
  points=[{'step':s,'value':m[k]} for s,m in sorted(hist[l].items()) if k in m]
  observed[ds][l]={'initial':points[0] if points else None,'latest':points[-1] if points else None,'best':min(points,key=lambda x:x['value']) if points else None,'points':points}
  for s,m in hist[l].items():
   if k in m and alias in m and abs(m[k]-m[alias])>1e-12:aliases.append({'run':l,'step':s,'dataset':ds,'core':m[k],'aux':m[alias]})
STEP=max(set.intersection(*(set(v) for v in shared.values())));assert STEP==600 and dumps['step']==STEP
for ds in datasets:
 b,t=(hist[l][STEP][f'val-core/{ds}/p_err/mean@1'] for l in labels);validation[ds]={'baseline':b,'target':t,'delta_pp':100*(t-b),'relative_error_reduction_pct':100*(b-t)/b if b else None}
inventory={l:{f:sorted(set(k for m in hist[l].values() for k in m if k.startswith(f+'/') and k.endswith('/mean'))) for f in ['reward-core','reward-delta','reward-vote']} for l in labels};reward={}
for k in sorted(set(k for inv in inventory.values() for arr in inv.values() for k in arr)):
 common=sorted(s for s in set(hist['baseline'])&set(hist['target']) if 0<s<=STEP and all(k in hist[l][s] for l in labels));window=[s for s in common if STEP-9<=s<=STEP]
 reward[k]={'shared_steps':common,'window':window,'n':len(window),**{l:sum(hist[l][s][k] for s in window)/len(window) if window else None for l in labels}}
examples={}
for ds,sources in dumps['datasets'].items():
 maps={};stats={};allrows={}
 for l in labels:
  rows=[dict(r,_line=i) for i,r in enumerate(sources[l]['rows'],1)];allrows[l]=rows;groups=collections.defaultdict(list)
  for r in rows:
   if ds=='earning_chunk_verb':groups[r['parent_audio_path']].append(r)
  scored=[r for r in rows if r.get('n_err') is not None and r.get('n_ref') is not None];m={};duplicate=0
  for r in scored:
   ident=r['parent_audio_path'] if ds=='earning_chunk_verb' else str(r.get('id') or r.get('audio_path'));key=(ident,r['gts']);q=dict(r)
   if ds=='earning_chunk_verb':
    members=sorted(groups[ident],key=lambda x:(x['seg_start'],x['_line']));assert len(members)==r['n_segments'] and len({x['seg_index'] for x in members})==len(members)
    assert len({x['gts'] for x in members})==1
    q['clean_output']='\n'.join(x['clean_output'] for x in members);q['raw_segments']=[x['output'] for x in members];q['_lines']=[x['_line'] for x in members];q['_segments']=len(members)
   if key in m:duplicate+=1
   else:m[key]=q
  maps[l]=m;stats[l]={'raw_rows':len(rows),'scored_units':len(scored),'null_count_rows_excluded':len(rows)-len(scored),'unique_units':len(m),'duplicates':duplicate,'n_err':sum(x['n_err'] for x in m.values()),'n_ref':sum(x['n_ref'] for x in m.values()),'source':sources[l]['path']}
  ratio=stats[l]['n_err']/stats[l]['n_ref'];assert abs(ratio-validation[ds][l])<1e-10,(ds,l,ratio,validation[ds][l])
 common=maps['baseline'].keys()&maps['target'].keys();counts=collections.Counter();pairs=[]
 for key in sorted(common):
  b,t=(maps[l][key] for l in labels);be,te=b['n_err'],t['n_err'];typ='both_correct' if be==te==0 else 'improved' if te<be else 'newly_failed' if be==0 and te>0 else 'worsened_persistent' if te>be else 'persistent_equal_errors';counts[typ]+=1
  q={'id':key[0],'reference':key[1],'baseline_output':b['clean_output'],'target_output':t['clean_output'],'baseline_errors':be,'target_errors':te,'delta_errors':te-be,'baseline_n_ref':b['n_ref'],'target_n_ref':t['n_ref'],'baseline_line':b['_line'],'target_line':t['_line'],'language':b.get('language'),'pattern':typ,'baseline_raw':b.get('raw_segments',b['output']),'target_raw':t.get('raw_segments',t['output'])}
  if ds=='earning_chunk_verb':q.update(baseline_lines=b['_lines'],target_lines=t['_lines'],segments=b['_segments'])
  sm=difflib.SequenceMatcher(None,q['baseline_output'].split(),q['target_output'].split());changed=[];bw=q['baseline_output'].split();tw=q['target_output'].split()
  for tag,i,j,x,y in sm.get_opcodes():
   if tag!='equal':changed.append({'baseline':' '.join(bw[max(0,i-5):min(len(bw),j+5)]),'target':' '.join(tw[max(0,x-5):min(len(tw),y+5)]),'operation':tag})
  q['raw_text_differences']=changed;pairs.append(q)
 pairs.sort(key=lambda x:x['delta_errors']);nonzero=[x for x in pairs if x['delta_errors']!=0];chosen=nonzero[:2]+nonzero[-2:]
 examples[ds]={'stats':stats,'matched':len(common),'baseline_only':len(maps['baseline'])-len(common),'target_only':len(maps['target'])-len(common),'counts':dict(counts),'threshold':'n_err > 0; improved means any decrease, including persistent errors','delta_errors':sum(x['delta_errors'] for x in pairs),'denominator_changed_units':sum(x['baseline_n_ref']!=x['target_n_ref'] for x in pairs),'examples':chosen,'all_pairs':pairs}
# Retrieve actual tables; flatten cumulative snapshots then deduplicate identical entries.
tables={}
for l,run in runs.items():
 entries={};allsteps=set()
 for f in run['tables']:
  if 'local_path' not in f:continue
  table=json.loads((P/f['local_path']).read_text());cols=table['columns']
  for ri,raw in enumerate(table['data'],1):
   row=dict(zip(cols,raw));step=int(row['step']);allsteps.add(step)
   if step!=STEP:continue
   for column in cols:
    match=re.fullmatch(r'output_(\d+)',column)
    if not match:continue
    slot=match.group(1);q={'step':step,'input':row.get('input_'+slot),'output':row[column],'reference':row.get('ground_truth_'+slot),'score':row.get('score_'+slot),'table':f['local_path'],'row':ri,'slot':int(slot)}
    identity=(q['step'],q['input'],q['output'],q['reference'],q['score']);entries.setdefault(identity,q)
 resolved=[]
 for q in entries.values():
  matches=[]
  for ds,data in dumps['datasets'].items():
   for i,r in enumerate(data[l]['rows'],1):
    if r['input']==q['input'] and r['output']==q['output'] and r['gts']==q['reference'] and abs(float(r['score'])-float(q['score']))<1e-5:matches.append({'dataset':ds,'id':r.get('id'),'parent':r.get('parent_audio_path'),'jsonl_line':i})
  q['matches']=matches;resolved.append(q)
 tables[l]={'snapshot_count':len(run['tables']),'all_steps':sorted(allsteps),'deduplicated_matched_step_examples':resolved,'matched_examples':sum(len(x['matches'])==1 for x in resolved),'ambiguous':sum(len(x['matches'])>1 for x in resolved),'unmatched':sum(not x['matches'] for x in resolved),'dataset_counts':dict(collections.Counter(x['matches'][0]['dataset'] for x in resolved if len(x['matches'])==1))}
limits=['Baseline is L2_t1 without replay, temperature1; target is BCR16_T1 temperature1. Reward divisor changes4 to3 as well as replay; causal isolation unavailable.', 'Both runs were explicitly user-stopped. Target current recovery750 restored but stopped during initial validation before any new optimizer update; historical last finite optimizer755.', 'Target multiple infrastructure recoveries reset replay; W&B reuses one identity and may omit repeated steps. Monotonic W&B history is not a single uninterrupted lineage. Immutable per-segment startup proofs retained in source state.', 'Equal optimizer steps do not imply equal fresh-data exposure: baseline128 fresh, target112 fresh plus up to16 replay. Training batches may differ.', 'OpenML hypothesis-dependent normalization can change n_ref; do not assume fixed denominator or replace official p_err.', 'Initial/latest/best are descriptive; best steps differ across datasets and latest budgets differ. Selected table examples are not prevalence estimates.', 'All validation examples matched by stable identity plus identical reference; raw output differences are descriptive, not official edit alignment or acoustic diagnoses.']
evidence={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'matched_validation_step':STEP,'run_inventory':{l:{k:v for k,v in run.items() if k not in ['history','tables']} for l,run in runs.items()},'live_ray':json.loads((S/'stop_evidence.json').read_text()),'histories':hist,'validation_metric_keys':keys,'shared_validation_steps':shared,'validation_comparison':validation,'observed_initial_latest_best':observed,'reward_inventory':inventory,'reward_window':reward,'alias_conflicts':aliases,'metric_conflicts':conflicts,'invalid_observations':invalid,'axis_conflicts':axes,'examples':examples,'val_generations':tables,'source_audit':json.loads((S/'source_audit.json').read_text()),'config_differences':json.loads((S/'config_differences.json').read_text()),'limits':limits,'sources':{str(x.relative_to(P)):{'sha256':hashlib.sha256(x.read_bytes()).hexdigest(),'bytes':x.stat().st_size} for x in S.rglob('*.json')},'figures':[]}
(P/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'matched_step':STEP,'validation':validation,'latest':{ds:{l:o['latest'] for l,o in runs.items()} for ds,runs in observed.items()},'examples':{ds:{'matched':x['matched'],'counts':x['counts'],'delta_errors':x['delta_errors'],'nref_changed':x['denominator_changed_units']} for ds,x in examples.items()},'tables':{l:{k:v for k,v in q.items() if k!='deduplicated_matched_step_examples'} for l,q in tables.items()},'conflicts':len(conflicts),'alias_conflicts':len(aliases)},ensure_ascii=False,indent=2))

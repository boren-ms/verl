import json,re,socket,datetime,hashlib,math,os
from pathlib import Path
import wandb
P=Path(__file__).resolve().parent;S=P/'sources'
orig=socket.getaddrinfo
def routed(host,*args,**kwargs):
 return orig('172.28.40.11' if host=='msaip.wandb.io' else host,*args,**kwargs)
socket.getaddrinfo=routed
api=wandb.Api(overrides={'base_url':'https://msaip.wandb.io'},timeout=20)
for label,rid in [('baseline','d807d91db6c775c7d097c5ef7ae2f734'),('target','73acea8095b222e76c04affb7133c92e')]:
 try:
  start=datetime.datetime.now(datetime.timezone.utc).isoformat();run=api.run('genai/v2611/'+rid)
  rows=[];count=0
  for row in run.scan_history(page_size=1000):
   count+=1
   metrics={k:v for k,v in row.items() if re.match(r'^reward-(core|delta|vote)/.+/mean$',k) or re.match(r'^val-(core|aux)/.+/p_err/mean@1$',k) or k in ['actor/pg_loss','actor/grad_norm','actor/lr','timing_s/step']}
   if metrics or 'val/generations' in row:
    q={k:row.get(k) for k in ['step','training/global_step','_step','_timestamp']};q['metrics']=metrics
    if 'val/generations' in row:q['val/generations']=row['val/generations']
    rows.append(q)
  cfg=run.config;settings={}
  for path in ['actor_rollout_ref.rollout.temperature','actor_rollout_ref.rollout.val_kwargs.temperature','actor_rollout_ref.rollout.val_kwargs.n','actor_rollout_ref.model.path','actor_rollout_ref.actor.optim.lr','data.train_batch_size','trainer.experiment_name','trainer.validation_data_dir','trainer.total_training_steps','trainer.save_freq','trainer.test_freq','trainer.resume_mode','trainer.resume_from_path','algorithm.adv_estimator','algorithm.bad_case_replay','reward_functions','data','actor_rollout_ref.rollout','actor_rollout_ref.model.use_remove_padding','trainer.log_val_generations','trainer.total_epochs']:
   d=cfg
   for k in path.split('.'):
    d=d.get(k) if isinstance(d,dict) else None
   settings[path]=d
  data={'id':rid,'url':run.url,'display_name':run.name,'wandb_tracking_state':run.state,'commit':getattr(run,'commit',None),'created_at':run.created_at,'history':rows,'raw_history_rows':count,'complete_unsampled_scan':True,'method':'scan_history(page_size=1000), unfiltered from beginning, iterator exhausted','collected_started_at_utc':start,'collected_finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'settings':settings,'transport':'process-local internal DNS route with original hostname and TLS verification','tables':[]}
  for f in run.files():
   if f.name.startswith('media/table/') and f.name.endswith('.table.json'):
    dest=S/label;dest.mkdir(exist_ok=True)
    try:
     saved=f.download(root=str(dest),replace=True);p=Path(saved.name);t=json.loads(p.read_text());steps=sorted(set(x[0] for x in t['data'] if isinstance(x,list) and x))
     data['tables'].append({'remote_name':f.name,'local_path':str(p.relative_to(P)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'columns':t['columns'],'steps':steps,'rows':len(t['data'])})
    except Exception as e:data['tables'].append({'remote_name':f.name,'error_type':type(e).__name__})
  (S/(label+'_wandb.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'run':label,'rows':count,'relevant_rows':len(rows),'tables':len(data['tables']),'state':data['wandb_tracking_state']}),flush=True)
 except Exception as e:
  (S/(label+'_fetch_failure.json')).write_text(json.dumps({'error_type':type(e).__name__,'raw_error_omitted':True}))
  print(json.dumps({'run':label,'error_type':type(e).__name__}),flush=True)

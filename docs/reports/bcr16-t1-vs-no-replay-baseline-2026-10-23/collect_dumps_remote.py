import blobfile as bf,json,datetime
base='az://orngwus2cresco/data/boren/outputs/v2611/';out={'step':600,'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'datasets':{}}
for ds in ['ls_clean','ls_other','earning_chunk_verb','openml_bad']:
 out['datasets'][ds]={}
 for label,name in [('baseline','remax_dev_base0_v1h2a_fix_L2_t1'),('target','remax_dev_base0_v1h2a_fix_L2_RCB16_T1')]:
  path=base+name+'/val_data_gen/'+ds+'/600.jsonl'
  try:
   with bf.BlobFile(path,'r') as f:rows=[json.loads(l) for l in f if l.strip()]
   out['datasets'][ds][label]={'path':path,'rows':rows}
  except Exception as e:out['datasets'][ds][label]={'path':path,'error_type':type(e).__name__}
out['success']=True
print('T1_JSON='+json.dumps(out))

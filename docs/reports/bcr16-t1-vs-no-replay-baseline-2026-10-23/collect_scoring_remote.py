import blobfile as bf,json
from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer,measure_wer
from recipe.phimm.utils.languages import get_language_code
base='az://orngwus2cresco/data/boren/outputs/v2611/'
chosen={'ls_clean':['1188-133604-0019','3729-6852-0046'],'ls_other':['6070-86745-0002','2609-156975-0016'],'openml_bad':['11746087412974645092','482398938592448850']};out=[]
for ds,ids in chosen.items():
 for l,run in [('baseline','remax_dev_base0_v1h2a_fix_L2_t1'),('target','remax_dev_base0_v1h2a_fix_L2_RCB16_T1')]:
  with bf.BlobFile(base+run+'/val_data_gen/'+ds+'/600.jsonl','r') as f:rows=[json.loads(x) for x in f if x.strip()]
  for i,row in enumerate(rows,1):
   if str(row['id']) not in ids:continue
   hyp=row['clean_output'];ref=row['gts'];lang=get_language_code(row['language'].lower())
   if ds.startswith('ls_'):m=measure_openasr_en_wer(hyp,ref)
   else:
    hn,rn=normalize_for_wer(hyp,ref,lang=lang);m=measure_wer(hyp,ref,lang=lang);m.update(normalized_ref=rn,normalized_hyp=hn)
   out.append({'dataset':ds,'id':row['id'],'run':l,'line':i,'raw_output':row['output'],'clean_output':hyp,'reference':ref,'logged_n_err':row['n_err'],'logged_n_ref':row['n_ref'],'target_pinned_scorer_replay':m})
print('T1_JSON='+json.dumps({'success':True,'source_revision':'1d72d5a4a58c1ff5534d4efbeabfc5a5f0aa8a2f','scope':'same target pinned scorer applied to both logged cleaned outputs; no official metric replacement','cases':out}))

#!/usr/bin/env python3
"""Extract per-locale MLS diagnostic validation subsets from archived paired scores."""
import argparse, concurrent.futures as cf, csv, hashlib, json
from pathlib import Path
import blobfile as bf
import yaml
LANGS={'es':'Spanish','nl':'Dutch','it':'Italian','pt':'Portuguese','fr':'French'}
REMOTE='az://orngwus2cresco/data/boren/data/verl/mls_bad_val_v1_20261027'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).open()]
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def dataset(lang,regressed=False):
 suffix='regressed' if regressed else 'bad'
 return {'dataset_name':'jsonl','jsonl_paths':f'{REMOTE}/{lang}/'+('regressed.jsonl' if regressed else 'data.jsonl'),'cache_name':f'mls_{lang}_{suffix}_base2_560_v1','add_task_info':{'task':'lang_asr_verb','language':LANGS[lang],'prefix_prob':0.0,'lang_hint':True},'post_process':{'add_field':{'fields':{'data_source':f'{lang}_mls_{suffix}'}},'verl_format':{'prompt_key':'prompt'}}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='tmp/mls_bad_val_v1');p.add_argument('--publish',action='store_true');a=p.parse_args();root=Path(a.root);(root/'sources').mkdir(parents=True,exist_ok=True)
 for lang in LANGS:
  dest=root/'sources'/f'{lang}.jsonl'
  if not dest.exists():bf.copy(f'az://orngwus2cresco/data/boren/data/openasr_ml_jsonl/mls/{lang}/data.jsonl',str(dest),overwrite=False)
 summaries={};artifacts=[];combined=[];audio=[]
 for lang in LANGS:
  source=root/'sources'/f'{lang}.jsonl';src=read(source);original={str(r['id']):r for r in src};assert len(original)==len(src)
  folder=Path(f'tmp/asr-detail-compare/base2_best560_openall/{lang}_mls');csvpath=next(folder.glob('*.full.csv'));scores=list(csv.DictReader(csvpath.open()));assert len(scores)==len(src)
  targets=read(folder/'target_matched.jsonl');target={str(r['id']):r for r in targets};assert len(target)==len(targets);assert set(original)==set(target)=={r['id'] for r in scores}
  bad=[];regressed=[];cases=[];mismatches=[]
  for score in sorted(scores,key=lambda s:s['id']):
   ident=score['id'];r=original[ident];t=target[ident]
   if r['text']!=t['gts'] or r['text']!=score['raw_ref']:mismatches.append(ident);continue
   if int(score['target_errors'])==0:continue
   assert float(t['n_err'])==float(score['target_errors'])
   assert r.get('audio_path')
   out=dict(r);out['language']=LANGS[lang];out['data_source']=f'{lang}_mls_bad'
   # Keep hypotheses outside the runnable validation rows.
   bad.append(out);audio.append(out['audio_path'])
   delta=int(score['error_delta']);isreg=delta>0
   if isreg:regressed.append({**out,'data_source':f'{lang}_mls_regressed'})
   cases.append({'id':ident,'locale':lang,'source_audio_path':r['audio_path'],'reference':r['text'],'baseline_hypothesis':score['raw_hyp_baseline'],'target_hypothesis':score['raw_hyp_target'],'baseline_errors':int(score['baseline_errors']),'target_errors':int(score['target_errors']),'error_delta':delta,'regressed':isreg,'baseline_substitutions':int(score['baseline_substitutions']),'target_substitutions':int(score['target_substitutions']),'baseline_deletions':int(score['baseline_deletions']),'target_deletions':int(score['target_deletions']),'baseline_insertions':int(score['baseline_insertions']),'target_insertions':int(score['target_insertions'])})
  assert not mismatches, (lang,mismatches[:10])
  assert bad and regressed
  for name,rows in [('data.jsonl',bad),('regressed.jsonl',regressed),('error_cases.jsonl',cases)]:
   dest=root/lang/name;write(dest,rows);artifacts.append(dest)
  summaries[lang]={'source_rows':len(src),'bad_rows':len(bad),'regressed_rows':len(regressed),'reference_mismatches':0,'unmatched_ids':0,'bad_target_errors':sum(r['target_errors'] for r in cases),'source_manifest':f'az://orngwus2cresco/data/boren/data/openasr_ml_jsonl/mls/{lang}/data.jsonl','source_manifest_sha256':digest(source),'paired_csv':str(csvpath),'paired_csv_sha256':digest(csvpath),'target_jsonl_sha256':digest(folder/'target_matched.jsonl')}
  combined.append(dataset(lang))
  print(lang,summaries[lang],flush=True)
 # Keep one directly usable five-locale validation data config.
 # Scorer routing belongs to the consuming base config (base2_mls_v1.yaml).
 config=Path('recipe/phimm/config/data/val_data/mls_bad_verb_langhint.yaml')
 config.write_text('# Five locale-specific MLS diagnostic sets selected using base2@560 errors.\n'+yaml.safe_dump(combined,sort_keys=False,allow_unicode=True))
 evidence={'dataset':'mls_bad_val_v1','selection':'bad: archived target_errors > 0; regressed: target_errors > baseline_errors; all rows, not top-N','target':'original base2 checkpoint 560 (v3.5.4-r2 initialization)','baseline':'user-approved historical v3.5.3-r2, W&B 9ktcaceh; baseline used only for regressed subset','locales':summaries,'total_bad':sum(x['bad_rows'] for x in summaries.values()),'total_regressed':sum(x['regressed_rows'] for x in summaries.values()),'limitations':['Derived from held-out MLS evaluation, never use as training data.','Selection uses base2@560 outcomes; diagnostic and selection-biased, retain full MLS validation for overall quality.','Historical baseline has a different initialization; regression is descriptive, not isolated RL causality.','All word-error types retained, including spelling/normalization mismatches; no audio-based adjudication.'],'scorer':'original OpenASR-ML language-aware scoring via val_reward openasr_ml','build_script_sha256':digest(__file__)}
 (root/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2));artifacts.append(root/'evidence.json')
 # Check every referenced audio asset without materializing the full audio dataset.
 def stat(u):
  s=bf.stat(u);assert s.size>0;return {'url':u,'bytes':s.size}
 with cf.ThreadPoolExecutor(48) as ex:assets=list(ex.map(stat,sorted(set(audio))))
 (root/'audio_assets.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2));artifacts.append(root/'audio_assets.json')
 from recipe.phimm.data.dataset import jsonl_dataset,add_task_info
 for lang in LANGS:
  ds=jsonl_dataset(str(root/lang/'data.jsonl'));assert len(ds)==summaries[lang]['bad_rows']
  sample=add_task_info(ds.select([0]),task='lang_asr_verb',language=LANGS[lang],lang_hint=True,prefix_prob=0)
  assert sample[0]['text'] in sample[0]['gt_output'];assert LANGS[lang].lower() in json.dumps(sample[0]['prompt']).lower()
 print('All audio assets and native JSONL/task loading verified',flush=True)
 if a.publish:
  def upload(p):
   url=REMOTE+'/'+str(p.relative_to(root));h=digest(p)
   if bf.exists(url):
    with bf.BlobFile(url,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==h,url
   else:bf.copy(str(p),url,overwrite=False)
   with bf.BlobFile(url,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==h
   return {'path':url,'bytes':p.stat().st_size,'sha256':h}
  with cf.ThreadPoolExecutor(12) as ex:published=list(ex.map(upload,artifacts))
  (root/'publication.json').write_text(json.dumps({'artifacts':published},ensure_ascii=False,indent=2));print('Remote artifacts SHA256 verified',flush=True)
 print('COMPLETE',evidence['total_bad'],evidence['total_regressed'],flush=True)
if __name__=='__main__':main()

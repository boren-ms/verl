#!/usr/bin/env python3
"""Build audited FLEURS diagnostic subsets and train-only feature-selected data."""
import argparse, concurrent.futures as cf, csv, difflib, hashlib, io, json, re, unicodedata
from collections import Counter
from pathlib import Path
import blobfile as bf
import yaml
LANGS={'de':'German','es':'Spanish','fr':'French','it':'Italian','nl':'Dutch','pt':'Portuguese'}
ROOT=Path('tmp/fleurs_pattern_v1')
REMOTE='az://orngwus2cresco/data/boren/data/verl/fleurs_pattern_v1_20261027'
def sha(b):return hashlib.sha256(b).hexdigest()
def az(s):return re.sub(r'^/datablob1?/', 'az://orngwus2cresco/data/speech/',s)
def toks(t):return re.findall(r'[^\W_]+',unicodedata.normalize('NFC',t).casefold())
def norm(t):return ' '.join(toks(t))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def lines(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def fetch(url,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():bf.copy(url,str(p))
 return p.read_bytes()
def parse(raw,n):
 f=io.BytesIO(raw);assert f.read(3).lower()==b'sft';f.read(4);out=[]
 for i in range(n):
  assert int.from_bytes(f.read(4),'little')==i
  size=int.from_bytes(f.read(4),'little');b=f.read(size);assert len(b)==size;out.append(json.loads(b))
 assert not f.read(1);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 heldout=set();summaries={};lexicons={};examples=[];val=[];patterns={};artifacts=[]
 for lang,language in LANGS.items():
  folder=Path(f'tmp/asr-detail-compare/base2_best560_openall/{lang}_fleurs')
  scores=list(csv.DictReader(next(folder.glob('*.full.csv')).open()))
  source=ROOT/'sources'/f'{lang}_test.jsonl';fetch(f'az://orngwus2cresco/data/boren/data/openasr_ml_jsonl/fleurs/{lang}/data.jsonl',source)
  originals={str(r['id']):r for r in map(json.loads,source.read_text().splitlines())}
  assert len(originals)==len(scores);assert set(originals)=={s['id'] for s in scores}
  bad=[];reg=[];cases=[];words=Counter();edits=Counter()
  for s in scores:
   r=originals[s['id']];assert r['text']==s['raw_ref'];heldout.add(norm(r['text']))
   if int(s['target_errors'])==0:continue
   row={**r,'language':language,'data_source':f'{lang}_fleurs_bad'};assert row.get('audio_path');bad.append(row)
   delta=int(s['error_delta'])
   if delta>0:reg.append({**row,'data_source':f'{lang}_fleurs_regressed'})
   case={'id':s['id'],'language':language,'reference':s['raw_ref'],'baseline':s['raw_hyp_baseline'],'target':s['raw_hyp_target'],'error_delta':delta,'target_errors':int(s['target_errors']),'regressed':delta>0};cases.append(case)
   for tag,i,j,k,l in difflib.SequenceMatcher(None,s['ref'].split(),s['hyp_target'].split(),autojunk=False).get_opcodes():
    if tag=='equal':continue
    edits[tag]+=1
    for w in s['ref'].split()[i:j]:
     for t in toks(w):words[t]+=1
  docfreq=Counter(t for r in originals.values() for t in set(toks(r['text'])))
  lexicons[lang]={t:n for t,n in words.most_common() if len(t)>=4 and docfreq[t]<=max(5,len(originals)*0.10)}
  lexicons[lang]=dict(list(lexicons[lang].items())[:100]);patterns[lang]={'edit_span_counts':dict(edits),'reference_error_tokens':words.most_common(30),'examples':sorted(cases,key=lambda r:(-r['error_delta'],-r['target_errors']))[:8]}
  for name,rows in [('data.jsonl',bad),('regressed.jsonl',reg),('error_cases.jsonl',cases)]:
   dest=ROOT/'validation'/lang/name;lines(dest,rows);artifacts.append(dest)
  s=json.loads(next(folder.glob('*.summary.json')).read_text());summaries[lang]={k:s[k] for k in ['rows_compared','baseline_wer','target_wer','baseline_total_errors','target_total_errors','improved_count','degraded_count']};summaries[lang].update(bad_rows=len(bad),regressed_rows=len(reg))
  val.append({'dataset_name':'jsonl','jsonl_paths':f'{REMOTE}/validation/{lang}/data.jsonl','add_task_info':{'task':'lang_asr_verb','language':language,'prefix_prob':0.0,'lang_hint':True},'post_process':{'add_field':{'fields':{'data_source':f'{lang}_fleurs_bad'}},'verl_format':{'prompt_key':'prompt'}}})
 dump(ROOT/'error_patterns.json',patterns);dump(ROOT/'comparison_summary.json',summaries)
 tasks=[];sources=[]
 for name in ['asr_chunk_fleurs','asr_chunk_fleurs_tier2']:
  url=f'az://orngwus2cresco/data/speech/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/{name}.json';sp=ROOT/'sources'/f'{name}.json';ds=json.loads(fetch(url,sp))
  for source in ds['data_sources']:
   lang=source['language']
   # Exclude random-prompt replicas, preserving original same-language ASR train data.
   if lang not in LANGS or '_random_pt' in source['chunk_path']:continue
   assert 'file_set_train' in source['manifest_file'];mp=ROOT/'sources'/f'{lang}_train_manifest.json';manifest=json.loads(fetch(az(source['manifest_file']),mp));sources.append({'spec':url,'spec_sha256':sha(sp.read_bytes()),'source':source,'manifest_sha256':sha(mp.read_bytes())})
   for ch in manifest['fileInfo']:tasks.append((lang,source,ch))
 candidates={l:[] for l in LANGS};stats={l:Counter() for l in LANGS}
 def load(task):
  lang,s,ch=task;url=az(s['trans_path'].rstrip('/')+'/'+ch['name']+'.sft');raw=fetch(url,ROOT/'sft_cache'/f'{lang}_{ch["name"]}.sft');return task,url,sha(raw),parse(raw,ch['count'])
 proofs=[]
 print('Scanning',len(tasks),'FLEURS train chunks',flush=True)
 with cf.ThreadPoolExecutor(24) as ex:
  for (lang,s,ch),url,digest,rows in ex.map(load,tasks):
   proofs.append({'url':url,'sha256':digest,'rows':len(rows)});stats[lang]['scanned']+=len(rows)
   for i,obj in enumerate(rows):
    assert isinstance(obj,list) and len(obj)==1
    text=obj[0]['messages'][1]['content'];assert isinstance(text,str)
    ts=toks(text);canonical=norm(text)
    if canonical in heldout:stats[lang]['heldout_text_excluded']+=1;continue
    if not 5<=len(ts)<=160:continue
    hits=sorted(set(ts)&set(lexicons[lang]));digits=[t for t in ts if any(c.isdigit() for c in t)];accent=[t for t in ts if any(unicodedata.combining(c) for c in unicodedata.normalize('NFD',t))]
    tags={}
    if hits:tags['observed_error_reference_lexemes']=hits
    if digits:tags['numeric_expressions']=digits
    if len(accent)>=3:tags['accent_dense_reference']=accent
    if not (len(hits)>=2 or digits or len(accent)>=3):continue
    score=sum(min(lexicons[lang][t],5) for t in hits)+4*len(digits)+min(len(accent),6)
    audio=az(s['chunk_path'].rstrip('/')+'/'+ch['name']+'.audios')
    ident=f'fleurs_train_{lang}_{ch["name"]}_{i}'
    candidates[lang].append({'id':ident,'audio_chunk':f'{audio}:{ch["count"]}:{i}','text':text,'language':LANGS[lang],'selection':{'tags':tags,'score':score,'text_sha256':sha(canonical.encode()),'source_split':'train','sft_chunk':url,'sft_chunk_sha256':digest,'chunk_index':i,'evidence':'reference_features_not_confirmed_model_errors'}})
 selected=[];seen=set()
 for lang in LANGS:
  counts=Counter()
  for r in sorted(candidates[lang],key=lambda r:(-r['selection']['score'],sha(r['id'].encode()))):
   h=r['selection']['text_sha256'];ch=r['selection']['sft_chunk']
   if h in seen or counts[ch]>=200:continue
   seen.add(h);counts[ch]+=1;selected.append(r);stats[lang]['selected']+=1
   if stats[lang]['selected']>=2000:break
  assert stats[lang]['selected']>0
  examples.extend([r for r in selected if r['language']==LANGS[lang]][:2])
 selected.sort(key=lambda r:sha(r['id'].encode()));out=ROOT/'fleurs_pattern_train_v1.jsonl';lines(out,selected);artifacts.append(out)
 dump(ROOT/'two_examples_per_language.json',examples);dump(ROOT/'chunk_provenance.json',proofs)
 evidence={'source_config':'recipe/phimm/config/data/train_data/oss/no_en_v1.yaml','sources':sources,'train_rows':len(selected),'stats':stats,'comparison':summaries,'baseline':'v3.5.3-r2','target':'base2@560 v3.5.4-r2 initialization','limitations':['Model-version difference confounds causal interpretation.','Bad validation is diagnostic and selected on target errors, not an unbiased test.','Training candidates selected using reference features, not confirmed decoded errors.','Exact normalized test text excluded across all six locales; speaker disjointness not proven.'],'regression_used_in_validation':False,'train_sha256':sha(out.read_bytes())};dump(ROOT/'evidence.json',evidence)
 # Only one combined validation YAML and one training-data YAML.
 Path('recipe/phimm/config/data/val_data/fleurs_bad_verb_langhint.yaml').write_text(yaml.safe_dump(val,allow_unicode=True,sort_keys=False))
 train=[{'dataset_name':'jsonl','jsonl_paths':REMOTE+'/fleurs_pattern_train_v1.jsonl','add_task_info':{'task':'lang_asr_verb','prefix_prob':0.0,'lang_hint':True},'post_process':{'cache_audio':{},'add_field':{'fields':{'data_source':'openml'}},'shuffle':{'seed':44},'verl_format':{'prompt_key':'prompt'}}}]
 Path('recipe/phimm/config/data/train_data/fleurs_pattern_verb_langhint_v1.yaml').write_text(yaml.safe_dump(train,sort_keys=False))
 base=Path('recipe/phimm/config/v2611_dev/base2_mls_v1.yaml').read_text();base=base[base.index('defaults:'):].replace('mls','fleurs').replace('9880',str(len(selected)))
 base=base.replace('    es_fleurs_bad:', '    de_fleurs_bad: openasr_ml\n    es_fleurs_bad:')
 Path('recipe/phimm/config/v2611_dev/base2_fleurs_v1.yaml').write_text('# FLEURS train-only feature subset; bad validation only; regression retained for analysis.\n'+base)
 entry=Path('recipe/phimm/config/v2611_dev/dev/remax_dev_base2_mls_v1.yaml').read_text().replace('MLS','FLEURS').replace('mls','fleurs');Path('recipe/phimm/config/v2611_dev/dev/remax_dev_base2_fleurs_v1.yaml').write_text(entry)
 if a.publish:
  audios=sorted({r['audio_chunk'].rsplit(':',2)[0] for r in selected})
  with cf.ThreadPoolExecutor(32) as ex:assets=list(ex.map(lambda u:{'url':u,'bytes':bf.stat(u).size},audios))
  assert all(r['bytes']>9 for r in assets);dump(ROOT/'audio_assets.json',assets)
  artifacts += [ROOT/n for n in ['evidence.json','error_patterns.json','comparison_summary.json','two_examples_per_language.json','chunk_provenance.json','audio_assets.json']]
  def upload(p):
   url=REMOTE+'/'+str(p.relative_to(ROOT));bf.copy(str(p),url,overwrite=True)
   with bf.BlobFile(url,'rb') as f:assert sha(f.read())==sha(p.read_bytes())
   return {'url':url,'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
  with cf.ThreadPoolExecutor(12) as ex:publication=list(ex.map(upload,artifacts))
  dump(ROOT/'publication.json',publication)
 print(json.dumps(evidence,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

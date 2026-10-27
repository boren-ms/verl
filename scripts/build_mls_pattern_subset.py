#!/usr/bin/env python3
"""Select unchanged MLS training references by observed base2@560 error features.

Scans training transcription chunks only; does not decode or label model errors.
Caches each chunk to allow restart. Outputs reusable JSONL and selection audit.
"""
import argparse, concurrent.futures as cf, hashlib, io, json, re, time, unicodedata
from collections import Counter
from pathlib import Path
import blobfile as bf

LANGS = {'es':'Spanish','nl':'Dutch','it':'Italian','pt':'Portuguese','fr':'French'}
QUOTAS = {'es':3000,'nl':3000,'it':2000,'pt':2000,'fr':1000}
WORDS = {
 'es': set('a o fue más mas había habia usted adiós adios algún algun intención intencion mío mio señor menhir'.split()),
 'nl': set('zijn mijn hij zij wij jij een het vriendelijk vriendelijke eigenlijk vijf lijf'.split()),
 'it': set('più sì così perché né sé è articolo articoli armadio'.split()),
 'pt': set('coisa cousa dois dous novais novaes ocasião occasião reto recto colo collo tais taes beleza belleza emília'.split()),
 'fr': set('ces ses ce se sûr sur sûr dû où ou vingt trois onze'.split()),
}

NUMBER_RE = {
 'es': r'^(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieci\w*|veint\w*|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|cien\w*|doscientos|trescientos|cuatrocientos|quinientos|seiscientos|setecientos|ochocientos|novecientos|mil|millones?)$',
 'it': r'^(?:due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|tredici|quattordici|quindici|sedici|diciassette|diciotto|diciannove|venti|ventuno|ventidue|ventitré|ventiquattro|venticinque|ventisei|ventisette|ventotto|ventinove|trenta|quaranta|cinquanta|sessanta|settanta|ottanta|novanta|cento|duecento|trecento|quattrocento|cinquecento|seicento|settecento|ottocento|novecento|mille|duemila|tremila|mila|milioni?)$',
 'pt': r'^(?:dois|duas|três|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|catorze|quinze|dez\w*|vinte|trinta|quarenta|cinquenta|sessenta|setenta|oitenta|noventa|cem|cento|duzentos|trezentos|quatrocentos|quinhentos|seiscentos|setecentos|oitocentos|novecentos|mil|milhão|milhões)$',
 'nl': r'^(?:twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf|dertien|veertien|vijftien|zestien|zeventien|achttien|negentien|\w*(?:twintig|dertig|veertig|vijftig|zestig|zeventig|tachtig|negentig)|honderd|tweehonderd|driehonderd|duizend|tweeduizend|miljoen)$',
 'fr': r'^(?:deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|soixante|cent|cents|mille|million|millions)$',
}

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(t): return ' '.join(re.findall(r"[^\W_]+",unicodedata.normalize('NFC',t).casefold()))
def az(s): return re.sub(r'^/datablob1?/', 'az://orngwus2cresco/data/speech/',s)
def tokens(t): return re.findall(r"[^\W\d_]+|\d+",unicodedata.normalize("NFC",t).casefold())
def features(lang,text):
 ts=tokens(text); counts=Counter(ts); tags={}
 hits=sorted(set(ts)&WORDS[lang])
 # Count accents, but keep cedillas, ñ and ij distinctions intact in the dataset.
 accents=[t for t in ts if any(unicodedata.combining(c) for c in unicodedata.normalize('NFD',t))]
 if lang in ('es','it','pt') and accents: tags['accent_tokens']=len(accents)
 if hits: tags['observed_confusion_lexemes']=len(hits)
 if lang=='nl':
  n=sum('ij' in t for t in ts)
  if n: tags['dutch_ij']=n
  n=sum(counts[t] for t in ('zijn','mijn','hij','zij','wij','jij','een','het'))
  if n: tags['dutch_full_function_words']=n
 if lang=='pt':
  n=sum(counts[t] for t in WORDS['pt'])
  if n: tags['portuguese_spelling_variants']=n
 nums=[t for t in ts if t.isdigit()]
 number_words=[t for t in ts if re.fullmatch(NUMBER_RE[lang],t)]
 if len(number_words)>=2:tags['multiple_number_words']=len(number_words)
 if nums: tags['digit_sequences']=len(nums)
 if lang=='it' and any(t in counts for t in ('articolo','articoli')):tags['italian_article_context']=1
 # A targeted end-position feature, not a claim that arbitrary long rows omit words.
 tail=ts[-6:]
 tailhits=sorted(set(tail)&(WORDS[lang]-{'a','o','mas','een','het'}))
 if tailhits: tags['target_lexeme_in_last_6_tokens']=len(tailhits)
 score=(3*len(hits)+min(len(accents),8)+3*tags.get('dutch_ij',0)
        +min(tags.get('dutch_full_function_words',0),8)+5*len(nums)
        +5*tags.get('italian_article_context',0)+3*len(tailhits)+min(tags.get('multiple_number_words',0),6)*2)
 return ts,tags,hits,tailhits,score

def parse_chunk(raw,count):
 f=io.BytesIO(raw)
 assert f.read(13).lower()==b'transcription'
 assert len(f.read(4))==4
 rows=[]
 for i in range(count):
  assert int.from_bytes(f.read(4),'little')==i
  n=int.from_bytes(f.read(4),'little');b=f.read(n)
  assert len(b)==n
  rows.append(b.decode('utf-8'))
 assert not f.read(1), 'unexpected trailing bytes'
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='tmp/mls_pattern_train_v1');p.add_argument('--workers',type=int,default=32);p.add_argument('--quota-scale',type=float,default=1);a=p.parse_args()
 root=Path(a.root);cache=root/'transcription_cache';cache.mkdir(parents=True,exist_ok=True)
 (root/'sources').mkdir(exist_ok=True)
 # Resolve original training DataSpecs exactly as oss/no_en_v0.yaml does.
 def prepare_source(lang):
  spec=root/'sources'/f'open_mls_{lang}_asr.json'
  if not spec.exists():
   bf.copy(f'az://orngwus2cresco/data/speech/users/ruchaofan/DataSpecs/mlang_asr_data_2605/oss/open_mls_{lang}_asr.json',str(spec),overwrite=False)
  ds=json.loads(spec.read_text())['data_sources'];assert len(ds)==1
  man=root/'sources'/f'{lang}_file_set.json'
  if not man.exists():bf.copy(az(ds[0]['manifest_file']),str(man),overwrite=False)
 with cf.ThreadPoolExecutor(5) as ex:list(ex.map(prepare_source,LANGS))
 heldout=set();eval_files=[]
 for lang in LANGS:
  path=Path(f'tmp/asr-detail-compare/base2_best560_openall/{lang}_mls/target_matched.jsonl')
  assert path.exists(),path
  eval_files.append({'path':str(path),'sha256':sha(path.read_bytes())})
  for line in path.open():
   row=json.loads(line);v=row['gts'];
   for text in (v if isinstance(v,list) else [v]):
    if isinstance(text,str):heldout.add(norm(text))
 tasks=[];sources=[]
 for lang in LANGS:
  spec=root/'sources'/f'open_mls_{lang}_asr.json';man=root/'sources'/f'{lang}_file_set.json'
  ds=json.loads(spec.read_text())['data_sources'];assert len(ds)==1
  ds=ds[0];assert '/train/' in ds['manifest_file']
  sources.append({'language':lang,'spec':str(spec),'spec_sha256':sha(spec.read_bytes()),'manifest':str(man),'manifest_sha256':sha(man.read_bytes()),'source':ds})
  for chunk in json.loads(man.read_text())['fileInfo']:tasks.append((lang,ds,chunk))
 candidates={l:[] for l in LANGS};stats={l:Counter() for l in LANGS};failures=[];chunk_proof=[]
 def load(task):
  lang,ds,ch=task;url=az(ds.get('trans_path',ds['chunk_path']).rstrip('/')+'/'+ch['name']+'.transcription');path=cache/f'{lang}_{ch["name"]}.transcription'
  for retry in range(3):
   try:
    if not path.exists():bf.copy(url,str(path),overwrite=True)
    raw=path.read_bytes();rows=parse_chunk(raw,ch['count']);return task,url,sha(raw),rows
   except Exception:
    path.unlink(missing_ok=True)
    if retry==2:raise
    time.sleep(retry+1)
 print('Scanning',len(tasks),'transcription chunks',flush=True)
 with cf.ThreadPoolExecutor(a.workers) as ex:
  fs={ex.submit(load,t):t for t in tasks}
  for done,f in enumerate(cf.as_completed(fs),1):
   try:task,url,digest,rows=f.result()
   except Exception as e:failures.append({'task':fs[f],'error':repr(e)});continue
   lang,ds,ch=task;stats[lang]['scanned']+=len(rows)
   chunk_proof.append({'language':lang,'name':ch['name'],'count':ch['count'],'sha256':digest,'url':url})
   audio=az(ds.get('audio_path',ds['chunk_path']).rstrip('/')+'/'+ch['name']+'.audio')
   for idx,text in enumerate(rows):
    canonical=norm(text)
    if canonical in heldout:stats[lang]['heldout_text_excluded']+=1;continue
    ts,tags,hits,tailhits,score=features(lang,text)
    if not 5<=len(ts)<=160:stats[lang]['length_excluded']+=1;continue
    # Require target vocabulary, accent density, Dutch ij, digits or article context.
    eligible=(len(hits)>=2 or len([t for t in ts if any(unicodedata.combining(c) for c in unicodedata.normalize('NFD',t))])>=3 or tags.get('dutch_ij',0)>=2 or tags.get('digit_sequences',0)>0 or tags.get('italian_article_context',0)>0 or tags.get('multiple_number_words',0)>0)
    if not eligible:continue
    stats[lang]['eligible']+=1
    stable=f'mls_train_{lang}_{ch["name"]}_{idx}'
    candidates[lang].append({'id':stable,'audio_chunk':f'{audio}:{ch["count"]}:{idx}','text':text,'language':lang,'data_source':f'mls_pattern_{lang}','selection':{'tags':tags,'matched_lexemes':hits,'tail_lexemes':tailhits,'score':score,'text_sha256':sha(canonical.encode()),'source_spec':sources[list(LANGS).index(lang)]['spec'],'transcription_chunk':url,'chunk_index':idx,'source_split':'train','evidence':'reference_text_features_not_model_error','role':'french_control' if lang=='fr' else 'regression_target'}})
   if done%250==0:print('chunks',done,'/',len(tasks),'scanned',sum(s['scanned'] for s in stats.values()),flush=True)
 (root/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
 if failures:raise RuntimeError(f'{len(failures)} chunks failed; cached progress retained; rerun')
 selected=[];seen=set();tagcounts={l:Counter() for l in LANGS}
 for lang in LANGS:
  ranked=sorted(candidates[lang],key=lambda r:(-r['selection']['score'],sha(r['id'].encode())))
  # Cap each chunk at five rows to reduce concentration in single recording groups.
  chunkcounts=Counter()
  # Reserve coverage for article contexts and numeric expressions before general ranking.
  article=[r for r in ranked if r['selection']['tags'].get('italian_article_context')]
  numbers=[r for r in ranked if r['selection']['tags'].get('multiple_number_words') or r['selection']['tags'].get('digit_sequences')]
  priority=article[:50]+numbers[:round(QUOTAS[lang]*0.2)]+ranked
  for r in priority:
   h=r['selection']['text_sha256'];ch=r['selection']['transcription_chunk']
   if h in seen:stats[lang]['duplicate_candidates_skipped']+=1;continue
   if chunkcounts[ch]>=5:continue
   seen.add(h);chunkcounts[ch]+=1;selected.append(r);stats[lang]['selected']+=1;tagcounts[lang].update(r['selection']['tags'].keys())
   if stats[lang]['selected']>=round(QUOTAS[lang]*a.quota_scale):break
  assert stats[lang]['selected']>0
 selected.sort(key=lambda r:sha(r['id'].encode()))
 control=root/'mls_pattern_fr_control_v1.jsonl'
 with control.open('w') as f:
  for row in selected:
   if row['language']=='fr':f.write(json.dumps(row,ensure_ascii=False)+'\n')
 selected=[r for r in selected if r['language']!='fr']
 out=root/'mls_pattern_train_v1.jsonl'
 with out.open('w') as f:
  for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 audit={'dataset':'mls_pattern_train_v1','source_config':'recipe/phimm/config/data/train_data/oss/no_en_v0.yaml','sources':sources,'eval_exclusion_files':eval_files,'stats':stats,'selected_tag_counts':tagcounts,'quotas':QUOTAS,'selected':len(selected),'french_control_rows':stats['fr']['selected'],'french_control_sha256':sha(control.read_bytes()),'jsonl_sha256':sha(out.read_bytes()),'source_chunk_count':len(chunk_proof),'failed_chunks':0,'selection':'unchanged_reference_text_features; >=2 observed lexemes OR >=3 accented tokens OR >=2 ij tokens OR digits OR >=2 number words OR Italian articolo; reserve up to 50 article-context rows and 20% numeric-context candidates before general rank; 5..160 tokens; deterministic rank; <=5 per chunk; French control','limitations':['No model decoding or listening; selection is not confirmed model errors.','Train/test exact normalized text overlap excluded for all five analyzed MLS eval sets; speaker/book disjointness not established.','Priority ranks may favor long/dense utterances; no claimed improvement.','Audio references preserve original source; only sampled audio accessibility/loader checks are separate.']}
 (root/'evidence.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 (root/'chunk_provenance.json').write_text(json.dumps(chunk_proof,ensure_ascii=False))
 print(json.dumps({'selected':len(selected),'stats':stats},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

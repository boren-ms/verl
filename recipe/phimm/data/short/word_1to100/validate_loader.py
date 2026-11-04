#!/usr/bin/env python3
"""Load every final dataset and check longest real audio with the Base0 processor."""
import json,pathlib,os
import hf_qwen35_audio
import yaml
from omegaconf import OmegaConf
from verl.utils import hf_tokenizer,hf_processor
from recipe.phimm.data.rl_dataset import RLHFDataset,AUDIO_PAD_TOKEN_ID
r=pathlib.Path('/dev/shm/librispeech_word_1to100_20261104');repo=pathlib.Path('/root/code/verl');base=r/'base_processor_assets';tokenizer=hf_tokenizer(str(base),trust_remote_code=False);processor=hf_processor(str(base),trust_remote_code=False,use_fast=True);assert processor is not None
cfg=OmegaConf.create({'version':2609,'max_prompt_length':2048,'max_audio_dur':60,'truncation':'error','num_proc':1,'max_audio_load_retries':0,'use_interleave':False,'shuffle':False,'train_batch_size':64,'data_source':None})
entries=[('train',repo/'recipe/phimm/config/data/train_data/short/word_1to100/librispeech.yaml')]
for split in ('clean','other'):
 for lo in range(1,101,10):entries.append((f'test-{split}_{lo:03d}_{lo+9:03d}',repo/f'recipe/phimm/config/data/val_data/short/word_1to100/test_{split}_{lo:03d}_{lo+9:03d}.yaml'))
results={}
for name,path in entries:
 conf=yaml.safe_load(path.read_text());expected=9982 if name=='train' else 100
 # The verifier independently checked all metadata and WAVs; choose the true maximum-duration row for every group.
 records=[json.loads(line) for line in (r/'merged_a1'/name/'data.jsonl').open()];assert len(records)==expected;longest=max(records,key=lambda x:x['duration'])
 local_cfg=OmegaConf.create(OmegaConf.to_container(cfg));local_cfg.mix=[{'shuffle':True,'datasource':[{'name':'ls_word100_train','sample':9982,'shuffle':True}]}] if name=='train' else None
 ds=RLHFDataset([conf],tokenizer,local_cfg,processor,is_train=(name=='train'));assert len(ds)==expected
 ids=[row['id'] for row in ds.ds['extra_info']];idx=ids.index(longest['id']);row=ds[idx];wave,sr=row['multi_modal_data']['audio'][0];assert sr==16000 and len(wave)==round(longest['duration']*sr)
 n_pad=int((row['input_ids']==AUDIO_PAD_TOKEN_ID).sum());embed=int(row['multi_modal_inputs']['audio_embed_sizes'].sum());prompt_tokens=int(row['attention_mask'].sum());assert n_pad==embed>0 and prompt_tokens<=2048 and longest['duration']<=60
 assert row['reward_model']['ground_truth']==longest['text'];results[name]={'rows':len(ds),'longest_id':longest['id'],'longest_seconds':longest['duration'],'audio_pad_tokens':n_pad,'prompt_tokens':prompt_tokens,'ground_truth_matches':True,'audio_frames':len(wave),'data_source':row['data_source']}
 print(json.dumps({name:results[name]}),flush=True)
out=r/'loader_validation_a1.json';assert not out.exists();out.write_text(json.dumps({'model':'fast-llm-2609-qwen9b-flashenc-mtp-s2-data-v3.5.4-r2/54000/qwen_hf','transformers':'5.7.0','max_audio_dur':60,'max_prompt_length':2048,'truncation':'error','subsets':results},indent=2)+'\n');print(json.dumps({'complete':len(results),'report':str(out)}),flush=True)

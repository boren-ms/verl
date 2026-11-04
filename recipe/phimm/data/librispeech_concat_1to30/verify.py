#!/usr/bin/env python3
"""Verify every rendered WAV and manifest before publication."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
import soundfile as sf
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
completion=json.loads((root/'COMPLETE.json').read_text());result={};all_ids=set(); all_source_ids={};all_paths=set()
for folder in sorted(x for x in root.iterdir() if x.is_dir()):
 subset=folder.name; records=[]; histogram=collections.Counter(); sources=set(); segment_counts=collections.Counter(); max_error=0
 for line in (folder/'data.jsonl').open():
  x=json.loads(line);uid=x['id'];assert uid not in all_ids;all_ids.add(uid)
  local=folder/'audio'/f'{uid}.wav';remote=x['audio_path'];assert remote not in all_paths;all_paths.add(remote)
  assert local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest()==x['sha256']
  audio,sr=sf.read(local,dtype='float32');assert audio.ndim==1 and sr==16000 and len(audio)>0 and np.isfinite(audio).all() and np.any(audio!=0)
  duration=len(audio)/sr;assert 0<duration<30 and abs(duration-x['duration'])<1e-8 and abs(duration-x['target_duration'])<=.5001
  segs=x['source_segments'];assert 1<=len(segs)<=2 and len({s['id'] for s in segs})==len(segs)
  assert x['text']==' '.join(s['text'] for s in segs) and x['text'].strip()
  for s in segs:
   assert s['sample_rate']==sr and 0<=s['start_sample']<s['end_sample']<=s['source_frames'];sources.add(s['id'].removesuffix('.flac'))
   assert s['source_split'] in (('train-clean-100','train-clean-360','train-other-500') if subset=='train' else (subset.rsplit('_',1)[0],))
  assert len(audio)==sum(s['end_sample']-s['start_sample'] for s in segs)
  histogram[str(math.floor(duration))]+=1;segment_counts[str(len(segs))]+=1;max_error=max(max_error,abs(duration-x['target_duration']));records.append(x)
 assert len(records)==completion['counts'][subset]['written']
 audiofiles=list((folder/'audio').glob('*.wav'));assert len(audiofiles)==len(records)
 all_source_ids[subset]=sources
 result[subset]={'count':len(records),'histogram_1s':dict(sorted(histogram.items(),key=lambda z:int(z[0]))),'segment_counts':dict(segment_counts),'max_abs_target_error':max_error,'distinct_sources':len(sources),'manifest_sha256':hashlib.sha256((folder/'data.jsonl').read_bytes()).hexdigest(),'min_duration':min(x['duration'] for x in records),'max_duration':max(x['duration'] for x in records)}
for subset,ids in all_source_ids.items():
 if subset!='train': assert not (ids&all_source_ids['train']),f'split leak in {subset}'
report={'subsets':result,'total':len(all_ids),'train_test_source_overlap':0,'checks':'all WAV hashes, decode, finite nonzero mono 16kHz, frame count, duration, target error, segment count, text, source split and source ID disjointness'}
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

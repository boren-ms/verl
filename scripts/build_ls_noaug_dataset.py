#!/usr/bin/env python3
"""Publish the exact retained LS cohort referencing original, unmodified FLAC."""
import argparse,concurrent.futures,hashlib,io,json,math
from pathlib import Path
import blobfile as bf
import numpy as np
import soundfile as sf

def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--work-dir',required=True);p.add_argument('--workers',type=int,default=48);p.add_argument('--max-examples',type=int);a=p.parse_args();work=Path(a.work_dir);(work/'rows').mkdir(parents=True,exist_ok=True);out=a.output_dir.rstrip('/')
 with bf.BlobFile(a.source_manifest,'rb') as f:raw=f.read()
 assert digest(raw)=='14e3ea1f46e5523de9d08c4d3de6bae5c28961acae5bc461aae8c50a31f89b81';rows=[json.loads(l) for l in raw.splitlines() if l.strip()];assert len(rows)==20424 and len({r['source_audio_path'] for r in rows})==20424
 fingerprint=digest(raw+Path(__file__).read_bytes()+out.encode());(work/'original_manifest.jsonl').write_bytes(raw)
 stale=['response','raw_response','wer','edge_wer','speed_factor','noise_path','snr_db','rir_path']
 def audit(pair):
  i,row=pair;marker=work/'rows'/f'{i:08d}.json'
  if marker.exists():
   v=json.loads(marker.read_text());assert v['build_fingerprint']==fingerprint;return v
  path=row['source_audio_path'];assert '/LibriSpeech/' in path and path.endswith('.flac')
  with bf.BlobFile(path,'rb') as f:data=f.read()
  audio,sr=sf.read(io.BytesIO(data),dtype='float32');assert audio.size and np.isfinite(audio).all() and np.any(audio)
  duration=len(audio)/sr;assert duration<=40,{'index':i,'raw_duration':duration,'reason':'Would exceed unchanged loader40s limit; do not filter'}
  v={k:x for k,x in row.items() if k not in stale};v.update(audio_path=path,audio_chunk=None,parent_cached_audio_path=row['audio_path'],parent_augmentation={k:row[k] for k in stale if k in row},speed_factor=1.0,noise_path=None,rir_path=None,snr_db=None,augmentation_order=[],audio_sha256=digest(data),audio_bytes=len(data),audio_format='FLAC',sample_rate=sr,audio_num_samples=len(audio),duration=duration,build_fingerprint=fingerprint)
  assert v['text']==row['text'] and v['id']==row['id'];marker.write_text(json.dumps(v,ensure_ascii=False));return v
 selected=rows[:a.max_examples] if a.max_examples else rows
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:ordered=list(ex.map(audit,enumerate(selected)))
 manifest=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered).encode();(work/'data.jsonl').write_bytes(manifest)
 summary={'rows':len(ordered),'source_rows':20424,'source_manifest':a.source_manifest,'source_manifest_sha256':digest(raw),'output_manifest_sha256':digest(manifest),'build_fingerprint':fingerprint,'augmentation_order':[],'speed_factor':1.0,'noise_prob':0,'rir_prob':0,'audio_policy':'Original source FLAC bytes and paths; no waveform processing, reencoding or gain normalization','original_samples_and_order_preserved':True,'all_audio_decoded_finite':True,'hours':sum(r['duration'] for r in ordered)/3600,'max_duration_seconds':max(r['duration'] for r in ordered),'over40s':0,'total_audio_bytes':sum(r['audio_bytes'] for r in ordered),'smoke_only':bool(a.max_examples)}
 (work/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 if not a.max_examples:
  verification={'status':'PASSED','canonical_loader_rows':None,'rows':20424,'all_audio_decoded_finite':True,'labels_order_source_pointers_verified':True,'no_waveform_transformation':True,'manifest_sha256':digest(manifest)}
  (work/'verification.json').write_text(json.dumps(verification,indent=2))
  assert not bf.exists(out+'/data.jsonl'),'Immutable manifest already exists'
  for n in ['original_manifest.jsonl','summary.json','verification.json']:bf.copy(str(work/n),out+'/'+n)
  bf.copy(str(work/'data.jsonl'),out+'/data.jsonl')
if __name__=='__main__':main()

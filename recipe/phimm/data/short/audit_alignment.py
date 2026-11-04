#!/usr/bin/env python3
"""Publish conservative train boundary filtering and targeted review samples."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import blobfile as bf
import soundfile as sf

WORK=Path('/root/data/short_librispeech_20261104')
OUT=WORK/'final'
REMOTE='az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1'

def sha(x): return hashlib.sha256(x).hexdigest()
def read(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]
def dump(path, rows):
    with open(path,'w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def copy(src,dst): subprocess.run(['bbb','cp',str(src),str(dst)],check=True)
def flags(r):
    w=r['word_timestamps']; z=[i for i,x in enumerate(w) if x['start']==x['end']]
    return {'any_zero_word_duration':bool(z),'first_word_zero_duration':0 in z,'last_word_zero_duration':len(w)-1 in z,'clip_under_160ms':r['source_end']-r['source_start'] < .159}
def main():
    sets={s:read(OUT/s/'data.jsonl') for s in ('train','test-clean','test-other')}
    audit={}
    for split, rows in sets.items():
        data=[flags(x) for x in rows]
        audit[split]={'rows':len(rows), **{k:sum(f[k] for f in data) for k in data[0]}}
    kept=[x for x in sets['train'] if not (flags(x)['any_zero_word_duration'] or flags(x)['clip_under_160ms'])]
    excluded=[x for x in sets['train'] if flags(x)['any_zero_word_duration'] or flags(x)['clip_under_160ms']]
    if not kept: raise ValueError('filtered training set is empty')
    audit['train']['excluded_zero_word_or_under_160ms']=len(excluded)
    audit['train']['filtered_rows']=len(kept)
    dump(OUT/'train'/'data_filtered.jsonl',kept);dump(OUT/'train'/'boundary_excluded.jsonl',excluded)
    for split in ('test-clean', 'test-other'):
        flagged = [{**r, 'alignment_warnings': flags(r)} for r in sets[split]]
        dump(OUT/split/'data_flagged.jsonl', flagged)
    def ranked(rows): return sorted(rows,key=lambda x: hashlib.sha256(('reviewv2:'+x['id']).encode()).digest())
    def choose(rows, n, flagged):
        yes=ranked([r for r in rows if flags(r)['any_zero_word_duration']]); no=ranked([r for r in rows if not flags(r)['any_zero_word_duration']]); return yes[:flagged]+no[:n-flagged]
    review={'train':choose(kept,10,0),'validation':choose(sets['test-clean'],5,2)+choose(sets['test-other'],5,2)}
    evidence=[]
    for group,rows in review.items():
        for r in rows:
            r['alignment_warnings']=flags(r)
            with bf.BlobFile(r['audio_path'],'rb') as f: blob=f.read()
            expected=r.get('clip_sha256')
            if expected is None: expected=sha((WORK/'output'/'test-clean'/'audio'/Path(r['audio_path']).name).read_bytes())
            if sha(blob)!=expected: raise ValueError(f"audio hash mismatch {r['id']}")
            info=sf.info(io.BytesIO(blob))
            if not info.frames: raise ValueError(f"empty clip {r['id']}")
            evidence.append({'id':r['id'],'split':r['split'],'sha256':expected,'sample_rate':info.samplerate,'frames':info.frames})
        dump(OUT/f'human_review_{group}_10.jsonl',rows)
    (OUT/'quality_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'review_readback_sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for name in ('data_filtered.jsonl','boundary_excluded.jsonl'): copy(OUT/'train'/name,f'{REMOTE}/train/{name}')
    for split in ('test-clean', 'test-other'):
        copy(OUT/split/'data_flagged.jsonl', f'{REMOTE}/{split}/data_flagged.jsonl')
        with bf.BlobFile(f'{REMOTE}/{split}/data_flagged.jsonl','rb') as f:
            if sha(f.read())!=sha((OUT/split/'data_flagged.jsonl').read_bytes()):
                raise ValueError(f'{split}: flagged manifest read-back hash mismatch')
    for name in ('human_review_train_10.jsonl','human_review_validation_10.jsonl','quality_audit.json','review_readback_sha256.json'): copy(OUT/name,f'{REMOTE}/{name}')
    with bf.BlobFile(f'{REMOTE}/train/data_filtered.jsonl','rb') as f:
        if sha(f.read())!=sha((OUT/'train'/'data_filtered.jsonl').read_bytes()): raise ValueError('filtered manifest read-back hash mismatch')
    print('AUDIT COMPLETE',json.dumps(audit),flush=True)
if __name__=='__main__':main()

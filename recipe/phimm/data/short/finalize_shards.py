#!/usr/bin/env python3
"""Merge successful remote shards and publish verified final manifests."""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import subprocess
import blobfile as bf
import soundfile as sf

WORK = Path('/root/data/short_librispeech_20261104')
OUT = WORK / 'final'
REMOTE = 'az://orngwus2cresco/data/boren/data/LibriSpeech/short_2to5_v1'
STAGE = 'az://orngwus2cresco/data/boren/data/verl/staging/short_librispeech_20261104/shards'


def sha(data): return hashlib.sha256(data).hexdigest()
def read_rows(path):
    with bf.BlobFile(str(path), 'r') as f: return [json.loads(x) for x in f if x.strip()]
def dump_rows(path, rows):
    with path.open('w') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
def copy(src, dst): subprocess.run(['bbb','cp',str(src),str(dst)], check=True)


def main():
    OUT.mkdir(exist_ok=False)
    results = {}
    rejections = {}
    clean = WORK / 'output' / 'test-clean'
    results['test-clean'] = read_rows(clean/'data.jsonl')
    rejections['test-clean'] = read_rows(clean/'rejected.jsonl')
    for split in ('train','test-other'):
        results[split] = []
        rejections[split] = []
        for shard in (0,1):
            base = f'{STAGE}/output/{shard}/{split}'
            rows = read_rows(f'{base}/data.jsonl')
            rejected = read_rows(f'{base}/rejected.jsonl')
            source_count = sum(1 for _ in open(WORK/'shards'/'input'/f'{split}-{shard}.jsonl'))
            if len(rows)+len(rejected) != source_count:
                raise ValueError(f'{split} shard {shard}: input/output count mismatch')
            results[split] += rows
            rejections[split] += rejected
    if len(results['train']) < 20424: raise ValueError('Not enough training rows for the configured sample count')
    all_ids = set()
    for split, rows in results.items():
        ids = [r['id'] for r in rows]
        if len(set(ids)) != len(ids): raise ValueError(f'{split}: duplicate IDs')
        if all_ids.intersection(ids): raise ValueError(f'{split}: cross-split duplicate IDs')
        all_ids.update(ids)
        remote_files = set(bf.glob(f'{REMOTE}/{split}/audio/*.flac'))
        expected = set(r['audio_path'] for r in rows)
        if remote_files != expected: raise ValueError(f'{split}: remote audio listing mismatch {len(remote_files)} vs {len(expected)}')
        print(split, 'remote_audio',len(remote_files), 'rejected',len(rejections[split]), flush=True)
    ranked = lambda rows: sorted(rows, key=lambda x: hashlib.sha256(('review42:'+x['id']).encode()).digest())
    reviews = {'train': ranked(results['train'])[:10], 'validation': ranked(results['test-clean'])[:5]+ranked(results['test-other'])[:5]}
    evidence = []
    for group, rows in reviews.items():
        for r in rows:
            with bf.BlobFile(r['audio_path'], 'rb') as f: data=f.read()
            if r.get('clip_sha256'):
                expected=r['clip_sha256']
            else:
                expected=sha((clean/'audio'/Path(r['audio_path']).name).read_bytes())
            if sha(data)!=expected: raise ValueError(f"read-back hash mismatch {r['id']}")
            info=sf.info(io.BytesIO(data))
            if not info.frames or abs(info.duration-(r['source_end']-r['source_start'])) > 1/info.samplerate+1e-5:
                raise ValueError(f"duration mismatch {r['id']}")
            evidence.append({'id':r['id'],'split':r['split'],'sha256':expected,'sample_rate':info.samplerate,'frames':info.frames})
        dump_rows(OUT/f'human_review_{group}_10.jsonl',rows)
    stats={}
    for split, rows in results.items():
        folder=OUT/split;folder.mkdir()
        dump_rows(folder/'data.jsonl', rows)
        dump_rows(folder/'rejected.jsonl', rejections[split])
        wc=Counter(str(len(r['text'].split())) for r in rows)
        stats[split]={'written':len(rows),'rejected':len(rejections[split]),'word_counts':dict(sorted(wc.items()))}
        (folder/'stats.json').write_text(json.dumps(stats[split],indent=2)+'\n')
        for name in ('data.jsonl','rejected.jsonl','stats.json'): copy(folder/name,f'{REMOTE}/{split}/{name}')
    (OUT/'review_readback_sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps({'model':'Qwen/Qwen3-ForcedAligner-0.6B','model_revision':'c7cbfc2048c462b0d63a45797104fc9db3ad62b7','model_weights_sha256':'47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7','qwen_asr':'0.0.6','transformers':'4.57.6','seed':42,'stats':stats},indent=2)+'\n')
    for name in ('human_review_train_10.jsonl','human_review_validation_10.jsonl','review_readback_sha256.json','summary.json'): copy(OUT/name,f'{REMOTE}/{name}')
    copy(WORK/'aligned'/'test-clean.jsonl', f'{REMOTE}/aligned/test-clean.jsonl')
    for split in ('train', 'test-other'):
        for shard in (0, 1):
            name = f'{split}-{shard}.jsonl'
            local = OUT / name
            copy(f'{STAGE}/aligned/{name}', local)
            copy(local, f'{REMOTE}/aligned/{name}')
    for split in results:
        local=(OUT/split/'data.jsonl').read_bytes()
        with bf.BlobFile(f'{REMOTE}/{split}/data.jsonl','rb') as f:
            if sha(local)!=sha(f.read()): raise ValueError(f'{split}: manifest read-back hash mismatch')
    print('REMOTE VERIFICATION PASSED', json.dumps(stats), flush=True)

if __name__ == '__main__': main()

#!/usr/bin/env python3
"""Crop a single aligned shard, in parallel, from its node-local audio cache."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import soundfile as sf
from recipe.phimm.data.short.build_librispeech_short import make_plan, rows_from
from verl.audio_cache import local_audio_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--split', required=True, choices=['train','test-clean','test-other'])
    p.add_argument('--output-dir', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    root = Path(args.output_dir) / args.split
    audio_dir = root / 'audio'
    if root.exists():
        raise FileExistsError(root)
    audio_dir.mkdir(parents=True)
    rows = list(rows_from(args.input))
    ids = [str(x['id']) for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate ids in shard')
    def crop(row):
        uid = str(row['id'])
        try:
            text, start, end, stamps = make_plan(row, args.seed)
        except (ValueError, KeyError, TypeError) as e:
            return None, {'id': uid, 'reason': str(e)}
        if '/' in uid or uid in ('.','..'):
            raise ValueError('unsafe sample ID')
        src = local_audio_source(row['audio_path'])
        if not Path(src).is_file():
            raise FileNotFoundError(f'alignment did not cache source: {src}')
        info = sf.info(src)
        a, b = int(start * info.samplerate), int(end * info.samplerate)
        if not 0 <= a < b <= info.frames:
            return None, {'id':uid, 'reason':'timestamp beyond source audio'}
        audio, sr = sf.read(src, start=a, stop=b)
        filename = uid if uid.endswith('.flac') else uid + '.flac'
        sf.write(audio_dir / filename, audio, sr, format='FLAC', subtype='PCM_16')
        sha256 = hashlib.sha256((audio_dir / filename).read_bytes()).hexdigest()
        return {'id': uid, 'text': text, 'audio_path': f"{args.remote_root.rstrip('/')}/{args.split}/audio/{filename}", 'source_audio_path': row['audio_path'], 'source_text': row['text'], 'source_start': start, 'source_end': end, 'word_timestamps': stamps, 'split': args.split, 'clip_sha256': sha256}, None
    stats = Counter(input=len(rows))
    with (root/'data.jsonl').open('w') as out, (root/'rejected.jsonl').open('w') as rejects, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record, error in pool.map(crop, rows):
            if error:
                stats['rejected'] += 1
                rejects.write(json.dumps(error, ensure_ascii=False)+'\n')
            else:
                stats['written'] += 1
                stats[f"words_{len(record['text'].split())}"] += 1
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
            if (stats['written'] + stats['rejected']) % 1000 == 0:
                print(args.split, dict(stats), flush=True)
    (root/'stats.json').write_text(json.dumps(dict(stats), indent=2)+'\n')
    print(args.split, dict(stats), flush=True)


if __name__ == '__main__':
    main()

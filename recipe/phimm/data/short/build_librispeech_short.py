#!/usr/bin/env python3
"""Build short LibriSpeech clips from Qwen3 forced-alignment JSONL/parquet.

Each input must provide id, text, audio_path and word_timestamps. Raw alignment
results and original sources should be kept separately for provenance.
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*", text.lower())


def make_plan(row, seed):
    words = row['text'].split()
    stamps = row['word_timestamps']
    if isinstance(stamps, str):
        stamps = json.loads(stamps)
    if not words or not stamps:
        raise ValueError('empty transcript or alignment')
    if tokens(' '.join(x['text'] for x in stamps)) != tokens(row['text']):
        raise ValueError('aligned words do not match transcript')
    # Only use a prefix whose whitespace token boundaries agree with alignment.
    count = 2 + int.from_bytes(hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()[:8], 'big') % 4
    count = min(count, len(words))
    prefix = ' '.join(words[:count])
    prefix_tokens = tokens(prefix)
    n = 0
    for i, stamp in enumerate(stamps):
        n += len(tokens(stamp['text']))
        if n >= len(prefix_tokens):
            break
    else:
        raise ValueError('prefix does not align')
    if tokens(' '.join(x['text'] for x in stamps[:i + 1])) != prefix_tokens:
        raise ValueError('prefix crosses an alignment token')
    selected = stamps[:i + 1]
    for stamp in selected:
        start, end = float(stamp['start']), float(stamp['end'])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('invalid timestamps')
    start, end = float(selected[0]['start']), float(selected[-1]['end'])
    if end <= start or any(float(s['start']) < start or float(s['end']) > end for s in selected):
        raise ValueError('invalid prefix boundaries')
    if i + 1 < len(stamps) and float(stamps[i + 1]['start']) < end:
        raise ValueError('next word overlaps crop')
    return prefix, start, end, selected


def rows_from(path):
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--split', required=True, choices=['train', 'test-clean', 'test-other'])
    p.add_argument('--output-dir', required=True)
    p.add_argument('--remote-root', required=True, help='Final blob root containing split/audio and split/data.jsonl')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle-input', action='store_true', help='Deterministically sample across the entire input before applying --limit')
    args = p.parse_args()
    from recipe.phimm.utils.audio import load_raw_audio
    import soundfile as sf
    out = Path(args.output_dir) / args.split
    audio_dir = out / 'audio'
    if (out / 'data.jsonl').exists():
        raise FileExistsError(f'{out}/data.jsonl already exists; choose a fresh output directory')
    audio_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    seen = set()
    audio_failures = 0
    rows = rows_from(args.input)
    if args.shuffle_input:
        rows = sorted(rows, key=lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).digest())
    with (out / 'data.jsonl').open('w', encoding='utf-8') as data, (out / 'rejected.jsonl').open('w', encoding='utf-8') as rejected:
        for row in rows:
            if args.limit is not None and stats['written'] >= args.limit:
                break
            stats['input'] += 1
            try:
                uid = str(row['id'])
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', uid) or uid in seen:
                    raise ValueError('unsafe or duplicate id')
                seen.add(uid)
                text, start, end, stamps = make_plan(row, args.seed)
                try:
                    audio, sr = load_raw_audio({'audio_path': row['audio_path']})
                except Exception as e:
                    audio_failures += 1
                    if audio_failures >= 3:
                        raise RuntimeError('three consecutive audio load failures; stop and inspect access') from e
                    raise ValueError(f'audio load failed: {e}') from e
                audio_failures = 0
                if sr <= 0 or end > len(audio) / sr + 1e-6:
                    raise ValueError('timestamps exceed source duration')
                a, b = int(start * sr), int(end * sr)
                if b <= a:
                    raise ValueError('empty audio crop')
                filename = uid if uid.endswith('.flac') else f'{uid}.flac'
                path = audio_dir / filename
                sf.write(path, audio[a:b], sr, format='FLAC', subtype='PCM_16')
                record = {'id': uid, 'text': text, 'audio_path': f"{args.remote_root.rstrip('/')}/{args.split}/audio/{filename}", 'source_audio_path': row['audio_path'], 'source_text': row['text'], 'source_start': start, 'source_end': end, 'word_timestamps': stamps, 'split': args.split}
                data.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['written'] += 1
                stats[f'words_{len(text.split())}'] += 1
            except RuntimeError:
                raise
            except Exception as e:
                stats['rejected'] += 1
                rejected.write(json.dumps({'id': row.get('id'), 'reason': str(e)}, ensure_ascii=False) + '\n')
    (out / 'stats.json').write_text(json.dumps(dict(stats), indent=2) + '\n')
    print(json.dumps(dict(stats), indent=2))


if __name__ == '__main__':
    main()

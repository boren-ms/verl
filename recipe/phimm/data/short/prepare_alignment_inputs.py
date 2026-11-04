#!/usr/bin/env python3
"""Prepare clean, disjoint LibriSpeech input manifests for fresh forced alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-parquet', required=True)
    p.add_argument('--val-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--train-candidates', type=int, default=21000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.train_parquet, columns=['id', 'text', 'audio_path']).to_pylist()
    rows.sort(key=lambda x: hashlib.sha256(f"{args.seed}:{x['id']}".encode()).digest())
    if args.train_candidates > len(rows):
        raise ValueError('not enough training candidates')
    splits = {'train': rows[:args.train_candidates]}
    for name in ('test-clean', 'test-other'):
        src = Path(args.val_dir) / (name + '.jsonl')
        parsed = []
        for line in src.open(encoding='utf-8'):
            x = json.loads(line)
            path = x['audio_path']
            prefix = '/root/data/LibriSpeech/' + name + '/'
            if not path.startswith(prefix):
                raise ValueError(f'unexpected {name} path: {path}')
            parsed.append({'id': x['id'], 'text': x['text'], 'audio_path': 'az://orngwus2cresco/data/boren/data/LibriSpeech/' + name + '/' + path[len(prefix):]})
        splits[name] = parsed
    for split, data in splits.items():
        if len(set(x['id'] for x in data)) != len(data):
            raise ValueError(f'duplicate ID in {split}')
        out = root / (split + '.jsonl')
        with out.open('x', encoding='utf-8') as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
        print(split, len(data), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Align a JSONL shard using a local or cached Qwen3 model."""
import argparse
import blobfile as bf
from recipe.phimm import align_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', help='local model directory; default is the repository model cache')
    p.add_argument('--n-gpus', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers-per-gpu', type=int, default=1)
    p.add_argument('--io-threads', type=int, default=3,
                   help='concurrent Azure audio prefetches per alignment worker')
    p.add_argument('--language', default='English')
    args = p.parse_args()
    if bf.exists(args.output):
        raise FileExistsError(f'output exists: {args.output}')
    ds = align_dataset.load_ds(args.input)
    ds = align_dataset.align_ds(
        ds, language=args.language, batch_size=args.batch_size, n_gpus=args.n_gpus,
        workers_per_gpu=args.workers_per_gpu, model_path=args.model, io_threads=args.io_threads,
    )
    align_dataset.save_ds(ds, args.output)
    errors = sum(1 for error in ds['alignment_error'] if error)
    if errors:
        raise RuntimeError(f'{errors}/{len(ds)} alignment failures; inspect alignment_error in {args.output}')


if __name__ == '__main__':
    main()

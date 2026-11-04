#!/usr/bin/env python3
"""Deterministically shard aligned manifests and create cropped audio subsets.

Input JSONL: {id, text, audio_path, word_timestamps}; timestamps are a list
(or JSON-encoded list) of {text, start, end} in seconds. Original inputs remain
untouched. This tool does not upload data or run an aligner.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_int(value):
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8], 'big')


def write_json_atomic(path, value):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{n}: invalid JSON') from e
                if not isinstance(obj, dict):
                    raise ValueError(f'{path}:{n}: expected an object')
                yield obj


def row_id(row):
    value = row.get('id')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('each row requires a nonempty string id')
    return value


def token_list(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def plan_prefix(row, minimum=2, maximum=5, seed=42, start_word=0):
    if not 1 <= minimum <= maximum:
        raise ValueError('word range must satisfy 1 <= min <= max')
    uid = row_id(row)
    if row.get('alignment_error'):
        raise ValueError(f"alignment failed: {row['alignment_error']}")
    text = row.get('text')
    if not isinstance(text, str) or not text.split():
        raise ValueError('empty transcript')
    words = text.split()
    if start_word < 0 or start_word >= len(words):
        raise ValueError('start word outside transcript')
    if not all(token_list(word) for word in words):
        raise ValueError('transcript has punctuation-only token')
    stamps = row.get('word_timestamps')
    if isinstance(stamps, str):
        stamps = json.loads(stamps)
    if not isinstance(stamps, list) or not stamps:
        raise ValueError('empty alignment')
    normalized = []
    for w in stamps:
        if not isinstance(w, dict) or not isinstance(w.get('text'), str):
            raise ValueError('invalid alignment entry')
        start = float(w.get('start', w.get('start_time')))
        end = float(w.get('end', w.get('end_time')))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('invalid timestamp')
        normalized.append({'text': w['text'], 'start': start, 'end': end})
    if any(b['start'] < a['start'] or b['end'] < a['end'] for a, b in zip(normalized, normalized[1:])):
        raise ValueError('non-monotonic alignment')
    aligned_tokens = []
    boundaries = {0: 0}
    for i, stamp in enumerate(normalized, 1):
        unit_tokens = token_list(stamp['text'])
        if not unit_tokens:
            raise ValueError('alignment unit has no word tokens')
        aligned_tokens.extend(unit_tokens)
        boundaries[len(aligned_tokens)] = i
    if aligned_tokens != token_list(text):
        raise ValueError('aligned text differs from transcript')
    target = minimum + stable_int(f'{seed}:{uid}') % (maximum - minimum + 1)
    prefix = ' '.join(words[start_word:min(start_word + target, len(words))])
    before = token_list(' '.join(words[:start_word]))
    through = token_list(' '.join(words[:min(start_word + target, len(words))]))
    left = boundaries.get(len(before))
    if left is None or aligned_tokens[:len(before)] != before:
        raise ValueError('window start crosses an alignment unit')
    right = boundaries.get(len(through))
    if right is None or aligned_tokens[:len(through)] != through or right == left:
        raise ValueError('window end crosses an alignment unit')
    selected = normalized[left:right]
    start, end = selected[0]['start'], selected[-1]['end']
    if end <= start or any(w['start'] < start or w['end'] > end for w in selected):
        raise ValueError('invalid prefix bounds')
    if left and normalized[left - 1]['end'] > start:
        raise ValueError('previous word overlaps crop')
    if right < len(normalized) and normalized[right]['start'] < end:
        raise ValueError('next word overlaps crop')
    return prefix, start, end, selected


def shard(args):
    if args.count < 1:
        raise ValueError('shard count must be positive')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    files = [(out/f'shard-{i:04d}.jsonl').open('x', encoding='utf-8') for i in range(args.count)]
    counts = [0]*args.count
    seen = set()
    try:
        for row in read_jsonl(args.input):
            uid = row_id(row)
            if uid in seen:
                raise ValueError(f'duplicate id: {uid}')
            seen.add(uid)
            i = stable_int(f'{args.seed}:{uid}') % args.count
            files[i].write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[i] += 1
    finally:
        for f in files: f.close()
    metadata = {'input_sha256':digest_file(args.input), 'seed':args.seed, 'counts':counts, 'total':sum(counts)}
    write_json_atomic(out/'COMPLETE.json', metadata)
    print(json.dumps(metadata))


def ordered_bounded(pool, func, rows, limit):
    pending = deque()
    for row in rows:
        pending.append(pool.submit(func, row))
        if len(pending) >= limit:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def crop(args):
    import soundfile as sf
    source = Path(args.input).resolve()
    if args.workers < 1 or args.queue_factor < 1:
        raise ValueError('workers and queue factor must be positive')
    output = Path(args.output_dir).resolve()
    options = {'input_sha256':digest_file(source), 'min_words':args.min_words,
               'max_words':args.max_words, 'seed':args.seed, 'zero_duration':args.zero_duration,
               'audio_base':args.audio_base, 'audio_format':args.audio_format,
               'min_clip_seconds':args.min_clip_seconds, 'start_word':args.start_word,
               'repo_root':str(Path(args.repo_root).resolve()) if args.repo_root else None}
    meta_path, done_path = output/'run.json', output/'COMPLETE.json'
    if output.exists():
        if not args.resume or done_path.exists() or not meta_path.is_file():
            raise FileExistsError('output exists; use --resume only for an incomplete run')
        if json.loads(meta_path.read_text()) != options:
            raise ValueError('resume configuration or input checksum differs')
    else:
        output.mkdir(parents=True)
        write_json_atomic(meta_path, options)
    audio_dir = output/'audio'; audio_dir.mkdir(exist_ok=True)
    data_path, rejected_path = output/'data.jsonl', output/'rejected.jsonl'
    completed = set()
    for path in (data_path, rejected_path):
        if path.exists():
            for row in read_jsonl(path):
                uid = row_id(row)
                if uid in completed: raise ValueError(f'duplicate completed id: {uid}')
                if path == data_path:
                    clip = audio_dir/row['clip_filename']
                    if not clip.is_file() or digest_file(clip) != row['clip_sha256']:
                        raise ValueError(f'missing or changed completed clip: {uid}')
                completed.add(uid)
    if args.repo_root:
        sys.path.insert(0, str(Path(args.repo_root).resolve()))
    def work(row):
        uid = row_id(row)
        try:
            prefix, start, end, stamps = plan_prefix(row, args.min_words, args.max_words, args.seed, args.start_word)
        except (ValueError, KeyError, TypeError) as exc:
            return {'id':uid, 'reason':str(exc)}, False
        warnings = {'zero_duration_word':any(w['start']==w['end'] for w in stamps),
                    'clip_under_160ms':end-start < 0.159}
        if end-start + 1e-9 < args.min_clip_seconds:
            return {'id':uid, 'reason':'clip shorter than requested minimum'}, False
        if warnings['zero_duration_word'] and args.zero_duration == 'exclude':
            return {'id':uid, 'reason':'zero-duration selected word'}, False
        if warnings['zero_duration_word'] and args.zero_duration == 'error':
            raise ValueError(f'{uid}: zero-duration selected word')
        audio_source = row.get('audio_path')
        if not isinstance(audio_source, str) or not audio_source:
            raise ValueError(f'{uid}: missing audio_path')
        if '://' in audio_source:
            if not args.repo_root or not audio_source.startswith('az://'):
                raise ValueError(f'{uid}: remote audio requires --repo-root and an az:// path')
            from verl.audio_cache import localize_audio_source
            audio_source = localize_audio_source(audio_source)
        info = sf.info(audio_source)
        a, b = int(start*info.samplerate), int(end*info.samplerate)
        if not 0 <= a < b <= info.frames:
            return {'id':uid,'reason':'timestamp outside source audio'}, False
        audio, sr = sf.read(audio_source, start=a, stop=b)
        filename = hashlib.sha256(uid.encode()).hexdigest() + '.' + args.audio_format.lower()
        dest = audio_dir/filename
        temp = audio_dir/(filename+'.'+uuid.uuid4().hex+'.tmp')
        try:
            sf.write(temp, audio, sr, format=args.audio_format.upper(), subtype='PCM_16')
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
        path = f"{args.audio_base.rstrip('/')}/{filename}" if args.audio_base else str(dest)
        return {'id':uid,'text':prefix,'audio_path':path,'clip_filename':filename,
                'clip_sha256':digest_file(dest),'source_audio_path':row['audio_path'],
                'source_text':row['text'],'source_start':start,'source_end':end,
                'word_timestamps':stamps,'alignment_warnings':warnings}, True
    seen = set()
    def pending_rows():
        for row in read_jsonl(source):
            uid = row_id(row)
            if uid in seen: raise ValueError(f'duplicate input id: {uid}')
            seen.add(uid)
            if uid not in completed: yield row
    counts = Counter()
    with data_path.open('a', encoding='utf-8') as data, rejected_path.open('a', encoding='utf-8') as rejected, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result, ok in ordered_bounded(pool, work, pending_rows(), args.workers*args.queue_factor):
            f = data if ok else rejected
            f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush()
            counts['written' if ok else 'rejected'] += 1
            if sum(counts.values()) % args.log_every == 0:
                print(json.dumps(dict(counts)), flush=True)
    counts['written_total'] = sum(1 for _ in read_jsonl(data_path))
    counts['rejected_total'] = sum(1 for _ in read_jsonl(rejected_path))
    if not seen:
        raise ValueError('input is empty')
    if counts['written_total']+counts['rejected_total'] != len(seen):
        raise ValueError('input and output counts differ')
    words = Counter()
    warnings = Counter()
    for row in read_jsonl(data_path):
        words[str(len(row['text'].split()))] += 1
        for key, value in row['alignment_warnings'].items():
            warnings[key] += bool(value)
    summary = {**counts, 'word_counts':dict(sorted(words.items())), 'warnings':dict(warnings)}
    write_json_atomic(output/'COMPLETE.json', summary)
    print(json.dumps(summary))


def merge(args):
    """Merge completed shard manifests; audio remains at the referenced paths."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    seen = set()
    counts = Counter()
    settings = None
    with (out/'data.jsonl').open('x',encoding='utf-8') as data, (out/'rejected.jsonl').open('x',encoding='utf-8') as rejected:
        for folder in map(Path, args.inputs):
            if not (folder/'COMPLETE.json').is_file():
                raise ValueError(f'incomplete shard: {folder}')
            run_settings = json.loads((folder/'run.json').read_text())
            run_settings.pop('input_sha256')
            if settings is None: settings = run_settings
            if settings != run_settings: raise ValueError(f'incompatible shard settings: {folder}')
            summary = json.loads((folder/'COMPLETE.json').read_text())
            for name, dst in (('data.jsonl', data), ('rejected.jsonl', rejected)):
                subtotal = 0
                for row in read_jsonl(folder/name):
                    uid = row_id(row)
                    if uid in seen: raise ValueError(f'duplicate id across shards: {uid}')
                    seen.add(uid)
                    dst.write(json.dumps(row,ensure_ascii=False)+'\n')
                    counts['written' if name == 'data.jsonl' else 'rejected'] += 1
                    subtotal += 1
                expected_key = 'written_total' if name == 'data.jsonl' else 'rejected_total'
                if subtotal != summary.get(expected_key): raise ValueError(f'shard count mismatch: {folder}/{name}')
    write_json_atomic(out/'COMPLETE.json', dict(counts))
    print(json.dumps(dict(counts)))


def review(args):
    if args.count < 1: raise ValueError('review count must be positive')
    rows = list(read_jsonl(args.input))
    if len(rows) < args.count: raise ValueError('not enough rows for review')
    selected = sorted(rows,key=lambda r:stable_int(f'{args.seed}:{row_id(r)}'))[:args.count]
    with open(args.output,'x',encoding='utf-8') as f:
        for row in selected: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'count':len(selected),'output':args.output}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('shard',help='split JSONL deterministically by id')
    s.add_argument('--input',required=True);s.add_argument('--output-dir',required=True)
    s.add_argument('--count',type=int,required=True);s.add_argument('--seed',type=int,default=42);s.set_defaults(func=shard)
    c=sub.add_parser('crop',help='crop a word window using aligned timestamps')
    c.add_argument('--input',required=True);c.add_argument('--output-dir',required=True)
    c.add_argument('--audio-base',help='final remote audio directory; publish audio separately')
    c.add_argument('--repo-root',help='verl checkout for az:// audio cache')
    c.add_argument('--min-words',type=int,default=2);c.add_argument('--max-words',type=int,default=5)
    c.add_argument('--start-word',type=int,default=0,help='zero-based first word in each selected window')
    c.add_argument('--seed',type=int,default=42);c.add_argument('--workers',type=int,default=8)
    c.add_argument('--queue-factor',type=int,default=4);c.add_argument('--log-every',type=int,default=1000)
    c.add_argument('--audio-format',choices=['flac','wav'],default='flac')
    c.add_argument('--min-clip-seconds',type=float,default=0.0,help='exclude clips below this duration')
    c.add_argument('--zero-duration',choices=['flag','exclude','error'],default='flag')
    c.add_argument('--resume',action='store_true');c.set_defaults(func=crop)
    m=sub.add_parser('merge',help='merge completed shard manifests without copying audio')
    m.add_argument('--inputs',nargs='+',required=True);m.add_argument('--output-dir',required=True);m.set_defaults(func=merge)
    r=sub.add_parser('review',help='sample an existing subset deterministically')
    r.add_argument('--input',required=True);r.add_argument('--output',required=True)
    r.add_argument('--count',type=int,default=10);r.add_argument('--seed',type=int,default=42);r.set_defaults(func=review)
    args=p.parse_args()
    if getattr(args,'log_every',1)<1: p.error('--log-every must be positive')
    if args.command == 'crop' and not 1 <= args.min_words <= args.max_words: p.error('word range must satisfy 1 <= min <= max')
    if args.command == 'crop' and args.start_word < 0: p.error('--start-word must be nonnegative')
    if not math.isfinite(getattr(args,'min_clip_seconds',0)) or getattr(args,'min_clip_seconds',0)<0: p.error('--min-clip-seconds must be finite and nonnegative')
    args.func(args)
if __name__=='__main__': main()

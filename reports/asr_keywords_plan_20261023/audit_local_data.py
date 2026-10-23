"""Read-only inventory for planning; no model inference or training-data mutation.

Run from any directory:
  python audit_local_data.py --data-root /home/boren/data --output audit.json
Requires soundfile only for optional audio-header verification.
"""
import argparse
import ast
import collections
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def words(text):
    return re.findall(r"[\w'\-]+", text.lower())


def split_for(speaker):
    bucket = int(hashlib.sha256(f'ls-keywords-v1:{speaker}'.encode()).hexdigest(), 16) % 100
    return 'dev' if bucket < 5 else 'internal_test' if bucket < 10 else 'train'


def audit(data_root):
    word_dir = data_root / 'librispeech_biasing/words'
    common5 = {line.split()[0].lower() for line in (word_dir / 'common_words_5k.txt').open() if line.strip()}
    rare = {line.split()[0].lower() for line in (word_dir / 'all_rare_words.txt').open() if line.strip()}
    out = {'scope': 'local raw TSV + local original biasing JSONL; cloud keyword JSONL not read',
           'split_rule': 'speaker SHA256(ls-keywords-v1:<speaker>) mod 100: 0..4 dev, 5..9 internal_test, else train',
           'word_lists': {'common5k_size': len(common5), 'rare_size': len(rare),
                          'common5k_sha256':digest(word_dir / 'common_words_5k.txt'),
                          'rare_list_sha256':digest(word_dir / 'all_rare_words.txt')}, 'evaluation': {}}
    test_ids, test_texts, test_speakers = set(), set(), set()
    for name in ('test-clean', 'test-other'):
        path = data_root / 'librispeech_biasing/ref' / f'{name}.biasing_100.jsonl'
        rows = [json.loads(line) for line in path.open() if line.strip()]
        c = collections.Counter(); groups = collections.defaultdict(list)
        examples = []; seconds = 0.; bad_audio = []; audio_available = 0
        try:
            import soundfile as sf
        except ImportError:
            sf = None
        for row in rows:
            sid, chapter, index = row['id'].split('-')
            keys = set(row['ground_truth']); ws = set(words(row['text']))
            test_ids.add(row['id']); test_texts.add(' '.join(words(row['text']))); test_speakers.add(sid)
            groups[(sid, chapter)].append((int(index), keys))
            c['nonempty'] += bool(keys); c['empty'] += not keys; c['keyword_items'] += len(keys)
            c['keywords_not_in_text'] += len(keys - ws)
            c['rows_differ_from_text_minus_common5k'] += keys != (ws - common5)
            c['rows_differ_from_text_intersect_rare_list'] += keys != (ws & rare)
            p = data_root / 'LibriSpeech' / name / sid / chapter / f"{row['id']}.flac"
            audio_available += p.is_file()
            if sf is not None and p.is_file():
                try:
                    info = sf.info(str(p)); seconds += info.duration
                    c[f'sample_rate_{info.samplerate}'] += 1
                    c[f'channels_{info.channels}'] += 1
                except (OSError, RuntimeError) as e:
                    bad_audio.append({'id': row['id'], 'error': str(e)})
            if keys and len(examples) < 2:
                examples.append({'id':row['id'], 'text':row['text'], 'keywords':row['ground_truth'], 'audio_path':str(p)})
        overlap = collections.Counter()
        for group in groups.values():
            ordered = sorted(group)
            for i in range(1, len(ordered)):
                prev_i, prev = ordered[i-1]; curr_i, curr = ordered[i]
                overlap['ordered_pairs'] += 1
                overlap['consecutive_number_pairs'] += curr_i == prev_i + 1
                overlap['pairs_with_repeated_keyword'] += bool(prev & curr)
                overlap['current_keyword_items_after_first'] += len(curr)
                overlap['repeated_items_from_previous'] += len(prev & curr)
                history = set().union(*(keys for _,keys in ordered[max(0,i-3):i]))
                overlap['repeated_items_from_previous_three'] += len(history & curr)
        out['evaluation'][name] = {'path':str(path), 'sha256':digest(path), 'rows':len(rows), **dict(c),
            'chapters':len(groups), 'speakers':len({x[0] for x in groups}), 'audio_available':audio_available,
            'audio_header_errors':bad_audio, 'audio_hours':seconds/3600 if sf else None,
            'ordered_chapter_diagnostics':dict(overlap), 'examples':examples}
    train_path = data_root / 'LibriSpeech/asr_train_transcribe.tsv'
    count = collections.Counter(); split_stats = collections.defaultdict(collections.Counter)
    speakers = collections.defaultdict(set); chapters = set(); ids = set(); probes = collections.defaultdict(list)
    for line in train_path.open():
        if not line.strip():
            continue
        name, audio_list, conversations = line.rstrip('\n').split('\t',2)
        uid = Path(name).stem; sid, chapter, _ = uid.split('-'); fold = split_for(sid)
        msgs = ast.literal_eval(conversations)
        texts = [m['content'] for conv in msgs for m in conv['messages'] if m['role'] == 'assistant']
        if len(texts) != 1:
            raise ValueError(f'{uid}: expected one assistant reference, got {len(texts)}')
        text = texts[0]; ws = set(words(text)); inferred = ws & rare; target5k = ws - common5
        count['rows'] += 1; count['duplicate_ids'] += uid in ids; ids.add(uid); chapters.add((sid,chapter))
        count['test_id_overlap'] += uid in test_ids
        count['test_normalized_text_overlap_rows'] += ' '.join(words(text)) in test_texts
        count['test_speaker_overlap_rows'] += sid in test_speakers
        count['nonempty_under_explicit_rare_list'] += bool(inferred)
        count['empty_under_explicit_rare_list'] += not inferred
        count['nonempty_under_common5k_exclusion'] += bool(target5k)
        count['empty_under_common5k_exclusion'] += not target5k
        split_stats[fold]['rows'] += 1; split_stats[fold]['nonempty_under_explicit_rare_list'] += bool(inferred)
        split_stats[fold]['empty_under_explicit_rare_list'] += not inferred
        split_stats[fold]['nonempty_under_common5k_exclusion'] += bool(target5k)
        split_stats[fold]['empty_under_common5k_exclusion'] += not target5k
        split_stats[fold]['test_normalized_text_overlap_rows'] += ' '.join(words(text)) in test_texts
        speakers[fold].add(sid)
        ap = ast.literal_eval(audio_list)
        if len(ap) != 1:
            raise ValueError(f'{uid}: expected one audio path')
        local = data_root / ap[0][len('/root/data/'):] if ap[0].startswith('/root/data/') else Path(ap[0])
        count['local_audio_available'] += local.is_file()
        split_stats[fold]['local_audio_available'] += local.is_file()
        if len(probes[fold]) < 3:
            probes[fold].append({'id':uid,'text':text,'derived_common5k_keywords_example':sorted(target5k),'audio_path':str(local)})
    out['training_tsv'] = {'path':str(train_path), 'sha256':digest(train_path), **dict(count),
        'chapters':len(chapters), 'splits':{k:{**dict(v),'speakers':len(speakers[k])} for k,v in split_stats.items()},
        'examples':dict(probes), 'note':'Derived-label counts do not verify the cloud keyword manifest. common5k exclusion matches all local test labels; explicit rare-list intersection does not.'}
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path('/home/boren/data'))
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=audit(a.data_root); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'training_tsv':{k:v for k,v in result['training_tsv'].items() if k!='examples'},
                      'evaluation':{k:{x:y for x,y in v.items() if x!='examples'} for k,v in result['evaluation'].items()}},ensure_ascii=False))

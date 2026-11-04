#!/usr/bin/env python3
"""Create deterministic, auditable LibriSpeech word-window plans; no audio is written."""
import argparse, collections, hashlib, json, math, random, re
from pathlib import Path

FILES={'train':['train-0','train-1'], 'test-clean':['test-clean'], 'test-other':['test-other-0','test-other-1']}
ALLOWED={'train':{'train-clean-100','train-clean-360','train-other-500'},'test-clean':{'test-clean'},'test-other':{'test-other'}}
TOKEN=re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")
def tokens(s): return TOKEN.findall(s.lower())
def load(root, split):
    pool=[]; seen=set(); rejected=collections.Counter(); hashes={}
    for name in FILES[split]:
        path=root/(name+'.jsonl'); hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            x=json.loads(line); ident=x['id'].removesuffix('.flac'); source=x['audio_path']; part=source.split('/LibriSpeech/')[1].split('/')[0]
            if ident in seen or part not in ALLOWED[split]: raise ValueError(f'duplicate or wrong split: {ident} {part}')
            seen.add(ident)
            try:
                words=x['word_timestamps']; words=json.loads(words) if isinstance(words,str) else words
                if not words or any(len(tokens(w['text']))!=1 for w in words) or tokens(' '.join(w['text'] for w in words))!=tokens(x['text']): raise ValueError('text_or_unit_mismatch')
                prev=0.0
                for w in words:
                    start,end=float(w['start']),float(w['end'])
                    if not math.isfinite(start) or not math.isfinite(end) or start<prev or end<=start: raise ValueError('invalid_zero_or_overlap')
                    prev=end
                pool.append({'id':ident,'audio_path':source,'source_split':part,'source_text':x['text'],'words':words})
            except (ValueError,TypeError,KeyError) as e: rejected[str(e)]+=1
    return pool,seen,dict(rejected),hashes

def segment(source, remaining, rng):
    words=source['words']; n=min(remaining,len(words)); start=rng.randrange(len(words)-n+1); selected=words[start:start+n]
    return {'id':source['id'],'audio_path':source['audio_path'],'source_split':source['source_split'],'source_text':source['source_text'],'start_word_index':start,'end_word_index_exclusive':start+n,'start':float(selected[0]['start']),'end':float(selected[-1]['end']),'text':' '.join(w['text'] for w in selected),'word_timestamps':selected}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--aligned-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=20261104);p.add_argument('--test-per-bin',type=int,default=100);p.add_argument('--train-per-word',type=int,default=100); a=p.parse_args()
    root=Path(a.aligned_dir); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);rng=random.Random(a.seed)
    loaded={k:load(root,k) for k in FILES}; assert not loaded['train'][1]&loaded['test-clean'][1] and not loaded['train'][1]&loaded['test-other'][1] and not loaded['test-clean'][1]&loaded['test-other'][1]
    summary={'seed':a.seed,'policy':'word count uniformly balanced; full positive non-overlapping alignment; random contiguous windows; test source used at most once per version and least-used across versions; plans only, audio bounds not yet checked','inputs':{k:{'input_rows':len(v[1]),'eligible':len(v[0]),'rejected':v[2],'sha256':v[3]} for k,v in loaded.items()},'subsets':{}}
    for split,(pool,_,_,_) in loaded.items():
        use=collections.Counter(); versions=[]; bins=[(1,100)] if split=='train' else [(i,i+9) for i in range(1,101,10)]
        for lo,hi in bins:
            name='train' if split=='train' else f'{split}_{lo:03d}_{hi:03d}'; used=set(); counts=collections.Counter();targets=([i for i in range(1,101) for _ in range(a.train_per_word)] if split=='train' else [lo+(i%10) for i in range(a.test_per_bin)]);rng.shuffle(targets)
            with (out/(name+'.jsonl')).open('x') as f, (out/(name+'.rejected.jsonl')).open('x') as rej:
                for i,target in enumerate(targets):
                    segments=[]; remaining=target;sample_ids=set()
                    while remaining:
                        if split=='train':
                            chosen=rng.choice(pool)
                            while chosen['id'] in sample_ids: chosen=rng.choice(pool)
                        else:
                            candidates=[x for x in pool if x['id'] not in used and x['id'] not in sample_ids]
                            if not candidates:break
                            minuse=min(use[x['id']] for x in candidates); chosen=rng.choice([x for x in candidates if use[x['id']]==minuse])
                        s=segment(chosen,remaining,rng);segments.append(s);sample_ids.add(chosen['id']);remaining-=len(s['word_timestamps'])
                    if remaining:
                        counts['unplanned']+=1;rej.write(json.dumps({'id':f'{name}_{i:05d}','target_words':target,'reason':'eligible sources exhausted'})+'\n');continue
                    if split!='train': used.update(sample_ids);use.update(sample_ids)
                    text=' '.join(s['text'] for s in segments); assert len(tokens(text))==target
                    row={'id':f'{name}_{i:05d}','seed':a.seed,'target_words':target,'text':text,'estimated_duration':sum(s['end']-s['start'] for s in segments),'segments':segments}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');counts['planned']+=1;counts[f'{len(segments)}_segments']+=1;counts[f'words_{target}']+=1
            versions.append((name,used));summary['subsets'][name]={'requested':len(targets),'source_count':len(used) if split!='train' else None,'counts':dict(counts)}
        if split!='train': summary[split+'_version_overlap']={f'{a}__{b}':len(sa&sb) for i,(a,sa) in enumerate(versions) for b,sb in versions[i+1:]};summary[split+'_unique_sources']=len(use)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()

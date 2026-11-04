#!/usr/bin/env python3
"""Prefetch distinct Orange source files with a shared HTTP session and cache locks."""
import argparse,asyncio,collections,fcntl,hashlib,json,os,uuid
from pathlib import Path
import boostedblob as bbb
import soundfile as sf
from verl.audio_cache import local_audio_source

@bbb.ensure_session
async def prefetch(args):
    seen=set();sources=[]
    for line in Path(args.input).open():
        src=json.loads(line)['audio_path']
        if src not in seen:seen.add(src);sources.append(src)
    results=Path(args.results)
    if results.exists() or Path(args.report).exists():raise FileExistsError('result paths already exist')
    queue=asyncio.Queue()
    for src in sources:queue.put_nowait(src)
    counts=collections.Counter();lockroot=Path('/tmp/verl-audio-locks');lockroot.mkdir(parents=True,exist_ok=True)
    async with bbb.BoostExecutor(args.workers) as executor:
        async def worker(out):
            while True:
                try:src=queue.get_nowait()
                except asyncio.QueueEmpty:return
                dest=Path(local_audio_source(src));status='unknown';detail=None
                try:
                    dest.parent.mkdir(parents=True,exist_ok=True);lock=lockroot/(hashlib.sha256(str(dest).encode()).hexdigest()+'.lock')
                    with lock.open('w') as f:
                        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:status='busy'
                        else:
                            if dest.exists():status='cached'
                            else:
                                tmp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp')
                                try:
                                    await asyncio.wait_for(bbb.copyfile(src,str(tmp),executor),timeout=args.timeout)
                                    info=sf.info(tmp)
                                    if info.samplerate!=16000 or info.channels!=1 or info.frames<=0:raise ValueError('invalid FLAC format')
                                    os.replace(tmp,dest);status='downloaded'
                                finally:tmp.unlink(missing_ok=True)
                except Exception as e:status='failed';detail=repr(e)
                counts[status]+=1;out.write(json.dumps({'audio_path':src,'status':status,'error':detail})+'\n')
                done=sum(counts.values())
                if done%100==0:out.flush();print(json.dumps({'done':done,'total':len(sources),'counts':dict(counts)}),flush=True)
                queue.task_done()
        with results.open('x') as out:await asyncio.gather(*(worker(out) for _ in range(args.workers)))
    report={'unique_sources':len(sources),'counts':dict(counts),'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),'cache_root':os.environ.get('VERL_AUDIO_CACHE_ROOT')}
    with Path(args.report).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)
    return 1 if counts['failed'] else 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--results',required=True);p.add_argument('--report',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--timeout',type=int,default=90);a=p.parse_args();assert 1<=a.workers<=32 and a.timeout>0
    raise SystemExit(asyncio.run(prefetch(a)))
if __name__=='__main__':main()

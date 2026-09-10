"""Optional transport accelerator; succeeds only on an exact user-local SHA256."""
import argparse,concurrent.futures,fcntl,hashlib,json,os
from pathlib import Path
import time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
lock=(root/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
chunk=64*1024*1024;target=root/'archive.part';fd=os.open(target,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,a.bytes)
identity=dict(url=a.url,bytes=a.bytes,sha256=a.sha256,chunk_bytes=chunk)
meta=root/'manifest.json'
if meta.exists() and json.loads(meta.read_text())!=identity:raise ValueError('Existing mirror attempt has a different identity')
meta.write_text(json.dumps(identity,indent=2))
def one(i):
 mark=root/f'{i:04d}.json';lo=i*chunk;hi=min(a.bytes,lo+chunk)-1
 if mark.exists():return
 for attempt in range(8):
  try:
   u=a.url+('?download=true&chunk=' if '?' not in a.url else '&chunk=')+str(i)
   req=urllib.request.Request(u,headers={'Range':f'bytes={lo}-{hi}'})
   with urllib.request.urlopen(req,timeout=45) as response:
    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{a.bytes}':raise ValueError('Incorrect range response')
    pos=lo;h=hashlib.sha256()
    while True:
     b=response.read(1024*1024)
     if not b:break
     if pos+len(b)>hi+1:raise ValueError('Oversized range')
     h.update(b);offset=0
     while offset<len(b):offset+=os.pwrite(fd,b[offset:],pos+offset)
     pos+=len(b)
    if pos!=hi+1:raise ValueError('Short range')
   os.fsync(fd)
   tmp=mark.with_suffix('.tmp');tmp.write_text(json.dumps(dict(index=i,bytes=hi-lo+1,sha256=h.hexdigest())));tmp.replace(mark)
   print(json.dumps(dict(event='chunk_done',index=i,bytes=hi-lo+1)),flush=True);return
  except Exception as e:
   print(json.dumps(dict(event='retry',index=i,attempt=attempt,error=str(e)[:160])),flush=True);time.sleep(min(15,attempt+1))
 raise RuntimeError('Failed chunk '+str(i))
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,range((a.bytes+chunk-1)//chunk)))
 os.close(fd);h=hashlib.sha256()
 with target.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=a.sha256:raise ValueError('Downloaded bytes do not match user-local archive')
 result=dict(status='verified',remote_path=str(target.resolve()),bytes=a.bytes,sha256=h.hexdigest(),remote_sha256=h.hexdigest(),source_url=a.url,transport='HF mirror; exact SHA256 match to user-local archive')
 (root/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=str(e))));raise

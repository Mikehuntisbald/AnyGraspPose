"""Fetch immutable official LFS weight and verify the publisher's SHA256."""
from pathlib import Path
import urllib.request,json,time,hashlib,os
ROOT=Path('/mnt/why/dexycb_lip/unified_jepa_20260921/official_flow_v65')
SHA='f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'
COMMIT='68f76055755f2a4a8967e13ece834f975f008bdf'
SIZE=1608850339
url=f'https://media.githubusercontent.com/media/facebookresearch/gotrack/{COMMIT}/gotrack_checkpoint.pt'
out=ROOT/'weights/gotrack.pt';part=out.with_suffix('.part')
def status(**kw):
 p=ROOT/'download_status.tmp';p.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),**kw),indent=2));p.replace(ROOT/'download_status.json')
if out.exists():
 assert out.stat().st_size==SIZE and hashlib.file_digest(out.open('rb'),'sha256').hexdigest()==SHA
 status(stage='complete',sha256=SHA,bytes=SIZE);raise SystemExit
try:
 for attempt in range(5):
  start=part.stat().st_size if part.exists() else 0
  if start==SIZE:break
  status(stage='downloading',bytes=start,expected=SIZE,attempt=attempt)
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {})
   with urllib.request.urlopen(req,timeout=30) as response:
    append=start>0 and response.status==206
    if append and not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise RuntimeError('Wrong resume offset')
    with part.open('ab' if append else 'wb') as f:
     last=time.monotonic()
     while block:=response.read(1024*1024):
      f.write(block)
      if time.monotonic()-last>5:
       f.flush();status(stage='downloading',bytes=f.tell(),expected=SIZE,attempt=attempt);last=time.monotonic()
   if part.stat().st_size==SIZE:break
  except Exception as exc:
   status(stage='retry',error=type(exc).__name__,bytes=part.stat().st_size if part.exists() else 0,attempt=attempt)
   if attempt==4:raise
   time.sleep(5)
 assert part.stat().st_size==SIZE
 status(stage='verifying',bytes=SIZE)
 assert hashlib.file_digest(part.open('rb'),'sha256').hexdigest()==SHA
 part.replace(out)
 (ROOT/'weight_receipt.json').write_text(json.dumps(dict(source_commit=COMMIT,source='Official GitHub LFS',url=url,sha256=SHA,bytes=SIZE,
  modelscope_search='No matching indexed result found; this is not proof no mirror exists',default_model_changed=False),indent=2)+'\n')
 status(stage='complete',sha256=SHA,bytes=SIZE)
except Exception as exc:
 status(stage='failed',error=type(exc).__name__);raise

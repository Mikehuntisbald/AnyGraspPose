"""Resumable upload; validates source gzip CRC and compares full SHA256 remotely."""
import concurrent.futures,fcntl,gzip,hashlib,json,os,pathlib,subprocess,time,traceback,shlex
BASE=pathlib.Path('/home/haoyi/Downloads/xd/dexycb_lip/runs/upload_dexycb')
MANIFEST=json.loads((BASE/'manifest.json').read_text())
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-p','10863','root@111.230.4.68']
REMOTE='/mnt/why/DexYCB'
lock=(BASE/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def event(kind,**kw):
 print(json.dumps(dict(event=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)),flush=True)
def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def validate(row):
 src=pathlib.Path(row['path']);stat=src.stat();receipt=BASE/(row['name']+'.source.json')
 if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('Source changed since manifest: '+src.name)
 if receipt.exists():
  old=json.loads(receipt.read_text())
  if old['bytes']==stat.st_size and old['mtime_ns']==stat.st_mtime_ns:return old
 event('validate_start',file=src.name,bytes=stat.st_size)
 h=hashlib.sha256()
 class Reader:
  def __init__(self,f):self.f=f
  def read(self,n=-1):
   b=self.f.read(n);h.update(b);return b
 with src.open('rb') as f:
  with gzip.GzipFile(fileobj=Reader(f),mode='rb') as z:
   first=z.read(512)
   if len(first)!=512 or first[257:262] not in (b'ustar',):raise ValueError('Not a recognized tar archive: '+src.name)
   while z.read(8*1024*1024):pass
 final=src.stat()
 if (final.st_size,final.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise RuntimeError('Source changed during validation')
 result=dict(**row,sha256=h.hexdigest(),gzip_crc_verified=True)
 write(receipt,result);event('validate_done',file=src.name,sha256=result['sha256']);return result

def upload(row):
 r=validate(row);name=row['name'];stage=REMOTE+'/.upload/'+name+'.part';dest=REMOTE+'/'+name
 receipt=BASE/(name+'.uploaded.json')
 if receipt.exists():
  old=json.loads(receipt.read_text())
  if old.get('sha256')==r['sha256'] and old.get('remote_sha256')==r['sha256']:
   event('already_uploaded',file=name);return old
 command=['rsync','--times','--perms','--partial','--append-verify','--info=progress2','--outbuf=L','--timeout=120',
          '--rsync-path=/mnt/why/dexycb_lip/.transfer-tools/rsync-run','-e',
          'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -p 10863',
          row['path'],'root@111.230.4.68:'+stage]
 start=time.monotonic();event('upload_start',file=name,bytes=row['bytes'])
 for attempt in range(1,5):
  with (BASE/(name+'.rsync.log')).open('a') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode==0:break
  event('upload_retry',file=name,attempt=attempt,returncode=result.returncode);time.sleep(min(30,attempt*5))
 else:raise RuntimeError('rsync failed: '+name)
 if pathlib.Path(row['path']).stat().st_mtime_ns!=row['mtime_ns']:raise RuntimeError('Source changed during upload')
 event('remote_verify_start',file=name)
 code='''import pathlib,hashlib,json,os,sys
stage=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2]);expected=sys.argv[3];size=int(sys.argv[4])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert stage.stat().st_size==size, 'Size mismatch'
digest=sha(stage)
assert digest==expected, 'SHA256 mismatch'
try:os.link(stage,dest)
except FileExistsError:
 assert dest.stat().st_size==size and sha(dest)==expected, 'Different destination already exists'
stage.unlink()
print(json.dumps(dict(remote_sha256=digest,remote_path=str(dest),bytes=size)))
'''
 result=subprocess.run(SSH+['python3','-',stage,dest,r['sha256'],str(r['bytes'])],input=code,text=True,capture_output=True)
 if result.returncode:raise RuntimeError(result.stderr)
 remote=json.loads(result.stdout);done=dict(r,**{k:v for k,v in remote.items() if k!='bytes'},transfer_verify_seconds=time.monotonic()-start,status='verified')
 write(receipt,done);event('upload_verified',file=name,bytes=row['bytes'],seconds=round(time.monotonic()-start,2));return done

write(BASE/'job.json',dict(pid=os.getpid(),status='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
event('start',files=len(MANIFEST['files']),total_bytes=MANIFEST['total_bytes'],workers=4)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(upload,sorted(MANIFEST['files'],key=lambda r:r['bytes'])))
 receipt=dict(status='complete',source=MANIFEST['source'],destination=REMOTE,total_bytes=sum(r['bytes'] for r in results),files=results,
              verification='Source gzip CRC + local SHA256 equals remote SHA256; does not assert publisher checksum',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 payload=json.dumps(receipt,indent=2)
 result=subprocess.run(SSH+["python3 -c "+shlex.quote("import sys,pathlib; p=pathlib.Path('/mnt/why/DexYCB/upload_receipt.json'); t=p.with_suffix('.tmp'); t.write_text(sys.stdin.read()); t.replace(p)")],input=payload,text=True,capture_output=True)
 if result.returncode:raise RuntimeError(result.stderr)
 write(BASE/'receipt.json',receipt);write(BASE/'job.json',dict(pid=os.getpid(),status='complete'))
 event('complete',files=len(results),bytes=receipt['total_bytes'])
except Exception as e:
 write(BASE/'job.json',dict(pid=os.getpid(),status='failed',error=str(e)));event('failed',error=str(e));traceback.print_exc();raise

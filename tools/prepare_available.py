"""Extract a frozen list of verified uploaded archives into project-owned cache."""
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess,time,fcntl
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--receipt-name',default='extraction_receipt.json');a=p.parse_args()
manifest=json.loads(pathlib.Path(a.manifest).read_text());out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
def extract_locked(r):
 src=pathlib.Path(r['remote_path']);name=src.name.removesuffix('.tar.gz');dest=out/name;mark=out/(name+'.extracted.json')
 if mark.exists():
  old=json.loads(mark.read_text());assert old['sha256']==r['sha256'] and dest.exists();return old
 if dest.exists():raise RuntimeError('Unsealed destination exists: '+str(dest))
 h=hashlib.sha256()
 with src.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==r['sha256'] and src.stat().st_size==r['bytes']
 stage=out/('.extract-'+name);stage.mkdir(exist_ok=True)
 # Input archives were user-downloaded and gzip-verified during transfer; GNU tar rejects traversal.
 subprocess.run(['tar','-xzf',str(src),'-C',str(stage),'--no-same-owner'],check=True)
 if name.startswith('2020'):
  assert len(list((stage/name).glob('*/meta.yml')))==100
 (stage/name).rename(dest)
 receipt=dict(archive=str(src),sha256=r['sha256'],bytes=r['bytes'],root=str(dest),status='extracted',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 mark.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);return receipt
def run(r):
 name=pathlib.Path(r['remote_path']).name.removesuffix('.tar.gz')
 with (out/(name+'.extract.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return extract_locked(r)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,manifest['files']))
(out/a.receipt_name).write_text(json.dumps(dict(files=results,scope=manifest['scope']),indent=2))

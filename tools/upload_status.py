import pathlib,json,re,subprocess,shlex
p=pathlib.Path('/home/haoyi/Downloads/xd/dexycb_lip/runs/upload_dexycb')
m=json.loads((p/'manifest.json').read_text());done=[];active=[];progress=0
for r in m['files']:
 receipt=p/(r['name']+'.uploaded.json')
 if receipt.exists():done.append(r['name']);progress+=r['bytes'];continue
 log=p/(r['name']+'.rsync.log');n=0;last=''
 if log.exists():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-8192));s=f.read().decode(errors='replace')
  matches=re.findall(r'([\d,]+)\s+\d+%\s+([^\r\n]+)',s)
  if matches:n=int(matches[-1][0].replace(',',''));last=matches[-1][1].strip()
  progress+=min(n,r['bytes']);active.append(dict(file=r['name'],uploaded_bytes=n,progress=last))
# rsync counts prefix re-verification as progress after a resume. Use actual
# remote file lengths for the total rather than mistaking that scan for transfer.
remote_progress=None;remote_error=None
code="""import json,sys,pathlib
root=pathlib.Path('/mnt/why/DexYCB');m=json.load(sys.stdin);n=0
for r in m['files']:
 final=root/r['name'];part=root/'.upload'/(r['name']+'.part')
 p=final if final.exists() else part
 if p.exists():n+=min(p.stat().st_size,r['bytes'])
print(n)
"""
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','10863','root@111.230.4.68',
                   'python3 -c '+shlex.quote(code)],input=json.dumps(m),text=True,capture_output=True,timeout=15)
 if r.returncode==0:remote_progress=int(r.stdout.strip())
 else:remote_error='Remote progress query failed'
except (OSError,ValueError,subprocess.TimeoutExpired):remote_error='Remote progress unavailable'
print(json.dumps(dict(job=json.loads((p/'job.json').read_text()),verified_files=len(done),total_files=len(m['files']),
                     verified_names=done,uploaded_or_verified_GB=round((remote_progress if remote_progress is not None else progress)/1e9,3),
                     progress_source='remote file lengths' if remote_progress is not None else 'local rsync estimate; may include prefix verification',
                     remote_error=remote_error,total_GB=round(m['total_bytes']/1e9,3),active=active),indent=2))

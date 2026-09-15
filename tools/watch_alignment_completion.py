"""Monitor the remote staged experiment and download its verified terminal archive."""
import json,pathlib,subprocess,time,shlex
root=pathlib.Path(__file__).resolve().parents[1]
folder=root/'runs/intraframe_alignment_20260915_completion';folder.mkdir(exist_ok=False)
remote='/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs'
state=dict(phase='waiting',started=time.time())
def save():
 p=folder/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(folder/'status.json')
try:
 save();deadline=time.time()+8*3600
 while True:
  try:
   raw=subprocess.check_output(['ssh','-p','10863','-o','BatchMode=yes','-o','ConnectTimeout=15','root@111.230.4.68','cat '+shlex.quote(remote+'/alignment_stages/status.json')],text=True,timeout=30)
   s=json.loads(raw)
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
   if time.time()>deadline:raise
   state['connection_error']=str(e);save();time.sleep(30);continue
  state['remote_phase']=s['phase'];save()
  if s['phase']=='failed':raise RuntimeError(s.get('error'))
  if s['phase']=='completed':break
  if time.time()>deadline:raise TimeoutError('Remote completion exceeded eight hours')
  time.sleep(30)
 state['phase']='fetching';save()
 subprocess.run(['python3','tools/watch_real_init_delivery.py','--host','root@111.230.4.68','--port','10863','--remote-out',remote+'/alignment_completed','--local-out',str(folder/'package')],cwd=root,check=True)
 state.update(phase='completed',completed=time.time());save()
except BaseException as error:
 state.update(phase='failed',error=repr(error));save();raise

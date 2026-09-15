"""Wait for the frozen iteration experiment, then preflight/train/evaluate the alignment pair."""
import json,os,subprocess,sys,time
from pathlib import Path


def main():
 root=Path(__file__).resolve().parents[1];runs=root/'runs';folder=runs/'alignment_stages';folder.mkdir(exist_ok=False)
 state=dict(phase='waiting_intraframe',started=time.time(),commands=[])
 def save():
  t=folder/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(folder/'status.json')
 def run(phase,cmd):
  state['phase']=phase
  with (folder/(phase+'.log')).open('x') as f:
   proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   state['commands'].append(dict(phase=phase,pid=proc.pid,command=cmd));save()
   if proc.wait()!=0:raise RuntimeError(phase+' failed; inspect phase log')
 try:
  save()
  while True:
   s=json.loads((runs/'intraframe/status.json').read_text())
   if s['phase']=='failed':raise RuntimeError('Frozen iteration experiment failed: '+str(s.get('error')))
   if s['phase']=='completed':break
   time.sleep(10)
  run('training',[sys.executable,'tools/run_rotation_alignment_training.py','--experiment',str(runs/'alignment_pair')])
  run('evaluation',[sys.executable,'tools/finish_rotation_alignment_pair.py','--experiment',str(runs/'alignment_pair'),
   '--fp-scored','/mnt/why/dexycb_lip/fp_val_20260915/runs/full_v2/scored','--residual-scored','/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/full_val_v2/residual/scored',
   '--visibility-reference','/mnt/why/dexycb_lip/adaptive_reference_20260914/runs/adaptive_pair/adaptive_reference/s0_val','--fp-root','/mnt/why/dexycb_lip/third_party/FoundationPose','--out',str(runs/'alignment_val')])
  run('delivery',[sys.executable,'tools/deliver_rotation_alignment_pair.py','--experiment',str(runs/'alignment_pair'),'--evaluation',str(runs/'alignment_val'),
   '--fp-runtime','/mnt/why/dexycb_lip/fp_val_20260915','--posecnn-runtime','/mnt/why/dexycb_lip/posecnn_train_20260915','--out',str(runs/'alignment_completed'),'--wait-completion'])
  state.update(phase='completed',completed=time.time());save()
 except BaseException as e:
  state.update(phase='failed',error=repr(e));save();raise

if __name__=='__main__':main()

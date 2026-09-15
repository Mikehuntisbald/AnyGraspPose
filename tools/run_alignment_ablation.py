"""Full native val learned/zero residual under one fixed checkpoint, with exact replay checks."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
 root=Path(__file__).resolve().parents[1];r=root/'runs/ablation';r.mkdir(exist_ok=False)
 old=Path('/mnt/why/dexycb_lip/intraframe_alignment_20260915/runs/alignment_val/alignment')
 checkpoint=old/'frozen.pt';config=old/'config.yaml';expected='014e13710125daf7b103f5b1050beb9e8876eb86036ff4dfdff5973019af9bb7';assert sha(checkpoint)==expected
 initial='/mnt/why/dexycb_lip/posecnn_val_20260915/runs/full/initializers.json';data='/mnt/why/dexycb_lip/cache/raw_full_20260910';index='/mnt/why/dexycb_lip/cache/dexycb_s0';vis='/mnt/why/dexycb_lip/adaptive_reference_20260914/runs/adaptive_pair/adaptive_reference/s0_val'
 state=dict(phase='starting',started=time.time(),commands=[],source_sha256=source_hash(),checkpoint_sha256=expected);jobs=[]
 def save():
  p=r/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(r/'status.json')
 def run(phase,commands):
  nonlocal jobs
  state['phase']=phase;jobs=[]
  for name,gpu,cmd,path in commands:
   path.parent.mkdir(parents=True,exist_ok=True);f=path.open('x');proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);jobs.append((proc,f));state['commands'].append(dict(name=name,pid=proc.pid,command=cmd,log=str(path)));save()
  while any(p.poll() is None for p,f in jobs):
   if any(p.poll() not in (None,0) for p,f in jobs):raise RuntimeError(phase+' failed')
   time.sleep(5)
  for p,f in jobs:f.close();assert p.returncode==0
  jobs=[]
 def infer(name,rank=0,world=1,extra=()):
  return [sys.executable,'tools/infer_alignment_intervention.py','--config',str(config),'--checkpoint',str(checkpoint),'--initializers',initial,'--data-root',data,'--index-root',index,'--intervention',name,'--rank',str(rank),'--world',str(world),'--out',str(r/name/'inference'/f'rank{rank}'),*extra]
 try:
  save()
  # Two real streams establish exact learned-route replay before the full ablation.
  cmd=infer('learned',extra=('--limit-streams','2','--max-frames','10'));cmd[cmd.index('--out')+1]=str(r/'preflight')
  run('preflight',[('learned','0',cmd,r/'preflight.log')])
  archived={(x['stream_id'],x['frame_index']):x for x in map(json.loads,(old/'scored/predictions.jsonl').read_text().splitlines())}
  for row in map(json.loads,(r/'preflight/predictions.jsonl').read_text().splitlines()):assert row['pose_centered']==archived[row['stream_id'],row['frame_index']]['pose_centered']
  cmds=[]
  for i,name in enumerate(('learned','zero')):
   for rank in range(3):cmds.append((name+str(rank),str(i*3+rank),infer(name,rank,3),r/name/'inference'/f'rank{rank}.log'))
  run('inference',cmds)
  cmds=[]
  for n in ('learned','zero'):
   cmds.append((n,'',[sys.executable,'tools/score_val_non_gt.py','--run',str(r/n/'inference'),'--world','3','--index-root',index,'--visibility-reference',vis,'--out',str(r/n/'scored')],r/n/'scoring.log'))
  run('scoring',cmds)
  learned=list(map(json.loads,(r/'learned/scored/predictions.jsonl').read_text().splitlines()))
  for row in learned:
   oldrow=archived[row['stream_id'],row['frame_index']]
   for key in ('pose_centered','add_01','adds_005','status'):assert row[key]==oldrow[key],(row['stream_id'],row['frame_index'],key)
  assert sha(checkpoint)==expected and source_hash()==state['source_sha256']
  state.update(phase='completed',completed=time.time(),full_learned_reproduction=True);save()
 except BaseException as error:
  for p,f in jobs:
   if p.poll() is None:p.terminate()
  for p,f in jobs:p.wait();f.close()
  state.update(phase='failed',error=repr(error));save();raise

if __name__=='__main__':main()

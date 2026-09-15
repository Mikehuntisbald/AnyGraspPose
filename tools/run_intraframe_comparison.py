"""Frozen retained checkpoint: full one/two-pass startup evaluation and isolated timing."""
import json,os,subprocess,sys,time
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
 root=Path(__file__).resolve().parents[1];r=root/'runs/intraframe';r.mkdir(exist_ok=False)
 old=Path('/mnt/why/dexycb_lip/real_init_training_20260915_v2/runs/full_val_v2/residual');checkpoint=old/'frozen.pt';expected='89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868';assert sha(checkpoint)==expected
 initial='/mnt/why/dexycb_lip/posecnn_val_20260915/runs/full/initializers.json';data='/mnt/why/dexycb_lip/cache/raw_full_20260910';index='/mnt/why/dexycb_lip/cache/dexycb_s0';vis='/mnt/why/dexycb_lip/adaptive_reference_20260914/runs/adaptive_pair/adaptive_reference/s0_val'
 frozen=source_hash();cfg=r/'config.yaml';cfg.write_text(yaml.safe_dump(torch.load(checkpoint,map_location='cpu',weights_only=False)['config']))
 state=dict(phase='starting',started=time.time(),commands=[]);jobs=[]
 def save():
  t=r/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(r/'status.json')
 def run(phase,commands):
  nonlocal jobs
  state['phase']=phase;jobs=[]
  for name,gpu,cmd,path in commands:
   path.parent.mkdir(parents=True,exist_ok=True);f=path.open('x');p=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f));state['commands'].append(dict(name=name,pid=p.pid,command=cmd,log=str(path)));save()
  while any(p.poll() is None for p,f in jobs):
   if any(p.poll() not in (None,0) for p,f in jobs):raise RuntimeError(phase+' failed')
   time.sleep(5)
  for p,f in jobs:f.close();assert p.returncode==0
  jobs=[]
 try:
  save();run('cuda_tests',[('cuda','0',[sys.executable,'-m','pytest','-c','pyproject.toml','tests/test_stream_batch_features.py','tests/test_stream_cross_cuda.py','-m','cuda','-q','--junitxml='+str(r/'cuda_tests.xml')],r/'cuda_tests.log')])
  commands=[]
  for ai,(name,iters) in enumerate((('residual',1),('two_pass',2))):
   for rank in range(4):
    folder=r/name/'inference'/f'rank{rank}';cmd=[sys.executable,'tools/infer_lip_val_non_gt.py','--config',str(cfg),'--checkpoint',str(checkpoint),'--initializers',initial,'--data-root',data,'--index-root',index,'--rank',str(rank),'--world','4','--startup-iterations',str(iters),'--out',str(folder)]
    commands.append((name+str(rank),str(ai*4+rank),cmd,folder.with_suffix('.log')))
  run('full_val_inference',commands)
  commands=[]
  for n in ('residual','two_pass'):
   commands.append((n,'',[sys.executable,'tools/score_val_non_gt.py','--run',str(r/n/'inference'),'--world','4','--index-root',index,'--visibility-reference',vis,'--out',str(r/n/'scored')],r/n/'score.log'))
  run('scoring',commands)
  previous={ (x['stream_id'],x['frame_index']):x for x in map(json.loads,(old/'scored/predictions.jsonl').read_text().splitlines())}
  rows=list(map(json.loads,(r/'residual/scored/predictions.jsonl').read_text().splitlines()))
  for row in rows:
   ref=previous[row['stream_id'],row['frame_index']]
   for key in ('pose_centered','add_01','adds_005','status'):assert row[key]==ref[key],(row['stream_id'],row['frame_index'],key)
  (r/'baseline_reproduction.json').write_text(json.dumps(dict(passed=True,frames=len(rows),all_poses_and_scores_identical=True)))
  run('comparison',[('compare','',[sys.executable,'tools/compare_fp_scorecard.py','--evaluation','fp=/mnt/why/dexycb_lip/fp_val_20260915/runs/full_v2/scored','--evaluation','residual='+str(r/'residual/scored'),'--evaluation','two_pass='+str(r/'two_pass/scored'),'--out',str(r/'comparison')],r/'comparison.log')])
  for n,iters in (('residual',1),('two_pass',2)):
   cmd=[sys.executable,'tools/benchmark_val_tracking.py','--method','lip','--config',str(cfg),'--checkpoint',str(checkpoint),'--expected-sha',expected,'--initializers',initial,'--data-root',data,'--index-root',index,'--fp-root','/mnt/why/dexycb_lip/third_party/FoundationPose','--startup-iterations',str(iters),'--out',str(r/'timing'/n)]
   run('timing_'+n,[(n,'0',cmd,r/'timing'/f'{n}.log')])
  assert sha(checkpoint)==expected and source_hash()==frozen
  result=dict(completed=True,checkpoint_sha256=expected,source_sha256=frozen,initializers_sha256=sha(initial),full_val_frames=23200,streams=320,training_steps=0,
   cache_protocol='Each iteration reads the original immutable past. Re-render with new base, retain actual inter-frame motion. Only final valid proposal and its current-frame KV are committed; second failure falls back to first.',
   selection='Diagnostic comparison only; no automatic checkpoint promotion or test launch.',comparison_sha256=sha(r/'comparison/comparison.json'))
  (r/'receipt.json').write_text(json.dumps(result,indent=2));state.update(phase='completed',completed=time.time());save()
 except BaseException as ex:
  for p,f in jobs:
   if p.poll() is None:p.terminate()
  for p,f in jobs:p.wait();f.close()
  state.update(phase='failed',error=repr(ex));save();raise

if __name__=='__main__':main()

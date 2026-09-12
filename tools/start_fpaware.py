"""Validate an isolated candidate, then resume FP-aware training from step 19000."""
import os,json,time,subprocess,pathlib,shutil,hashlib
R=pathlib.Path('/mnt/why/dexycb_lip');os.chdir(R);J=R/'runs/fpaware_19000';C=J/'candidate';E=R/'.venv-fp/bin/python';cfg=C/'configs/fpaware_19000.yaml'
def status(phase,**kw):
 r=dict(phase=phase,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
 t=J/'status.tmp';t.write_text(json.dumps(r,indent=2));t.replace(J/'status.json');print(json.dumps(r),flush=True)
status('waiting_pause')
while not (J/'paused.json').exists():time.sleep(2)
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(2)
assert json.loads((J/'probe.json').read_text())['passed']
assert '34 passed' in (J/'all_tests.log').read_text()
shutil.copy2(R/'runs/lip_v1_s0/best.pt',J/'pre_fpaware_best.pt')
env=os.environ.copy();env.update(PYTHONPATH=str(C/'src'),DEX_YCB_DIR=str(R/'cache/raw_full_20260910'),TORCH_HOME=str(R/'cache/torch'),OMP_NUM_THREADS='2',MAX_JOBS='4',TMPDIR=str(R/'cache/tmp'));env.pop('CUDA_VISIBLE_DEVICES',None)
def launch(out,extra,log):
 cmd=[str(E),'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','lip.train','--config',str(cfg),'--data-root',env['DEX_YCB_DIR'],'--index-root','cache/dexycb_s0','--output',str(out),'--resume',str(J/'resume.pt')]+extra
 with log.open('a') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);status('ddp_probe' if extra else 'training',child_pid=p.pid,output=str(out));return p.wait()
rc=launch(J/'ddp_probe',['--max-steps','19006'],J/'ddp_probe.log')
if rc:status('preflight_failed',returncode=rc);raise SystemExit(rc)
allrows=[]
for rank in range(8):
 rows=[json.loads(x) for x in (J/f'ddp_probe/rank{rank}.jsonl').read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(19001,19007))
 assert {r['history_mode'] for r in rows}=={'noisy_gt','lip_only','lip_fp'}
 assert all(r['rollout']==4 and r['nonfinite_count']==0 and r['scheduler_step']==r['step'] and r['sampler_position']==r['step'] for r in rows)
 if allrows:assert [(r['history_mode'],r['history_probs']) for r in rows]==[(r['history_mode'],r['history_probs']) for r in allrows[0]]
 allrows.append(rows)
import yaml
c=yaml.safe_load(cfg.read_text());c.update(preflight_approved=True,fpaware_preflight_passed=True);cfg.write_text(yaml.safe_dump(c,sort_keys=False))
receipt=dict(passed=True,rank_steps=[r[-1]['step'] for r in allrows],config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),scope='34 unit tests plus real batch32 three-mode backward probe and eight-rank six-step resumed FP-aware run')
(J/'preflight_passed.json').write_text(json.dumps(receipt,indent=2))
status('preflight_passed')
rc=launch(R/'runs/lip_v1_s0',[],J/'train.log');status('training_exited',returncode=rc)
if rc==0:subprocess.check_call([str(E),str(C/'tools/verify_training_completion.py'),'--expected-steps','40000'],env=env)

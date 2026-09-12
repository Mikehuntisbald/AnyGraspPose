"""Start basin continuation only after critic quality, drift, gradients and DDP pass."""
import pathlib,json,time,os,subprocess,hashlib,yaml
R=pathlib.Path('/mnt/why/dexycb_lip');os.chdir(R);J=R/'runs/basin_boundary_v3';C=J/'candidate';PY=R/'.venv-fp/bin/python'
def state(phase,**kw):
 r=dict(phase=phase,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);t=J/'status.tmp';t.write_text(json.dumps(r,indent=2));t.replace(J/'status.json');print(json.dumps(r),flush=True)
state('waiting_preflight')
for name in ['gpu_health.json','drift.json','gradient_probe.json']:
 while not (J/name).exists():time.sleep(2)
 if not json.loads((J/name).read_text())['passed']:state('blocked_preflight',failed_receipt=name);raise SystemExit(1)
while not (J/'tests.log').exists() or 'passed' not in (J/'tests.log').read_text():time.sleep(2)
assert 'failed' not in (J/'tests.log').read_text() and 'ERROR' not in (J/'tests.log').read_text()
quality=json.loads((J/'fit/receipt.json').read_text());assert quality['passed']
while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(2)
c=yaml.safe_load((R/'runs/fpaware_19000/candidate/configs/fpaware_19000.yaml').read_text());weight=J/'fit/critic.pt'
c.update(basin_enabled=True,basin_quality_policy='validated',basin_checkpoint=str(weight),basin_checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),basin_start_step=20000,basin_lambda_max=.01,basin_warmup_steps=1000,basin_quality_gate_passed=True,basin_preflight_passed=False,run_id='basin_boundary_v3')
formal=C/'configs/basin_validated.yaml';formal.write_text(yaml.safe_dump(c,sort_keys=False));probe=dict(c,basin_start_step=19999,basin_warmup_steps=1,run_id='basin_boundary_v3_ddp_probe');probe_path=C/'configs/basin_ddp_probe.yaml';probe_path.write_text(yaml.safe_dump(probe,sort_keys=False))
env=os.environ.copy();env.update(PYTHONPATH=str(C/'src'),DEX_YCB_DIR=str(R/'cache/raw_full_20260910'),TORCH_HOME=str(R/'cache/torch'),OMP_NUM_THREADS='2',MAX_JOBS='4',TMPDIR=str(R/'cache/tmp'));env.pop('CUDA_VISIBLE_DEVICES',None)
def launch(config,out,log,maxsteps=None):
 cmd=[str(PY),'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','lip.train','--config',str(config),'--data-root',env['DEX_YCB_DIR'],'--index-root','cache/dexycb_s0','--output',str(out),'--resume',str(J/'resume_20000.pt')]
 if maxsteps:cmd+=['--max-steps',str(maxsteps)]
 with log.open('a') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);state('ddp_probe' if maxsteps else 'training',child_pid=p.pid,output=str(out),config=str(config));return p.wait()
rc=launch(probe_path,J/'ddp_probe',J/'ddp_probe.log',20006)
if rc:state('ddp_failed',returncode=rc);raise SystemExit(rc)
rank_receipts=[]
for rank in range(8):
 rows=[json.loads(x) for x in (J/f'ddp_probe/rank{rank}.jsonl').read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(20001,20007))
 assert {r['history_mode'] for r in rows}=={'lip_fp','lip_only','noisy_gt'}
 assert all(r['nonfinite_count']==0 and r['basin_weight']==.01 and r['basin_loss']>0 and r['scheduler_step']==r['step'] and r['sampler_position']==r['step'] for r in rows)
 assert all(abs(r['loss']-r['pose_loss']-.01*r['basin_loss'])<2e-6 for r in rows)
 rank_receipts.append(dict(rank=rank,step=rows[-1]['step'],peak_bytes=max(r['peak_memory_bytes'] for r in rows)))
c.update(basin_preflight_passed=True);formal.write_text(yaml.safe_dump(c,sort_keys=False));(J/'preflight_passed.json').write_text(json.dumps(dict(passed=True,ranks=rank_receipts,config_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),critic_sha256=c['basin_checkpoint_sha256'],resume_step=20000,scope='critic quality plus current-actor drift plus real batch32 gradients plus 8-GPU full-lambda short run'),indent=2))
rc=launch(formal,R/'runs/lip_v1_s0',J/'train.log');state('training_exited',returncode=rc)
if rc==0:subprocess.check_call([str(PY),str(C/'tools/verify_training_completion.py'),'--expected-steps','40000'],env=env)

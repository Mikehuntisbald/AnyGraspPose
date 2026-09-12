"""Resume only after the bounded-loader 8-rank probe passes; record cgroup memory."""
import hashlib,json,os,pathlib,subprocess,time
R=pathlib.Path('/mnt/why/dexycb_lip');os.chdir(R)
J=R/'runs/basin_memory_fix';env=os.environ.copy()
env.update(PYTHONPATH=str(J/'candidate/src'),TORCH_HOME=str(R/'cache/torch'),TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'),TMPDIR=str(R/'cache/tmp'),OMP_NUM_THREADS='2')
def status(phase,**kw):
 (J/'status.json').write_text(json.dumps(dict(phase=phase,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2))
def memory():
 p=pathlib.Path('/sys/fs/cgroup/memory');s=dict(x.split() for x in (p/'memory.stat').read_text().splitlines())
 r=dict(utc=time.time(),usage=int((p/'memory.usage_in_bytes').read_text()),limit=int((p/'memory.limit_in_bytes').read_text()),rss=int(s['rss']),shmem=int(s['shmem']))
 with (J/'memory.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 return r
status('waiting_probe')
while not (J/'probe.exit').exists():memory();time.sleep(10)
assert (J/'probe.exit').read_text().strip()=='0'
while not (J/'tests.log').exists() or 'passed' not in (J/'tests.log').read_text():time.sleep(2)
assert 'failed' not in (J/'tests.log').read_text() and 'ERROR' not in (J/'tests.log').read_text()
for rank in range(8):
 rows=[json.loads(x) for x in (J/f'probe/rank{rank}.jsonl').read_text().splitlines()]
 assert [r['step'] for r in rows]==list(range(21001,21031))
 assert {r['history_mode'] for r in rows}=={'lip_fp','noisy_gt','lip_only'}
 assert all(r['nonfinite_count']==0 and r['scheduler_step']==r['step']==r['sampler_position'] and r['basin_weight']==.01 for r in rows)
 assert all(abs(r['loss']-r['pose_loss']-.01*r['basin_loss'])<2e-6 for r in rows)
mem=memory();assert mem['usage']<mem['limit']*.75
(J/'preflight.json').write_text(json.dumps(dict(passed=True,steps_per_rank=30,world_size=8,config_sha256=hashlib.sha256((J/'config.yaml').read_bytes()).hexdigest(),memory=mem),indent=2))
cmd=[str(R/'.venv-fp/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','lip.train','--config',str(J/'config.yaml'),'--data-root',str(R/'cache/raw_full_20260910'),'--index-root','cache/dexycb_s0','--output','runs/lip_v1_s0','--resume',str(J/'probe/last.pt')]
with (J/'train.log').open('a') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 status('training',child_pid=p.pid,resume_step=21030)
 while p.poll() is None:memory();time.sleep(10)
 status('training_exited',returncode=p.returncode)

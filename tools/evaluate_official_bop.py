"""Run unmodified pinned official BOP scripts on disjoint scene shards, merge official matches."""
import argparse,json,os,signal,subprocess,sys,time,select
from pathlib import Path
import numpy as np
for name,value in [("float",float),("int",int),("bool",bool)]:np.__dict__.setdefault(name,value)


def start_private_display(log_path):
 """Xvfb chooses and locks its display atomically; avoid xvfb-run -a races."""
 read_fd,write_fd=os.pipe();log=Path(log_path).open('w')
 try:
  proc=subprocess.Popen(['Xvfb','-displayfd',str(write_fd),'-screen','0','640x480x24','-nolisten','tcp','-ac'],pass_fds=(write_fd,),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 finally:
  os.close(write_fd);log.close()
 try:
  if not select.select([read_fd],[],[],20)[0]:raise RuntimeError('Private Xvfb did not become ready')
  display=os.read(read_fd,128).decode().strip()
  if not display.isdigit() or proc.poll() is not None:raise RuntimeError('Private Xvfb startup failed')
  return proc,':'+display
 except BaseException:
  proc.terminate();proc.wait();raise
 finally:os.close(read_fd)


def main():
 p=argparse.ArgumentParser(__doc__)
 for n in ['toolkit','data-root','csv','out']:p.add_argument('--'+n,required=True,type=Path)
 p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);toolkit=a.toolkit.resolve();sys.path.insert(0,str(toolkit));os.environ['DEX_YCB_DIR']=str(a.data_root.resolve())
 from dex_ycb_toolkit.bop_eval import BOPEvaluator
 from bop_toolkit_lib import inout,misc
 from dex_ycb_toolkit.logging import get_logger
 evaluator=BOPEvaluator('s0_test')
 targets_path=a.data_root/'bop/s0/test_targets_bop19.json';targets=json.loads(targets_path.read_text())
 name='bop-'+a.csv.stem.replace('_','-')+'_s0-test';converted=a.out/(name+'.csv')
 estimates=inout.load_bop_results(str(a.csv));estimates=[evaluator._convert_pose_to_bop(x) for x in estimates];inout.save_bop_results(str(converted),estimates)
 env=dict(os.environ,BOP_PATH=str((a.data_root/'bop').resolve()),PYTHONPATH='.',PATH='/mnt/why/dexycb_lip/cache/official_benchmark/bin'+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LP_NUM_THREADS='2')
 receipt=dict(completed=False,phase='official_evaluation',workers=a.workers,commands=[],targets=len(targets),estimates=len(estimates),coordinate_conversion='official BOPEvaluator._convert_pose_to_bop exactly once',aggregation='concatenate disjoint official matches, then official _derive_bop_results; never average shard AR',toolkit_commit=subprocess.check_output(['git','-C',str(toolkit),'rev-parse','HEAD'],text=True).strip(),bop_commit=subprocess.check_output(['git','-C',str(toolkit/'bop_toolkit'),'rev-parse','HEAD'],text=True).strip())
 jobs=[];display_proc=None
 def save():(a.out/'status.json').write_text(json.dumps(receipt,indent=2))
 try:
  display_proc,display=start_private_display(a.out/'xvfb.log');env['DISPLAY']=display;env.pop('XAUTHORITY',None)
  receipt['display_policy']='One private Xvfb per method, atomically allocated via -displayfd; independent renderer contexts per shard'
  receipt['display']=display
  for rank in range(a.workers):
   selected=[t for t in targets if t['scene_id']%a.workers==rank];tp=a.out/f'targets_{rank}.json';tp.write_text(json.dumps(selected));dest=a.out/f'rank{rank}';dest.mkdir()
   cmd=['python','scripts/eval_bop19.py','--renderer_type=python','--result_filenames='+str(converted),'--results_path='+str(a.out),'--eval_path='+str(dest),'--targets_filename='+str(tp)]
   log=(a.out/f'rank{rank}.log').open('w');proc=subprocess.Popen(cmd,cwd=toolkit/'bop_toolkit',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   jobs.append((proc,log));receipt['commands'].append(dict(command=cmd,pid=proc.pid))
  save()
  while any(proc.poll() is None for proc,log in jobs):
   if any(proc.poll() not in (None,0) for proc,log in jobs):raise RuntimeError('Official evaluator shard failed; inspect rank logs')
   receipt['completed_shards']=sum(proc.poll()==0 for proc,log in jobs);save();time.sleep(5)
  for proc,log in jobs:assert proc.returncode==0;log.close()
  receipt['phase']='merging_matches';save()
  signatures=[]
  for error in evaluator._p['errors']:
   if error['type']=='vsd':signatures.extend(misc.get_error_signature('vsd',-1,vsd_delta=15,vsd_tau=t) for t in error['vsd_taus'])
   else:signatures.append(misc.get_error_signature(error['type'],-1))
  for sig in signatures:
   first=a.out/'rank0'/name/sig;destination=a.out/name/sig;destination.mkdir(parents=True)
   for f in sorted(first.glob('matches_*.json')):
    merged=[]
    for rank in range(a.workers):
     chunk=inout.load_json(str(a.out/f'rank{rank}'/name/sig/f.name));assert all(r['scene_id']%a.workers==rank for r in chunk);merged.extend(chunk)
    keys=[(r['scene_id'],r['im_id'],r['obj_id'],r['gt_id']) for r in merged];assert len(keys)==len(set(keys))
    inout.save_json(str(destination/f.name),merged)
  logger=get_logger(str(a.out/'official_results.log'))
  results={label:evaluator._derive_bop_results(str(a.out),name,grasp,logger) for label,grasp in [('all',False),('grasp_only',True)]}
  (a.out/'results.json').write_text(json.dumps(results,indent=2));receipt.update(completed=True,phase='completed',results=results);save()
 except BaseException as error:
  for proc,log in jobs:
   if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
  for proc,log in jobs:proc.wait();log.close()
  receipt.update(phase='failed',error=repr(error));save();raise
 finally:
  if display_proc is not None:
   display_proc.terminate();display_proc.wait()

if __name__=='__main__':main()

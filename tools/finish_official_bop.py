"""Verify complete paired CSVs and schedule official scoring after inference finishes."""
import argparse,csv,json,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from lip.benchmark.bop_io import load_predictions,select_initializers,key


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--toolkit',required=True,type=Path);p.add_argument('--bop-python',required=True);a=p.parse_args()
 out=a.run.resolve();root=Path(__file__).resolve().parents[1];protocol=json.loads((out/'protocol.json').read_text());launch=json.loads((out/'launch.json').read_text())
 receipt=dict(phase='inference',protocol_sha256=sha(out/'protocol.json'),methods=protocol['methods'])
 def save():(out/'status.json').write_text(json.dumps(receipt,indent=2))
 try:
  while True:
   manifests=[];progress=[]
   for item in launch:
    rank=item['rank'];f=out/f'rank{rank}/manifest.json';m=json.loads(f.read_text()) if f.exists() else {};manifests.append(m)
    events=out/f'rank{rank}/events.jsonl';progress.append(sum(1 for _ in events.open()) if events.exists() else 0)
    if not m.get('completed'):
     cmd=Path(f"/proc/{item['pid']}/cmdline")
     if not cmd.exists() or b'infer_official_bop.py' not in cmd.read_bytes():raise RuntimeError(f'Inference rank {rank} exited without completion; inspect its log')
   receipt['predictions_per_rank']=progress;receipt['completed_predictions']=sum(progress);save()
   if all(m.get('completed') for m in manifests):break
   time.sleep(10)
  assert all(m['protocol_sha256']==receipt['protocol_sha256'] for m in manifests)
  rows,missing=select_initializers(load_predictions(protocol['initializer_csv']),json.loads(Path(protocol['targets']).read_text()))
  expected=Counter(key(r) for r in rows);methods=[m for m in protocol['methods'] if m!='posecnn']
  merged=out/'csv';merged.mkdir()
  for method in protocol['methods']:
   values=[]
   for rank,m in enumerate(manifests):
    path=out/f'rank{rank}'/(method+'.csv');assert sha(path)==m['csv_sha256'][method]
    with path.open() as f:values.extend(csv.DictReader(f))
   actual=Counter((int(r['scene_id']),int(r['im_id']),int(r['obj_id'])) for r in values);assert actual==expected
   values.sort(key=lambda r:(int(r['scene_id']),int(r['im_id']),int(r['obj_id']),-float(r['score'])))
   with (merged/(method+'.csv')).open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['scene_id','im_id','obj_id','score','R','t','time']);writer.writeheader();writer.writerows(values)
  receipt.update(phase='official_evaluation',estimates=len(rows),missing=sum(x['missing'] for x in missing),csv_sha256={m:sha(merged/(m+'.csv')) for m in protocol['methods']});save()
  methods+=['deepim_rgbd_reference'];jobs=[]
  for method in methods:
   source=Path('/mnt/why/dexycb_lip/cache/official_benchmark/bop_deepim_s0_test_RGBD.csv') if method=='deepim_rgbd_reference' else merged/(method+'.csv')
   command=[a.bop_python,str(root/'tools/evaluate_official_bop.py'),'--toolkit',str(a.toolkit),'--data-root',protocol['data_root'],'--csv',str(source),'--out',str(out/method),'--workers','8']
   log=(out/(method+'_evaluation.log')).open('w');proc=subprocess.Popen(command,cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   jobs.append((method,proc,log));receipt.setdefault('evaluation_commands',[]).append(dict(method=method,command=command,pid=proc.pid));save()
  for method,proc,log in jobs:
   code=proc.wait();log.close()
   if code:raise RuntimeError('Official evaluation failed: '+method)
  baseline=root/'runs/official_posecnn_final'
  while not (baseline/'results.json').exists():
   status=json.loads((baseline/'status.json').read_text())
   if status['phase']=='failed':raise RuntimeError('Official PoseCNN reproduction failed')
   time.sleep(10)
  results={'posecnn':json.loads((baseline/'results.json').read_text())}
  results.update({m:json.loads((out/m/'results.json').read_text()) for m in methods})
  (out/'comparison.json').write_text(json.dumps(dict(completed=True,protocol=protocol,results=results),indent=2))
  lines=['# Official DexYCB s0 test BOP results','','Frozen residual step 1,000; each target image is independent. PoseCNN supplies predicted poses, no GT pose/mask enters inference. Missing initial detections remain failures. All methods are scored by the pinned official evaluator; original-to-BOP coordinates are converted once.','','| Method | All AR | Grasped AR | Grasped VSD | Grasped MSSD | Grasped MSPD |','|---|---:|---:|---:|---:|---:|']
  for m,r in results.items():
   g=r['grasp_only'];lines.append(f"| {m} | {r['all']['mean']:.3f} | {g['mean']:.3f} | {g['vsd']:.3f} | {g['mssd']:.3f} | {g['mspd']:.3f} |")
  lines+=['','All table values are percentages. The three new pipelines use identical PoseCNN initial candidates and RGB-D inputs; FP uses two refinement iterations. PoseCNN alone is an RGB initializer reference. DeepIM RGB-D is the released 2021 comparison, not a claim about the latest SOTA. Training data and model budgets differ. This cold-start single-image use of the temporal LIP network must not be confused with GT-initialized tracking.','','Raw input CSVs are in csv/ using DexYCB original mesh coordinates (t in mm); standard BOP model-coordinate CSVs are in each official evaluation directory. Predictions, missing counts, protocol hashes, rank logs, and official match files are retained.']
  (out/'report.md').write_text('\n'.join(lines)+'\n');receipt['phase']='completed';save()
 except BaseException as error:
  receipt.update(phase='failed',error=repr(error));save();raise

if __name__=='__main__':main()

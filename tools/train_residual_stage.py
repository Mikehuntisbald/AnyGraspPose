"""Train only the added cross branch, then evaluate its gain over a verified parent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config,verify_preflight
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import sha,source_hash


def parent_unchanged(parent_path,child_path):
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    child=torch.load(child_path,map_location='cpu',weights_only=False)
    changed=[n for n,t in parent['model'].items() if n not in child['model'] or not torch.equal(t,child['model'][n])]
    if changed:raise RuntimeError('Parent tensors changed: '+repr(changed))
    return dict(passed=True,checked_tensors=len(parent['model']),parent_sha256=sha(parent_path),child_sha256=sha(child_path),stage_step=child['new_stage_step'])


def compare(evaluation,baseline,out):
    read=lambda p:json.loads(p.read_text())
    a=read(baseline/'manifest.json');b=read(evaluation/'manifest.json')
    for m in [a,b]:assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
    assert all(a[k]==b[k] for k in ['split_hash','mesh_hash','split'])
    keys=lambda folder:{(r['stream_id'],r['frame_index']) for r in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())}
    assert keys(baseline)==keys(evaluation)
    old=read(baseline/'metrics.json')['excluding_initialization']['macro_object'];new=read(evaluation/'metrics.json')['excluding_initialization']['macro_object']
    result=dict(completed=True,frames=23200,streams=320,baseline=old,current=new,delta={k:new[k]-old[k] for k in new},
        parent_checkpoint_sha256=a['checkpoint_sha256'],checkpoint_sha256=b['checkpoint_sha256'],source_sha256=source_hash(),
        scope='Frozen-parent added-branch experiment; object macro, initialization excluded; positive ADD/ADD-S delta is better')
    out.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','init','parent','baseline','equivalence','data-root','index-root','out']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    c=load_stream_config(a.config);audit=check_data_gate(a.index_root);verify_preflight(c,audit)
    if not 0<a.steps<=c.get('freeze_parent_steps',0):raise ValueError('This experiment must stay within the frozen-parent phase')
    eq=json.loads(a.equivalence.read_text())
    assert eq['passed'] and eq['frames']==23200 and eq['parent_checkpoint_sha256']==sha(a.parent)
    assert eq['init_checkpoint_sha256']==sha(a.init) and eq['source_sha256']==source_hash()
    parent_unchanged(a.parent,a.init)
    a.out.mkdir(parents=True,exist_ok=False);env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    train=a.out/'train';evaluation=a.out/'s0_val'
    receipt=dict(phase='training',steps=a.steps,source_sha256=source_hash(),init_sha256=sha(a.init),parent_sha256=sha(a.parent),commands=[])
    def save(): (a.out/'status.json').write_text(json.dumps(receipt,indent=2))
    def run(command,name):
        receipt['commands'].append(command);save()
        with (a.out/name).open('w') as log:subprocess.run(command,env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        save();run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','lip.train_stream',
            '--config',str(a.config),'--init-from',str(a.init),'--output',str(train),'--data-root',str(a.data_root),'--index-root',str(a.index_root),'--max-steps',str(a.steps)],'training.log')
        check=parent_unchanged(a.parent,train/'last.pt');assert check['stage_step']==a.steps
        (a.out/'parent_unchanged.json').write_text(json.dumps(check,indent=2))
        receipt['phase']='evaluating';save()
        run([sys.executable,str(root/'tools/evaluate_saved_stream.py'),'--config',str(a.config),'--checkpoint',str(train/'last.pt'),
             '--data-root',str(a.data_root),'--index-root',str(a.index_root),'--out',str(evaluation)],'evaluation.log')
        result=compare(evaluation,a.baseline,a.out/'comparison.json')
        lines=['# Frozen-parent cross residual: full s0 validation','','Parent tensors were verified bitwise unchanged after training. Only the additional attention and its gate were optimized.','','| Metric | Parent | Added cross | Delta |','|---|---:|---:|---:|']
        for key in ['add_01','adds_01','center_mm','rotation_deg']:
            lines.append(f"| {key} | {result['baseline'][key]:.6f} | {result['current'][key]:.6f} | {result['delta'][key]:+.6f} |")
        lines+=['','320 streams / 23,200 frames, object-macro aggregation excluding initialization. Success rates above are fractions. Same frame population, initial GT only, no FP. This is one training seed; no architecture-wide superiority is implied.']
        (a.out/'report.md').write_text('\n'.join(lines)+'\n');receipt['phase']='completed';receipt['comparison']=result;save()
    except BaseException as exc:
        receipt.update(phase='failed',error=repr(exc));save();raise

if __name__=='__main__':main()

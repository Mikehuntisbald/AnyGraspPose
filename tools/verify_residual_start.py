"""Verify real full-validation function preservation before training a new branch."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash
from train_residual_stage import parent_unchanged


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['baseline','evaluation','parent','init','out']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();read=lambda p:json.loads(p.read_text())
    old=read(a.baseline/'manifest.json');new=read(a.evaluation/'manifest.json')
    for m in [old,new]:assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
    assert old['checkpoint_sha256']==sha(a.parent) and new['checkpoint_sha256']==sha(a.init)
    assert new['source_sha256']==source_hash()
    assert all(old[k]==new[k] for k in ['split','split_hash','mesh_hash'])
    rows=lambda root:{(r['stream_id'],r['frame_index']):r for r in map(json.loads,(root/'predictions.jsonl').read_text().splitlines())}
    x=rows(a.baseline);y=rows(a.evaluation);assert len(x)==len(y)==23200 and x.keys()==y.keys()
    maximum=0.
    for key in x:
        assert x[key]['status']==y[key]['status'] and x[key]['initialization']==y[key]['initialization']
        maximum=max(maximum,float(np.max(np.abs(np.asarray(x[key]['pose_centered'])-np.asarray(y[key]['pose_centered'])))))
    assert maximum==0.,f'Migration changed real closed-loop predictions: {maximum}'
    old_metrics=read(a.baseline/'metrics.json')['excluding_initialization']['macro_object'];new_metrics=read(a.evaluation/'metrics.json')['excluding_initialization']['macro_object'];assert old_metrics==new_metrics
    report=dict(passed=True,frames=23200,streams=320,max_abs_pose_difference=maximum,metrics=new_metrics,
        parent_checkpoint_sha256=sha(a.parent),init_checkpoint_sha256=sha(a.init),source_sha256=source_hash(),
        parent_tensors=parent_unchanged(a.parent,a.init),predictions_sha256=dict(parent=sha(a.baseline/'predictions.jsonl'),initial=sha(a.evaluation/'predictions.jsonl')))
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

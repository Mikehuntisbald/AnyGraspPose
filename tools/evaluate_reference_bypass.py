"""Frozen streaming evaluation with explicit learned/zero reference coefficients.

Zero bypasses either reference feedback or reference writing, retaining its
computation and nonvisual cache. This isolates output use, not runtime savings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch


def install_mode(model,mode,component='feedback',zero_channels='both'):
    if mode not in ('learned','zero'):raise ValueError('Explicit learned or zero mode required')
    if component not in ('feedback','writer'):raise ValueError('Explicit reference component required')
    if zero_channels not in ('both','rotation','center'):raise ValueError('Explicit zero channels required')
    if zero_channels!='both' and (component!='writer' or mode!='zero'):raise ValueError('Partial zero channels require zero writer mode')
    if model.architecture_id not in ('stream_rk_pose_reference','stream_rk_adaptive_reference'):raise ValueError('Requires a pose-reference checkpoint')
    if component=='writer' and model.architecture_id!='stream_rk_adaptive_reference':raise ValueError('Reference writer requires an adaptive checkpoint')
    if mode=='learned':return None
    def zero_coefficients(module,args,result):
        if result.ndim!=2 or result.shape[-1]!=2:raise ValueError('Unexpected reference coefficient shape')
        if zero_channels=='both':return torch.zeros_like(result)
        masked=result.clone();masked[:,0 if zero_channels=='rotation' else 1]=0
        return masked
    module=model.pose_reference_feedback if component=='feedback' else model.reference_writer
    return module.register_forward_hook(zero_coefficients)


def checked_shards(root,world,mode,component='feedback',zero_channels='both'):
    if world<1:raise ValueError('Positive shard count required')
    receipts=[]
    for rank in range(world):
        folder=root/f'rank{rank}';receipt=json.loads((folder/'intervention.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        if not receipt['completed'] or receipt['mode']!=mode:raise ValueError('Incomplete or mixed intervention mode')
        if receipt.get('component','feedback')!=component:raise ValueError('Mixed intervention component')
        if receipt.get('zero_channels','both')!=zero_channels:raise ValueError('Mixed intervention channels')
        if not receipt['weights_bitwise_unchanged'] or not receipt['source_and_intervention_bound']:raise ValueError('Unverified frozen intervention')
        if manifest.get('inference_intervention')!=receipt:raise ValueError('Intervention receipt differs from manifest')
        digest=hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest()
        if digest!=receipt['predictions_sha256']:raise ValueError('Intervention predictions changed')
        receipts.append(receipt)
    for receipt in receipts:
        if any(receipt[k]!=receipts[0][k] for k in ('mode','checkpoint_sha256','base_source_sha256','wrapper_sha256')):
            raise ValueError('Incompatible intervention sources')
    return receipts


def main():
    p=argparse.ArgumentParser(__doc__,add_help=False)
    p.add_argument('--runtime',required=True,type=Path);p.add_argument('--mode',required=True,choices=('learned','zero'))
    p.add_argument('--component',choices=('feedback','writer'),default='feedback')
    p.add_argument('--zero-channels',choices=('both','rotation','center'),default='both')
    a,remaining=p.parse_known_args()
    if a.zero_channels!='both' and (a.component!='writer' or a.mode!='zero'):p.error('Partial zero channels require zero writer mode')
    def option(name):
        if name not in remaining:raise ValueError('Required evaluation flag '+name)
        return remaining[remaining.index(name)+1]
    out=Path(option('--out'))
    sys.path.insert(0,str(a.runtime/'src'))
    import lip.evaluate_stream as evaluation
    from lip.engine.stream_checkpoint import source_hash,sha
    if '--merge' in remaining:
        if (out/'predictions.jsonl').exists():raise FileExistsError(out/'predictions.jsonl')
        world=int(option('--world-size'));receipts=checked_shards(out,world,a.mode,a.component,a.zero_channels)
        evaluation.merge_shards(out,world)
        merged=dict(receipts[0],world_size=world,shard_predictions_sha256=[r['predictions_sha256'] for r in receipts],
            predictions_sha256=sha(out/'predictions.jsonl'),observed_coefficient_rows=sum(r['observed_coefficient_rows'] for r in receipts),merge_wrapper_sha256=sha(__file__))
        path=out/'manifest.json';manifest=json.loads(path.read_text());manifest['inference_intervention']=merged
        path.write_text(json.dumps(manifest,indent=2));(out/'intervention.json').write_text(json.dumps(merged,indent=2))
        print(json.dumps(merged,indent=2));return
    checkpoint=Path(option('--checkpoint'))
    if (out/'predictions.jsonl').exists() or (out/'intervention.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    receipt=dict(completed=False,mode=a.mode,component=a.component,zero_channels=a.zero_channels,checkpoint_sha256=sha(checkpoint),base_source_sha256=source_hash(),
        wrapper_sha256=sha(__file__),runtime=str(a.runtime.resolve()),scope=__doc__)
    (out/'intervention.json').write_text(json.dumps(receipt,indent=2))
    original_factory=evaluation.make_model;models=[];handles=[]
    def factory(config):
        model=original_factory(config);models.append(model);handles.append(install_mode(model,a.mode,a.component,a.zero_channels));return model
    evaluation.make_model=factory;old_argv=sys.argv;sys.argv=[old_argv[0],*remaining]
    try:
        evaluation.main()
        assert len(models)==1
        expected=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
        actual=models[0].state_dict()
        assert set(expected)==set(actual) and all(torch.equal(t,actual[name].detach().cpu()) for name,t in expected.items())
        manifest_path=out/'manifest.json';manifest=json.loads(manifest_path.read_text())
        assert manifest['completed'] and manifest['checkpoint_sha256']==receipt['checkpoint_sha256']
        assert manifest['fp_calls']==manifest['critic_calls']==0 and manifest['source_sha256']==receipt['base_source_sha256']
        rows=list(map(json.loads,(out/'predictions.jsonl').read_text().splitlines()))
        fields=(('reference_rotation_coefficient','reference_center_coefficient','reference_rotation_residual_norm','reference_center_residual_norm') if a.component=='feedback' else
            ('reference_write_rotation_coefficient','reference_write_center_coefficient','reference_write_rotation_norm','reference_write_center_norm'))
        observed=[r for r in rows if not r['initialization'] and fields[0] in r]
        if a.mode=='zero':
            selected=fields if a.zero_channels=='both' else tuple(fields[i] for i in ((0,2) if a.zero_channels=='rotation' else (1,3)))
            assert observed and all(r[k]==0 for r in observed for k in selected)
        receipt.update(completed=True,weights_bitwise_unchanged=True,observed_coefficient_rows=len(observed),
            predictions_sha256=hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest(),
            source_and_intervention_bound=True)
        manifest['inference_intervention']=receipt.copy()
        manifest_path.write_text(json.dumps(manifest,indent=2));(out/'intervention.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt,indent=2))
    finally:
        sys.argv=old_argv;evaluation.make_model=original_factory
        for handle in handles:
            if handle is not None:handle.remove()


if __name__=='__main__':main()

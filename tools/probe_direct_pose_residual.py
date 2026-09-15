"""Bounded real-train closed-loop check of a directly supervised residual head."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.models.direct_pose_residual import DirectPoseResidual,apply_direct_residual
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.engine.stream_training import StreamTrainingModule
from lip.engine.config import check_data_gate
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('parent-experiment','manifests','data-root','index-root','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--integrated',action='store_true',help='Exercise the registered DirectPoseRKTracker rather than the earlier hook prototype')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42);a.out.mkdir(parents=True,exist_ok=False)
    e=json.loads((a.parent_experiment/'experiment.json').read_text());parent=a.parent_experiment/'spatial/train/last.pt'
    expected='3371614e5efa788bf8c64cd10b64498040edf68a49d246c136d779e096ee0f2c';assert sha(parent)==expected
    c=load_stream_config(e['arms']['spatial']['config']);model=make_model(c).cuda().eval();load_init(parent,model,check_data_gate(a.index_root),c)
    for parameter in model.parameters():parameter.requires_grad_(False)
    parent_parameters={name:p.detach().clone() for name,p in model.state_dict().items()}
    manifests={name:json.loads((a.manifests/(name+'.json')).read_text()) for name in ('control','natural')}
    selections=list(map(json.loads,(a.manifests/'selections.jsonl').read_text().splitlines()))
    ids=[i for i,row in enumerate(selections) if row['hard_selected']][:2]
    sets={name:StreamClips(a.data_root,a.index_root,8,48,fixed=manifest) for name,manifest in manifests.items()}
    groups={'uniform':[sets['control'][i] for i in (0,1)],'natural_hard':[sets['natural'][i] for i in ids]}
    renderer=Renderer('cuda');baseline={}
    with torch.no_grad():
        for name,samples in groups.items():
            for precision in ('fp32','bf16'):
                baseline[(name,precision)]=StreamTrainingModule(model,dict(c,precision=precision),renderer)(samples,True)['predictions'].cpu()
    if a.integrated:
        c=dict(c,architecture_id='stream_rk_direct_pose');model=make_model(c).cuda().eval()
        loaded=model.load_state_dict(parent_parameters,strict=False)
        assert not loaded.unexpected_keys and all(k.startswith('direct_pose_residual.') for k in loaded.missing_keys)
        for name,parameter in model.named_parameters():parameter.requires_grad_(name.startswith('direct_pose_residual.'))
    else:model.add_module('direct_pose_residual',DirectPoseResidual().cuda())
    def hook(module,inputs,result):
        out,cache=result;aux=module.direct_pose_residual(out['latent'])
        return apply_direct_residual(out,inputs[0],aux),cache
    handle=None if a.integrated else model.register_forward_hook(hook);equivalence={}
    with torch.no_grad():
        for name,samples in groups.items():
            equivalence[name]={}
            for precision in ('fp32','bf16'):
                prediction=StreamTrainingModule(model,dict(c,precision=precision),renderer)(samples,True)['predictions'].cpu()
                assert torch.equal(prediction,baseline[(name,precision)])
                equivalence[name][precision]=dict(bitwise_equal=True,frames=prediction.shape[0]*prediction.shape[1])
    runner=StreamTrainingModule(model,c,renderer);model.train();optimizer=torch.optim.AdamW(model.direct_pose_residual.parameters(),lr=5e-5,weight_decay=.05)
    rows=[]
    # Four pilot steps only; distinct preset uniform and naturally occluded
    # fragments alternate. This is not an overfit benchmark or a val selection.
    for step,name in enumerate(('uniform','natural_hard','uniform','natural_hard'),1):
        optimizer.zero_grad(set_to_none=True);out=runner(groups[name],True)
        out['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.direct_pose_residual.parameters(),1.)
        assert torch.isfinite(grad) and grad>0
        for branch in (model.direct_pose_residual.rotation,model.direct_pose_residual.center):
            assert branch[-1].weight.grad is not None and torch.isfinite(branch[-1].weight.grad).all() and branch[-1].weight.grad.abs().sum()>0
        assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('direct_pose_residual.'))
        assert torch.isfinite(out['predictions']).all() and torch.isfinite(out['loss'])
        rotation=out['predictions'][...,:3,:3]
        assert (rotation.transpose(-1,-2)@rotation-torch.eye(3,device='cuda')).abs().max()<1e-3
        assert (out['predictions'][...,2,3]>.01).all()
        optimizer.step();rows.append(dict(step=step,population=name,loss=float(out['loss'].detach()),grad_norm=float(grad),supervised_frames=out['supervised_frames']))
    assert all(torch.equal(tensor,model.state_dict()[name]) for name,tensor in parent_parameters.items())
    assert model.direct_pose_residual.rotation[-1].weight.abs().sum()>0 and model.direct_pose_residual.center[-1].weight.abs().sum()>0
    torch.save(dict(state_dict=model.direct_pose_residual.state_dict(),parent_sha256=expected,prototype_only=True),a.out/'prototype_head.pt')
    report=dict(passed=True,scope=('Registered DirectPoseRKTracker, bounded integration check.' if a.integrated else 'Isolated prototype via a forward hook.')+' Four real train fragments, four head-only pilot optimizer steps; no val/test or FP. No accuracy-gain or full-DDP-preflight claim.',
        source_sha256=source_hash(),parent_sha256=expected,config=c,equivalence=equivalence,train_steps=rows,all_parent_tensors_unchanged=True,
        rotation_and_center_output_gradients_nonzero=True,real_zero_residual_equivalence_frames=224,
        samples={name:[s['sample'] for s in samples] for name,samples in groups.items()},
        precision='Network BF16 as in parent; pose update and loss FP32',
        head_sha256=sha(a.out/'prototype_head.pt'),architecture_registered=a.integrated,formal_training_approved=False)
    if handle is not None:handle.remove()
    (a.out/'probe.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2))


if __name__=='__main__':main()

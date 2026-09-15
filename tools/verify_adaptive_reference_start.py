"""Real S1O1 equivalence, future write credit, and a separate tiny optimizer probe."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config,optimizer_and_scheduler
from lip.engine.stream_checkpoint import source_hash,sha
from lip.engine.stream_training import StreamTrainingModule,supervision_positions
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args()
    e=json.loads((a.experiment/'experiment.json').read_text());torch.set_num_threads(2)
    assert source_hash()==e['source_sha256'] and sha(e['parent'])==e['parent_sha256']
    ds=StreamClips(e['data_root'],e['index_root'],8,48,fixed=json.loads(Path(e['training_manifest']).read_text()))
    samples=[ds[i] for i in range(2)];renderer=Renderer('cuda');models={};configs={};results={}
    for name in ('control','adaptive_reference'):
        c=load_stream_config(e['arms'][name]['config']);model=make_model(c).cuda().eval()
        checkpoint=torch.load(e['arms'][name]['init'],map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);model.migration_status=checkpoint['migration_status'];models[name]=model;configs[name]=c
    for precision in ('fp32','bf16'):
        outputs={}
        for name,model in models.items():
            with torch.no_grad():outputs[name]=StreamTrainingModule(model,dict(configs[name],precision=precision),renderer)(samples,True)
        assert torch.equal(outputs['control']['predictions'],outputs['adaptive_reference']['predictions'])
        assert torch.equal(outputs['control']['loss'],outputs['adaptive_reference']['loss'])
        results[precision]=dict(bitwise_equal=True,frames=112,loss_bitwise_equal=True)
    model=models['adaptive_reference'];c=configs['adaptive_reference'];model.train();gates=[];poses=[]
    def capture_gate(module,args,result):result.retain_grad();gates.append(result)
    def capture_pose(module,args,result):
        pose=result[0]['pose_centered']
        if pose.requires_grad:pose.retain_grad()
        poses.append(pose)
    hooks=[model.reference_writer.register_forward_hook(capture_gate),model.register_forward_hook(capture_pose)]
    out=StreamTrainingModule(model,c,renderer)(samples,True);out['loss'].backward()
    for hook in hooks:hook.remove()
    selected=supervision_positions(8,48,c['startup_supervision_frames']);assert len(poses)==len(gates)==56
    for i,(pose,supervised) in enumerate(zip(poses,selected)):
        if supervised:assert pose.grad is not None and torch.isfinite(pose.grad).all() and pose.grad.abs().sum()>0,i
        else:assert pose.grad is None or pose.grad.count_nonzero()==0,i
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert gates[0].grad is not None and (gates[0].grad.abs().sum(0)>0).all()
    assert gates[7].grad is None and gates[-1].grad is None
    assert out['reference_has_training_graph']
    gradients=dict(first_write_grad_abs=gates[0].grad.abs().sum(0).tolist(),boundary_write_has_no_future_gradient=True,
        historical_actor_pose_detached=True,reference_state_has_training_graph=True,all_trainable_gradients_finite=True)
    # These three tiny updates are discarded; the launcher starts from init.pt.
    initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    optimizer,scheduler=optimizer_and_scheduler(model,c);pilot=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);out=StreamTrainingModule(model,c,renderer)(samples)
        out['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip_norm'])
        assert torch.isfinite(norm);optimizer.step();scheduler.step()
        pilot.append(dict(step=step+1,loss=float(out['loss']),grad_norm=float(norm),write_diagnostics=out['reference_write_diagnostics'].tolist()))
    assert any(row['write_diagnostics'][0]>0 for row in pilot[1:]) and any(row['write_diagnostics'][1]>0 for row in pilot[1:])
    trainable=set(e['arms']['adaptive_reference']['trainable'])
    assert all(torch.equal(t,model.state_dict()[n].detach().cpu()) for n,t in initial.items() if n not in trainable)
    result=dict(passed=True,parent_sha256=e['parent_sha256'],source_sha256=source_hash(),results=results,gradient_routing=gradients,
        pilot=pilot,pilot_weights_discarded=True,scope='Two real train fragments, configured startup supervision and synthetic occlusion. Exact parent preservation and future-reference gradients; no validation or benefit claim.')
    (a.experiment/'equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

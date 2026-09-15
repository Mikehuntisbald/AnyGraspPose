"""Real fragment parent equivalence and first-update gradient checks for one arm."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import sha,source_hash
from lip.engine.stream_training import StreamTrainingModule,supervision_positions
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer
from prepare_startup_factorial import ARMS


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args()
    torch.set_num_threads(2);e=json.loads((a.experiment/'experiment.json').read_text());entry=e['arms'][a.arm]
    assert source_hash()==e['source_sha256'] and sha(e['parent'])==e['parent_sha256']
    c=load_stream_config(entry['config']);parent_obj=torch.load(e['parent'],map_location='cpu',weights_only=False)
    parent=make_model(parent_obj['config']).cuda().eval();parent.load_state_dict(parent_obj['model'])
    model=make_model(c).cuda().eval();model.load_state_dict(torch.load(entry['init'],map_location='cpu',weights_only=False)['model'])
    dataset=StreamClips(e['data_root'],e['index_root'],8,48,fixed=json.loads(Path(e['training_manifest']).read_text()));samples=[dataset[i] for i in (0,1)]
    renderer=Renderer('cuda');results={}
    # Match this arm's images and loss positions in the parent runner. The
    # startup augmentation arms intentionally do not share images with O0.
    for precision in ('fp32','bf16'):
        cfg=dict(c,precision=precision)
        with torch.no_grad():
            before=StreamTrainingModule(parent,cfg,renderer)(samples,True)
            after=StreamTrainingModule(model,cfg,renderer)(samples,True)
        assert torch.equal(before['predictions'],after['predictions']) and torch.equal(before['loss'],after['loss'])
        prediction=after['predictions'];rotation=prediction[...,:3,:3]
        assert torch.isfinite(prediction).all() and (prediction[...,2,3]>.01).all()
        assert (rotation.transpose(-1,-2)@rotation-torch.eye(3,device='cuda')).abs().max()<1e-3
        results[precision]=dict(frames=112,poses_bitwise_equal=True,loss_bitwise_equal=True)
    del parent;captures=[]
    def capture(module,args,result):
        pose=result[0]['pose_centered']
        if pose.requires_grad:pose.retain_grad()
        captures.append(pose)
    model.train();hook=model.register_forward_hook(capture)
    out=StreamTrainingModule(model,c,renderer)(samples,True);out['loss'].backward();hook.remove()
    selected=supervision_positions(8,48,c['startup_supervision_frames'])
    assert len(captures)==56 and out['supervised_frames']==96
    for i,(pose,supervised) in enumerate(zip(captures,selected)):
        if supervised:assert pose.grad is not None and torch.isfinite(pose.grad).all() and pose.grad.abs().sum()>0,(i,'missing direct pose loss')
        else:assert pose.grad is None or pose.grad.count_nonzero()==0,(i,'unexpected direct pose loss')
    assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    head_grad=model.pose_reference_feedback.readout[-1].weight.grad
    assert (head_grad.abs().sum(1)>0).all()
    result=dict(passed=True,arm=a.arm,parent_sha256=e['parent_sha256'],source_sha256=source_hash(),
        fp32_bf16_equivalence=results,real_gradient_routing_checked=True,supervised_observation_indices=[i for i,v in enumerate(selected) if v],
        startup_supervised_frames=out['startup_supervised_frames'],omitted_late_targets=out['omitted_late_targets'],
        all_trainable_gradients_finite=True,rotation_center_feedback_gradients_nonzero=True,
        no_optimizer_step=True,scope='Two actual train fragments, matched input augmentation and loss positions. No val/test accuracy or benefit claim.')
    (a.experiment/a.arm/'equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

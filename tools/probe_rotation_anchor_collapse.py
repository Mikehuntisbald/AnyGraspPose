"""Read-only final-checkpoint probe of clamp saturation on real train fragments."""
import argparse
import json
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','experiment','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.runtime/'src'))
    from lip.engine.stream_config import make_model,load_stream_config
    from lip.engine.stream_checkpoint import source_hash,sha
    from lip.engine.stream_training import StreamTrainingModule
    from lip.data.stream_clips import StreamClips
    from lip.geometry.renderer import Renderer
    torch.set_num_threads(2);torch.manual_seed(42)
    e=json.loads((a.experiment/'experiment.json').read_text());assert source_hash()==e['source_sha256']
    checkpoint=a.experiment/'rotation_anchor/train/last.pt';before=sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['new_stage_step']==1000
    c=load_stream_config(e['arms']['rotation_anchor']['config']);model=make_model(c).cuda().eval()
    model.load_state_dict(state['model']);raw=[];read=[]
    def hook(values):
        def capture(module,args,result):result.retain_grad();values.append(result)
        return capture
    hooks=[model.rotation_anchor_readout.readout[-1].register_forward_hook(hook(raw)),
           model.rotation_anchor_readout.register_forward_hook(hook(read))]
    ds=StreamClips(e['data_root'],e['index_root'],8,48,fixed=json.loads(Path(e['training_manifest']).read_text()))
    out=StreamTrainingModule(model,c,Renderer('cuda'))([ds[0],ds[1]])
    out['loss'].backward()
    for h in hooks:h.remove()
    logits=torch.cat([v.detach().float().flatten() for v in raw]);fractions=torch.cat([v.detach().float().flatten() for v in read])
    def grad_sum(values):return sum(float(v.grad.abs().sum()) for v in values if v.grad is not None)
    assert len(logits)==len(fractions)==112 and torch.isfinite(logits).all()
    assert all(torch.equal(value,model.state_dict()[key].cpu()) for key,value in state['model'].items())
    assert sha(checkpoint)==before
    result=dict(completed=True,checkpoint_sha256=before,source_sha256=source_hash(),script_sha256=sha(Path(__file__)),
        scope='Two real train clips with existing S1O1 augmentation, 112 observations. Eval-mode forward and backward only, no optimizer step or test-set access. Local gradient evidence, not proof of why every training sample favored this state.',
        frames=112,logit_min=float(logits.min()),logit_max=float(logits.max()),logit_mean=float(logits.mean()),
        negative_logits=int((logits<0).sum()),positive_fractions=int((fractions>0).sum()),
        loss_gradient_wrt_read_fraction_l1=grad_sum(read),loss_gradient_wrt_raw_logit_l1=grad_sum(raw),
        readout_weight_gradient_l1=float(model.rotation_anchor_readout.readout[-1].weight.grad.abs().sum()),
        model_state_bitwise_unchanged=True,checkpoint_file_unchanged=True)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

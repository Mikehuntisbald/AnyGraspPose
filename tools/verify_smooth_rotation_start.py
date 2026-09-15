"""Real parent equality and negative-logit recovery before fixed-budget training."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config,optimizer_and_scheduler
from lip.engine.stream_checkpoint import source_hash,sha
from lip.engine.stream_training import StreamTrainingModule
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args()
    e=json.loads((a.experiment/'experiment.json').read_text());torch.set_num_threads(2)
    assert e['reader_arm']=='smooth_rotation' and source_hash()==e['source_sha256']
    ds=StreamClips(e['data_root'],e['index_root'],8,48,fixed=json.loads(Path(e['training_manifest']).read_text()))
    samples=[ds[0],ds[1]];renderer=Renderer('cuda');models={};configs={};results={}
    for name in ('control','smooth_rotation'):
        c=load_stream_config(e['arms'][name]['config']);model=make_model(c).cuda().eval()
        ck=torch.load(e['arms'][name]['init'],map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);model.migration_status=ck['migration_status'];models[name]=model;configs[name]=c
    for precision in ('fp32','bf16'):
        predictions={}
        for name,model in models.items():
            with torch.no_grad():predictions[name]=StreamTrainingModule(model,dict(configs[name],precision=precision),renderer)(samples,True)
        assert torch.equal(predictions['control']['predictions'],predictions['smooth_rotation']['predictions'])
        assert torch.equal(predictions['control']['loss'],predictions['smooth_rotation']['loss'])
        results[precision]=dict(frames=112,predictions_bitwise_equal=True,loss_bitwise_equal=True)
    model=models['smooth_rotation'];c=configs['smooth_rotation'];model.train();model.zero_grad()
    model.rotation_anchor_readout.readout[-1].bias.data.fill_(-.001)
    raw=[];writes=[];reads=[]
    def capture(values):
        def hook(module,args,result):result.retain_grad();values.append(result)
        return hook
    hs=[model.rotation_anchor_readout.readout[-1].register_forward_hook(capture(raw)),
        model.rotation_anchor_readout.register_forward_hook(capture(reads)),model.reference_writer.register_forward_hook(capture(writes))]
    out=StreamTrainingModule(model,c,renderer)(samples);out['loss'].backward()
    for h in hs:h.remove()
    assert len(raw)==56 and all((v.detach()<0).all() for v in raw)
    grad=lambda values:sum(float(v.grad.abs().sum()) for v in values if v.grad is not None)
    assert grad(raw)>0 and grad(reads)>0 and model.rotation_anchor_readout.readout[-1].weight.grad.abs().sum()>0
    assert writes[0].grad is not None and (writes[0].grad.abs().sum(0)>0).all()
    assert writes[7].grad is None and writes[-1].grad is None
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    gradients=dict(negative_logits=112,raw_logit_grad_l1=grad(raw),read_grad_l1=grad(reads),
        output_weight_grad_l1=float(model.rotation_anchor_readout.readout[-1].weight.grad.abs().sum()),
        first_writer_grad_l1=writes[0].grad.abs().sum(0).tolist(),tbptt_boundary_verified=True)
    ck=torch.load(e['arms']['smooth_rotation']['init'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    optimizer,scheduler=optimizer_and_scheduler(model,c);pilot=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);out=StreamTrainingModule(model,c,renderer)(samples);out['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip_norm']);assert torch.isfinite(norm)
        optimizer.step();scheduler.step();pilot.append(dict(step=step+1,loss=float(out['loss'].detach()),
            grad_norm=float(norm),diagnostics=out['smooth_rotation_diagnostics'].tolist()))
    assert pilot[-1]['diagnostics'][1]>0
    trainable=set(e['arms']['smooth_rotation']['trainable'])
    assert all(torch.equal(value,model.state_dict()[name].cpu()) for name,value in ck['model'].items() if name not in trainable)
    assert not torch.equal(ck['model']['rotation_anchor_readout.readout.2.weight'],model.rotation_anchor_readout.readout[-1].weight.cpu())
    result=dict(passed=True,source_sha256=source_hash(),parent_sha256=e['parent_sha256'],results=results,
        gradient_routing=gradients,pilot=pilot,pilot_weights_discarded=True,
        scope='Two real train fragments. Negative-logit gradient and pilot activity verified; no accuracy improvement claim or held-out access.')
    (a.experiment/'equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

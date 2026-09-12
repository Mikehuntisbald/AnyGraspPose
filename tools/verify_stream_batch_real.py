"""Compare real complete unroll predictions and gradients with batched preparation."""
import argparse
import json
from pathlib import Path
import sys
from contextlib import nullcontext
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha,source_hash
from lip.engine.config import check_data_gate
from lip.engine.stream_training import StreamTrainingModule
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    p.add_argument('--init-from',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True);p.add_argument('--math-fp32',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_device(0)
    c=load_stream_config(a.config);audit=check_data_gate('cache/dexycb_s0')
    if a.math_fp32:
        c['precision']='fp32';torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    from torch.nn.attention import sdpa_kernel,SDPBackend
    context=lambda:sdpa_kernel(SDPBackend.MATH) if a.math_fp32 else nullcontext()
    ds=StreamClips(a.data_root,'cache/dexycb_s0',c['burn_in_frames'],c['supervised_unroll_frames'])
    manifest=ds.fixed_manifest(16);ds.fixed=manifest
    model=make_model(c).cuda().train();load_init(a.init_from,model,audit,c)
    renderer=Renderer('cuda');rows=[]
    for start in (0,8):
        samples=[ds[i] for i in range(start,start+8)]
        model.zero_grad(set_to_none=True)
        with context():
            before=StreamTrainingModule(model,dict(c,batch_current_features=False),renderer)(samples,True)
            before['loss'].backward()
        predictions=before['predictions'].detach().cpu();loss=float(before['loss'].detach())
        gradients={n:v.grad.detach().cpu().clone() for n,v in model.named_parameters() if v.grad is not None}
        del before;model.zero_grad(set_to_none=True)
        comparison={}
        for label,batched in [('unchanged_repeat',False),('batched',True)]:
            model.zero_grad(set_to_none=True)
            with context():
                after=StreamTrainingModule(model,dict(c,batch_current_features=batched),renderer)(samples,True)
                after['loss'].backward()
            pose_error=float((predictions-after['predictions'].detach().cpu()).abs().max())
            loss_error=abs(loss-float(after['loss'].detach()))
            diff=0.;norm=0.;max_abs=0.
            for name,v in model.named_parameters():
                if v.grad is None:assert name not in gradients;continue
                grad=v.grad.detach().cpu();old=gradients[name]
                diff+=float((grad-old).double().square().sum());norm+=float(old.double().square().sum())
                max_abs=max(max_abs,float((grad-old).abs().max()))
            comparison[label]=dict(pose_max_abs=pose_error,loss_abs=loss_error,
                                   gradient_max_abs=max_abs,gradient_relative_l2=(diff/max(norm,1e-30))**.5)
            del after
        rows.append(dict(start=start,clips=8,comparisons=comparison))
        print(json.dumps(rows[-1]),flush=True)
        current=comparison['batched'];repeat=comparison['unchanged_repeat']
        assert current['pose_max_abs']<1e-5 and current['loss_abs']<1e-5
        if a.math_fp32:assert current['gradient_relative_l2']<1e-5
        else:assert current['gradient_relative_l2']<=max(1e-3,1.25*repeat['gradient_relative_l2'])
        del gradients;model.zero_grad(set_to_none=True)
    report=dict(passed=True,rows=rows,real_fragments=16,burn_in=8,supervised_unroll=16,
                precision=c['precision'],math_fp32=a.math_fp32,source_sha256=source_hash(),init_sha256=sha(a.init_from),manifest=manifest)
    Path(a.out).write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()

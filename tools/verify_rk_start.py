"""Verify trained, nonzero parent outputs and actual RGB-D closed-loop inputs."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.stream_checkpoint import sha,source_hash
from lip.models.stream_tracker import StreamTracker
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.renderer import Renderer
from lip.data.stream_clips import StreamClips


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--arm',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(0)
    e=json.loads((a.experiment/'experiment.json').read_text());folder=a.experiment/a.arm;c=load_stream_config(folder/'config.yaml')
    init=torch.load(folder/'init.pt',map_location='cpu',weights_only=False)
    parent=torch.load(init['parent']['path'],map_location='cpu',weights_only=False)
    model=make_model(c).cuda().eval();model.load_state_dict(init['model'])
    baseline=StreamTracker('stream_dual_cross_residual',memory_frames=8,dropout=0.).cuda().eval();baseline.load_state_dict(parent['model'])
    for n,p in baseline.named_parameters():p.requires_grad_(dict(model.named_parameters())[n].requires_grad)
    ds=StreamClips(e['data_root'],e['index_root'],8,32,seed=42,decode_threads=4)
    samples=[ds[i] for i in range(2)];renderer=Renderer('cuda');results={}
    for precision in ('fp32','bf16'):
        cfg=dict(c,precision=precision);a_runner=StreamTrainingModule(baseline,cfg,renderer);b_runner=StreamTrainingModule(model,cfg,renderer)
        with torch.no_grad():
            x=a_runner(samples,True)['predictions'];y=b_runner(samples,True)['predictions']
        error=(x-y).abs();results[precision]=dict(max_pose_matrix_difference=float(error.max()),bitwise_equal=torch.equal(x,y),frames=x.shape[0]*x.shape[1])
        # Closed-loop numerical tolerance. A changed zero-init function is not
        # accepted merely because an untrained zero pose head hides it.
        torch.testing.assert_close(x,y,rtol=0,atol=2e-5 if precision=='fp32' else 2e-3)
    (folder/'equivalence.json').write_text(json.dumps(dict(passed=True,real_training_fragments=2,
        parent_sha256=sha(init['parent']['path']),init_sha256=sha(folder/'init.pt'),source_sha256=source_hash(),
        results=results,gt_current_enters_predictor=False,foundationpose_calls=0),indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()

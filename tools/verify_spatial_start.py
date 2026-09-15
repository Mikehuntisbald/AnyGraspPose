"""Actual trained-parent equivalence over real RGB-D closed loops."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config
from lip.engine.stream_checkpoint import sha,source_hash
from lip.engine.stream_training import StreamTrainingModule
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args();e=json.loads((a.experiment/'experiment.json').read_text())
    torch.set_num_threads(2);torch.cuda.set_device(0);models={};configs={}
    for name in ('control','spatial'):
        c=load_stream_config(e['arms'][name]['config']);configs[name]=c;m=make_model(c).cuda().eval()
        m.load_state_dict(torch.load(e['arms'][name]['init'],map_location='cpu',weights_only=False)['model']);models[name]=m
    ds=StreamClips(e['data_root'],e['index_root'],8,48,seed=42);samples=[ds[i] for i in range(2)];renderer=Renderer('cuda');results={}
    for precision in ('fp32','bf16'):
        predictions={}
        for name,m in models.items():
            runner=StreamTrainingModule(m,dict(configs[name],precision=precision),renderer)
            with torch.no_grad():predictions[name]=runner(samples,True)['predictions']
        x,y=predictions['control'],predictions['spatial'];assert torch.equal(x,y)
        results[precision]=dict(bitwise_equal=True,frames=x.shape[0]*x.shape[1],max_pose_difference=float((x-y).abs().max()))
    (a.experiment/'equivalence.json').write_text(json.dumps(dict(passed=True,parent_sha256=e['parent_sha256'],source_sha256=source_hash(),results=results,
        control_init_sha256=sha(e['arms']['control']['init']),spatial_init_sha256=sha(e['arms']['spatial']['init']),real_training_fragments=2),indent=2))


if __name__=='__main__':main()

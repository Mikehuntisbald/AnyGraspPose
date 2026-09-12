"""Profile one warmed streaming frame; never mix profiler overhead into timing results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha
from lip.engine.config import check_data_gate
from lip.data.index import read_frame
from lip.geometry.renderer import Renderer


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--index-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    c=load_stream_config(a.config);audit=check_data_gate(a.index_root);torch.set_num_threads(2)
    model=make_model(c).cuda().eval();load_init(a.checkpoint,model,audit,c)
    s=next(s for s in sorted(map(json.loads,(Path(a.index_root)/'streams.jsonl').read_text().splitlines()),key=lambda s:s['stream_id']) if s['split']=='val')
    with np.load(Path(a.index_root)/s['pose_cache']) as z:poses=z['poses'].copy();frames=z['frames'].copy()
    with np.load(Path(a.index_root)/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
    times=frames.astype('f8')/audit['fps'];state=model.initialize(poses[0],mesh,s['intrinsics'],s['stream_id'],times[0]);renderer=Renderer('cuda')
    class Scopes:
        def section(self,name):return torch.profiler.record_function(name)
    with torch.no_grad():
        for j in range(1,9):
            rgb,depth=read_frame(a.data_root,s,int(frames[j]),audit['depth_scale_to_m'])
            proposal,next_state=model.step(torch.from_numpy(rgb),torch.from_numpy(depth),times[j],state,renderer=renderer,precision=c['precision']);state=model.commit(proposal,next_state)
        rgb,depth=read_frame(a.data_root,s,int(frames[9]),audit['depth_scale_to_m'])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            proposal,next_state=model.step(torch.from_numpy(rgb),torch.from_numpy(depth),times[9],state,renderer=renderer,precision=c['precision'],profiler=Scopes())
            state=model.commit(proposal,next_state);torch.cuda.synchronize()
    kernels={}
    for event in prof.events():
        if not event.name.startswith('aten::_scaled_dot_product_'):continue
        parent=event.cpu_parent;scope='unknown'
        while parent is not None:
            if parent.name in ('spatial_fusion','temporal_readout'):scope=parent.name;break
            parent=parent.cpu_parent
        key=scope+':'+event.name;kernels[key]=kernels.get(key,0)+1
    Path(a.out).write_text(json.dumps(dict(completed=True,checkpoint_sha256=sha(a.checkpoint),stream_id=s['stream_id'],
        attention_operators=kernels,cache_bytes=state.cache.kv_bytes,scope='One warmed frame with 8-frame KV; profiler latency is not used in benchmark results'),indent=2))
    print(json.dumps(kernels,indent=2))

if __name__=='__main__':main()

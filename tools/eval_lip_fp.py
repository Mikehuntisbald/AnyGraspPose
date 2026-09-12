"""Evaluate whole camera streams independently; post-FP poses own hybrid history."""
import argparse,hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch
from lip.engine.config import load_config,check_data_gate
from lip.evaluate import evaluate
from lip.models.tracker import Tracker
from lip.evaluation.selection import fixed_balanced_subset

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--data-root',default=os.environ.get('DEX_YCB_DIR'));p.add_argument('--index-root',default='cache/dexycb_s0')
    p.add_argument('--out',required=True);p.add_argument('--method',choices=['lip','lip_fp'],required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--limit-streams',type=int);p.add_argument('--max-frames',type=int)
    a=p.parse_args();assert 0<=a.rank<a.world_size
    out=Path(a.out);assert not (out/'predictions.jsonl').exists(),'Refuse to overwrite an evaluation'
    c=load_config(a.config);audit=check_data_gate(a.index_root);torch.set_num_threads(2)
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.set_device(0)
    checkpoint=Path(a.checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['split_hash']==audit['split_hash'] and ck['mesh_hash']==audit['mesh_hash']
    model=Tracker(False).cuda();model.load_state_dict(ck['model']);model.eval()
    streams=[s for s in map(json.loads,(Path(a.index_root)/'streams.jsonl').read_text().splitlines()) if s['split']=='val']
    streams=fixed_balanced_subset(sorted(streams,key=lambda s:s['stream_id']),a.limit_streams,lambda s:s['object_id'],c['seed'])
    ids=[s['stream_id'] for s in streams][a.rank::a.world_size];assert ids
    fp=None
    if a.method=='lip_fp':
        from lip.integrations.frozen_fp import FrozenFoundationPose
        fp=FrozenFoundationPose(c['foundationpose_root'],a.data_root,torch.device('cuda',0),c['foundationpose_refiner_sha256'])
    info=dict(path=str(checkpoint.resolve()),global_step=ck['global_step'],sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),code_sha256=ck['code_sha256'])
    evaluate(model,c,a.data_root,a.index_root,out,limit_streams=a.limit_streams,max_frames=a.max_frames,
             checkpoint_info=info,fp_transition=fp,stream_ids=ids)
    manifest=json.loads((out/'manifest.json').read_text());manifest.update(shard_rank=a.rank,shard_count=a.world_size,
        fp_refiner_sha256=fp.weight_sha256 if fp else None,evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()

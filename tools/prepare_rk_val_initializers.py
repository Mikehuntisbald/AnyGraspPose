"""Explicit opt-in to a controlled, GT-derived validation initializer artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.config import check_data_gate
from lip.engine.stream_checkpoint import source_hash
from lip.data.perturb import noisy_history
from lip.geometry.so3 import center_pose,original_pose


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--index-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--controlled-noisy-gt',required=True,action='store_true',help='Acknowledge that this controlled initializer uses GT pose')
    p.add_argument('--seed',type=int,default=20260913);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    audit=check_data_gate(a.index_root)
    streams=sorted([s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines()) if s['split']=='val'],key=lambda s:s['stream_id'])
    assert len(streams)==320 and sum(s['num_frames'] for s in streams)==23200
    result=dict(split='val',split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],source_sha256=source_hash(),
        protocol='Fixed noisy GT first-frame pose; controlled mechanism ablation only',uses_gt_pose=True,
        seed=a.seed,noise='Existing 75/20/5 mixture: 2/8/20 degree axis scales, .01/.05/.10 diameter translation scales; capped at 45 degrees and .25d',
        pose_convention='Original mesh origin, object-to-camera transform, meters',poses={},streams={})
    for s in streams:
        with np.load(a.index_root/s['pose_cache']) as z:pose=torch.from_numpy(z['poses'][0].copy());frame=int(z['frames'][0])
        with np.load(a.index_root/s['mesh_cache']) as z:center=torch.from_numpy(z['center'].copy());diameter=float(z['diameter'])
        seed=int.from_bytes(hashlib.sha256(f'{a.seed}/{s["stream_id"]}'.encode()).digest()[:4],'little')
        centered=center_pose(pose,center)
        noisy,_=noisy_history(centered[None],diameter,torch.Generator().manual_seed(seed))
        result['poses'][s['stream_id']]=original_pose(noisy[0],center).tolist()
        result['streams'][s['stream_id']]=dict(first_frame=frame,noise_seed=seed)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

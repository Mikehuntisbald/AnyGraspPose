"""Prepare sealed full-texture CAD caches and validate real frozen encoders."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.engine.object_jepa_checkpoint import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    c=yaml.safe_load(a.config.read_text());m=build_model(c);store=make_store(c,m)
    index=Path(c['paths']['index_root']);root=Path(c['paths']['data_root'])
    models={s['object_id']:s for s in map(json.loads,(index/'streams.jsonl').read_text().splitlines()) if s['split'] in ('train','val')}
    receipts=[];begun=time.monotonic();a.out.mkdir(parents=True,exist_ok=True)
    for oid,s in sorted(models.items()):
        with np.load(index/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
        cad=store.get(root/s['mesh_path'],mesh)
        assert tuple(cad['appearance']['texture'].shape[1:3])==(4096,4096)
        receipts.append(dict(object_id=oid,**cad['receipt']))
        print(json.dumps(dict(object_id=oid,cache_key=cad['key'],features=list(cad['features'].shape),elapsed_s=time.monotonic()-begun)),flush=True)
    sample=next(iter(store.resident.values()))
    repeat=m.utonia([{k:sample[k] for k in ('coord','color','normal')}])[0]
    error=float((repeat-sample['features']).abs().max())
    if not torch.allclose(repeat,sample['features'],atol=1e-4,rtol=1e-4):raise ValueError('Cached/live CAD mismatch')
    atomic_json(a.out/'cad_receipt.json',dict(completed=True,objects=len(receipts),cad=receipts,cached_live_max_abs=error,
        frozen_encoders=True,geometry_feature_dim=m.utonia.feature_dim,elapsed_s=time.monotonic()-begun))

if __name__=='__main__':main()

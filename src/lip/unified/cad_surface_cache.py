"""Derived local surface cache; original 8192-point frozen cache is immutable."""
import hashlib,json,os,fcntl
from pathlib import Path
import numpy as np
import torch


def surface_tokens(cad,root,count=256):
    contract=dict(version='fps-local-utonia-v1',parent_key=cad['key'],
        parent_sha256=cad['receipt']['sha256'],count=count,
        preprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if cad.get('_surface_key')==key:return cad['_surface_data']
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/(key+'.pt')
    receipt=path.with_suffix('.json')
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            xyz=cad['coord'].detach().float().cpu().numpy();n=len(xyz)
            if n<count:raise ValueError('Surface bank smaller than token budget')
            chosen=[];distance=np.full(n,np.inf);index=int(np.square(xyz-xyz.mean(0)).sum(1).argmax())
            for _ in range(count):
                chosen.append(index);distance=np.minimum(distance,np.square(xyz-xyz[index]).sum(1))
                distance[chosen]=-1;index=int(distance.argmax())
            indices=torch.tensor(chosen,device=cad['coord'].device)
            anchors=cad['coord'][indices]
            assignment=torch.cdist(cad['coord'].float(),anchors.float()).argmin(-1)
            # No CUDA atomics: a small deterministic CPU aggregation runs once/object.
            features=cad['features'].detach().float().cpu();groups=assignment.cpu()
            pooled=torch.stack([features[groups==i].mean(0) for i in range(count)])
            record=dict(coord=anchors.cpu(),features=pooled,normal=cad['normal'][indices].cpu(),
                        color=cad['color'][indices].cpu(),indices=indices.cpu(),parent_key=cad['key'])
            temp=path.with_suffix(f'.{os.getpid()}.tmp');torch.save(record,temp);os.replace(temp,path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            temp=receipt.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(dict(contract=contract,sha256=digest)))
            os.replace(temp,receipt)
        meta=json.loads(receipt.read_text())
        if meta['contract']!=contract or hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('Derived CAD surface cache identity mismatch')
    data=torch.load(path,map_location=cad['coord'].device,weights_only=True)
    cad['_surface_key']=key;cad['_surface_data']=data
    return data

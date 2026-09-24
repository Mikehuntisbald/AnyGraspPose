"""Full-resolution textured assets and content-bound frozen CAD point features."""
import hashlib
import json
import os
import shlex
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lip.geometry.appearance_renderer import load_appearance_mesh
from lip.jepa.cad_anchors import cad_asset_hash
from lip.engine.jepa_checkpoint import sha
from lip.engine.object_jepa_checkpoint import atomic_json


def surface_points(appearance, count=8192, seed=42):
    vertices=appearance['vertices']; faces=appearance['faces'].long(); triangles=vertices[faces]
    areas=torch.linalg.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]).norm(dim=-1)
    rng=np.random.default_rng(seed)
    indices=torch.tensor(rng.choice(len(faces), count, p=(areas/areas.sum()).cpu().double().numpy()/float((areas/areas.sum()).cpu().double().sum())),device=vertices.device)
    uv=torch.tensor(rng.random((count,2)),device=vertices.device,dtype=torch.float32)
    uv=torch.where((uv.sum(-1)>1)[:,None],1-uv,uv)
    bary=torch.cat((1-uv.sum(-1,keepdim=True),uv),-1)
    selected=faces[indices]
    xyz=(vertices[selected]*bary[...,None]).sum(1)
    texcoord=(appearance['uv'][selected]*bary[...,None]).sum(1)
    # nvdiffrast texture centers follow normalized UV, matching align_corners=False.
    rgb=F.grid_sample(appearance['texture'].permute(0,3,1,2), (texcoord*2-1)[None,None],
                      align_corners=False,padding_mode='border').squeeze(0).squeeze(1).T
    normal=F.normalize((appearance['normals'][selected]*bary[...,None]).sum(1),dim=-1)
    return dict(coord=xyz,color=rgb,normal=normal)


def asset_files(path):
    path=Path(path);files={path}
    for line in path.read_text().splitlines():
        if line.startswith('mtllib '):
            for name in shlex.split(line)[1:]:
                material=path.parent/name;files.add(material)
                for row in material.read_text().splitlines():
                    if row.strip().startswith('map_Kd '):files.add(material.parent/shlex.split(row)[-1])
    files.update([Path(__file__),Path(__file__).with_name('utonia.py'),Path(__file__).with_name('utonia_fast.py'),Path(__file__).parents[1]/'geometry/appearance_renderer.py',Path(__file__).parents[1]/'geometry/renderer.py'])
    return sorted(files)


def file_stats(paths):return tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)


class CADStore:
    def __init__(self, root, point_encoder, device='cuda'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.point_encoder=point_encoder;self.device=device;self.resident={}
        self.encoder_versions=tuple(p._version for p in point_encoder.parameters())

    @torch.no_grad()
    def get(self, asset_path, mesh):
        identity=str(Path(asset_path).resolve())
        if self.encoder_versions!=tuple(p._version for p in self.point_encoder.parameters()):
            self.resident.clear();raise ValueError('Frozen point weights changed; rebind weights and rebuild CAD store')
        diameter=float(mesh['diameter']);center=np.asarray(torch.as_tensor(mesh['center']).cpu())
        if identity in self.resident:
            item=self.resident[identity];contract=item['receipt']['contract']
            if file_stats(item['_asset_files'])==item['_asset_stats'] and contract['utonia']==self.point_encoder.checkpoint_sha256 and contract['center']==center.tolist() and contract['diameter']==diameter:
                return item
            del self.resident[identity]
        asset=cad_asset_hash(asset_path)
        contract=dict(version='full-texture-utonia-v1',asset=asset,utonia=self.point_encoder.checkpoint_sha256,
                      center=center.tolist(),diameter=diameter,points=8192,seed=42,grid=.01,
                      normalize='CAD center and diameter; no per-cloud centering',texture_cap=4096,precision='fp32',
                      utonia_commit='da776a0bd3a48c6df83ac2ae0e27b26141cc7e31',utonia_sources_sha256=self.point_encoder.source_sha256,
                      preprocessing_sha256=sha(__file__),encoder_wrapper_sha256=sha(Path(__file__).with_name('utonia.py')),
                      fast_adapter_sha256=sha(Path(__file__).with_name('utonia_fast.py')) if getattr(self.point_encoder,'fast',False) else None,
                      texture_loader_sha256=sha(Path(__file__).parents[1]/'geometry/appearance_renderer.py'))
        key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        path=self.root/(key+'.pt');receipt=path.with_suffix('.json')
        appearance=load_appearance_mesh(asset_path,center,self.device,texture_size=4096)
        if not path.exists():
            cloud=surface_points(appearance);cloud['coord']=cloud['coord']/diameter
            features=self.point_encoder([cloud])[0]
            saved={k:v.cpu() for k,v in cloud.items()};saved['features']=features.cpu()
            # Each rank uses its own staging path; identical immutable result.
            temporary=path.with_suffix(f'.{os.getpid()}.tmp');torch.save(saved,temporary);os.replace(temporary,path)
            atomic_json(receipt,dict(completed=True,key=key,contract=contract,sha256=sha(path),
                texture_shape=list(appearance['texture'].shape),feature_shape=list(features.shape)))
        if not receipt.exists():raise ValueError('Unsealed CAD cache; prepare caches before distributed use')
        meta=json.loads(receipt.read_text())
        if meta['contract']!=contract or meta['sha256']!=sha(path):raise ValueError('CAD cache identity mismatch')
        saved=torch.load(path,map_location=self.device,weights_only=True)
        result=dict(**saved,appearance=appearance,key=key,receipt=meta,path=str(path))
        result['_asset_files']=asset_files(asset_path);result['_asset_stats']=file_stats(result['_asset_files'])
        self.resident[identity]=result
        return result

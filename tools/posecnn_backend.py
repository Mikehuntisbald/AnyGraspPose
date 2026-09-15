"""Pinned PoseCNN RGB inference shared by new train-initializer collection."""
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
import yaml
from lip.data.index import CLASSES
from lip.engine.stream_checkpoint import sha


class PoseCNNBackend:
    def __init__(self,source,native_build,deps,checkpoint,data_root,expected_sha,log):
        source,native_build=Path(source),Path(native_build);assert sha(checkpoint)==expected_sha
        torch.manual_seed(3);np.random.seed(3);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sys.path[:0]=[str(native_build),str(deps),str(source/'lib')]
        native=importlib.import_module('lip_posecnn_inference_cuda');sys.modules['posecnn_cuda']=native
        build=json.loads((native_build/'build_receipt.json').read_text());assert sha(native.__file__)==build['module_sha256']
        overlap=types.ModuleType('utils.cython_bbox')
        def forbidden(*args,**kwargs):raise RuntimeError('GT overlap unavailable during initializer inference')
        overlap.bbox_overlaps=forbidden;sys.modules['utils.cython_bbox']=overlap
        from easydict import EasyDict
        from fcn.config import cfg,_merge_a_into_b
        _merge_a_into_b(EasyDict(yaml.load((source/'experiments/cfgs/dex_ycb.yml').read_text(),Loader=yaml.FullLoader)),cfg)
        cfg.MODE='TEST';cfg.TEST.POSE_REFINE=False;cfg.TRAIN.GPUNUM=1
        from networks.PoseCNN import PoseCNN
        from utils.nms import nms
        from utils.se3 import allocentric2egocentric
        from transforms3d.quaternions import quat2mat
        self.nms=nms;self.allocentric=allocentric2egocentric;self.quat2mat=quat2mat
        self.ids=list(cfg.TRAIN.CLASSES);assert self.ids==[0,*range(1,19),20,21]
        with Path(log).open('w') as f,contextlib.redirect_stdout(f):self.model=PoseCNN(len(self.ids),cfg.TRAIN.NUM_UNITS)
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False),strict=True)
        self.model.cuda().eval();assert all(torch.isfinite(t).all() for t in self.model.state_dict().values())
        self.mean=torch.from_numpy(cfg.PIXEL_MEANS/255.).float();extents=np.zeros((len(self.ids),3),dtype='f4')
        for i,oid in enumerate(self.ids[1:],1):
            points=np.loadtxt(Path(data_root)/'models'/CLASSES[oid-1]/'points.xyz');extents[i]=2*np.max(np.abs(points),axis=0)
        self.extents=torch.from_numpy(extents[None]).cuda();n=len(self.ids)
        self.labels=torch.zeros(1,n,480,640,device='cuda');self.boxes=torch.zeros(1,n,5,device='cuda')
        self.poses=torch.zeros(1,n,9,device='cuda');self.points=torch.zeros(1,n,1,3,device='cuda');self.symmetry=torch.zeros(1,n,device='cuda')
        h=hashlib.sha256()
        for directory in ('lib','experiments/cfgs'):
            for p in sorted((source/directory).rglob('*')):
                if p.suffix in ('.py','.yml','.cu','.cpp'):h.update(str(p.relative_to(source)).encode());h.update(p.read_bytes())
        h.update(Path(__file__).read_bytes());self.source_sha256=h.hexdigest();self.native_build=build

    @torch.no_grad()
    def predict(self,bgr,K):
        assert bgr.shape==(480,640,3)
        value=torch.from_numpy(bgr)/255.;value-=self.mean;x=value.permute(2,0,1).float().contiguous()[None].cuda()
        K=np.asarray(K,dtype='f4');meta=np.zeros((1,18),dtype='f4');meta[0,:9]=K.reshape(-1);meta[0,9:]=np.linalg.pinv(K).reshape(-1)
        _,_,rois,poses,quaternions=self.model(x,self.labels,torch.from_numpy(meta).cuda(),self.extents,self.boxes,self.poses,self.points,self.symmetry)
        rois,poses,quaternions=[v.cpu().numpy() for v in (rois,poses,quaternions)];candidates=[]
        for j in self.nms(rois,.5):
            cls=int(rois[j,1])
            if cls<=0 or cls>=len(self.ids):continue
            qt=quaternions[j,4*cls:4*cls+4];length=np.linalg.norm(qt)
            if not np.isfinite(length) or length<=0:continue
            qt=qt/length;pose=poses[j].copy();pose[4:6]*=pose[6];pose[:4]=self.allocentric(qt,pose[4:])
            T=np.eye(4);T[:3,:3]=self.quat2mat(pose[:4]);T[:3,3]=pose[4:]
            candidates.append(dict(object_id=self.ids[cls],score=float(rois[j,6]),pose_original=T.tolist()))
        return candidates

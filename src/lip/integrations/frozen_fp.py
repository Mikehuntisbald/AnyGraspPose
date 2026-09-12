"""Frozen official FoundationPose state transition; no GT input or trainable parameters."""
import contextlib,logging,sys,random,hashlib
from pathlib import Path
import numpy as np
import torch
from lip.geometry.so3 import original_pose,center_pose
from lip.integrations.foundationpose import FoundationPoseAdapter

class FrozenFoundationPose:
    def __init__(self,root,data_root,device,expected_sha256=None):
        self.device=torch.device(device);self.data_root=Path(data_root);self.pool={}
        weight=Path(root)/'weights/2023-10-28-18-33-37/model_best.pth'
        self.weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest()
        if expected_sha256 and self.weight_sha256!=expected_sha256:raise ValueError('FoundationPose weight hash mismatch')
        sys.path.insert(0,str(Path(root).resolve()))
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        import nvdiffrast.torch as dr
        class TrackingOnly(FoundationPose):
            def make_rotation_grid(self,*args,**kwargs):pass
        self.estimator_class=TrackingOnly
        with self.preserve_rng(),torch.cuda.device(self.device):
            self.refiner=PoseRefinePredictor();self.refiner.model.requires_grad_(False);self.refiner.model.eval()
            self.context=dr.RasterizeCudaContext(device=self.device)
        logging.getLogger().setLevel(logging.WARNING)
    @contextlib.contextmanager
    def preserve_rng(self):
        py=random.getstate();npstate=np.random.get_state()
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            try:yield
            finally:random.setstate(py);np.random.set_state(npstate)
    @torch.no_grad()
    def __call__(self,pred,rgb,depth,K,mesh_path,center):
        assert not pred.requires_grad, 'FP receives detached LIP output only'
        old_type=torch.tensor(0.).type()
        with self.preserve_rng(),torch.cuda.device(self.device):
            try:
                if mesh_path not in self.pool:
                    import trimesh
                    raw=trimesh.load(self.data_root/mesh_path,process=False,force='mesh')
                    assert np.allclose((raw.vertices.max(0)+raw.vertices.min(0))/2,np.asarray(center),atol=1e-6)
                    np.random.seed(42)
                    e=self.estimator_class(raw.vertices,raw.vertex_normals,mesh=raw,scorer=object(),refiner=self.refiner,glctx=self.context,debug=0,debug_dir=str(self.data_root.parent/'fp_runtime_debug'))
                    e.diameter=float(e.diameter);self.pool[mesh_path]=e
                e=self.pool[mesh_path]
                c=torch.as_tensor(center,device=pred.device,dtype=pred.dtype)
                prior=original_pose(pred,c)
                FoundationPoseAdapter(e,self.device).accept(prior)
                color=np.rint(rgb.detach().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype('uint8')
                d=depth.detach().cpu().numpy()[0];k=K.detach().cpu().numpy().astype('f4')
                result=e.track_one(rgb=color,depth=d,K=k,iteration=2)
                result=torch.as_tensor(result,device=pred.device,dtype=pred.dtype)
                if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite FP transition')
                return center_pose(result,c).detach().clone()
            finally:torch.set_default_tensor_type(old_type)

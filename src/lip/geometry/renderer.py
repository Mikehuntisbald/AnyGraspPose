"""OpenCV pixel centers throughout. CUDA boundary maps row 0 to NDC y=-1.

nvdiffrast's tensor row 0 is the bottom NDC row; using y-down NDC here
directly produces OpenCV-ordered tensors, without an extra image flip.
"""
import numpy as np
import torch


class Renderer:
    def __init__(self, device='cpu', backend=None):
        self.device = torch.device(device)
        self.backend = backend or ('cuda' if self.device.type == 'cuda' else 'cpu')
        if self.backend == 'cuda':
            import nvdiffrast.torch as dr
            self.dr = dr
            self.context = dr.RasterizeCudaContext(device=self.device)

    @torch.no_grad()
    def __call__(self, mesh, pose, k, size=224):
        with torch.autocast(self.device.type, enabled=False):
            v = torch.as_tensor(mesh['vertices'], device=self.device, dtype=torch.float32)
            f = torch.as_tensor(mesh['faces'], device=self.device, dtype=torch.int32)
            pose, k = pose.float(), k.float()
            cam = v @ pose[:3, :3].T + pose[:3, 3]
            if self.backend == 'cpu':
                return self._cpu(v, f, cam, k, size)
            z = cam[:, 2]
            pix = cam @ k.T
            near, far = .001, 100.
            clip = torch.stack((2*(pix[:, 0]+.5*z)/size-z,
                                2*(pix[:, 1]+.5*z)/size-z,
                                (far+near)/(far-near)*z-2*far*near/(far-near), z), -1)
            rast, _ = self.dr.rasterize(self.context, clip[None], f, resolution=[size, size])
            attrs = torch.cat((z[:, None], v), -1)[None]
            attr, _ = self.dr.interpolate(attrs.contiguous(), rast, f)
            mask = rast[..., 3:] > 0
            attr = torch.where(mask, attr, 0)[0].permute(2, 0, 1)
            return attr[:1], attr[1:]

    def _cpu(self, v, faces, cam, k, size):
        v, faces, cam, k = [x.cpu().numpy() for x in (v, faces, cam, k)]
        p = cam @ k.T; uv = p[:, :2]/np.maximum(p[:, 2:], 1e-8)
        depth = np.full((size, size), np.inf, dtype='f4'); xyz = np.zeros((size, size, 3), dtype='f4')
        for face in faces:
            z = cam[face, 2]
            if np.any(z <= .001): continue
            t = uv[face]; lo = np.maximum(np.ceil(t.min(0)).astype(int), 0)
            hi = np.minimum(np.floor(t.max(0)).astype(int), size-1)
            if np.any(hi < lo): continue
            x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
            den = (t[1, 1]-t[2, 1])*(t[0, 0]-t[2, 0])+(t[2, 0]-t[1, 0])*(t[0, 1]-t[2, 1])
            if abs(den) < 1e-10: continue
            w0 = ((t[1, 1]-t[2, 1])*(x-t[2, 0])+(t[2, 0]-t[1, 0])*(y-t[2, 1]))/den
            w1 = ((t[2, 1]-t[0, 1])*(x-t[2, 0])+(t[0, 0]-t[2, 0])*(y-t[2, 1]))/den
            w = np.stack((w0, w1, 1-w0-w1), -1); inv = (w/z).sum(-1)
            zz = 1/np.maximum(inv, 1e-10)
            visible = (w.min(-1)>=-1e-6) & (zz < depth[y, x])
            yy, xx = y[visible], x[visible]
            depth[yy, xx] = zz[visible]
            xyz[yy, xx] = ((w[visible]/z)*zz[visible, None]) @ v[face]
        depth[~np.isfinite(depth)] = 0
        return torch.from_numpy(depth[None]).to(self.device), torch.from_numpy(xyz.transpose(2, 0, 1)).to(self.device)

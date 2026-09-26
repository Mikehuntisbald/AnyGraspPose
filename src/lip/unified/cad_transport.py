"""JEPA predicts image-to-CAD transport; the CAD map is a detached lookup table.

No GT, sensor mask or second encoder enters this decoder. Unsupported surfaces
retain the existing DPT prediction. Camera depth is corrected independently of
the estimated-pose rendering, so this is not copying the estimated pose.
"""
import torch
from torch import nn


def pixel_grid(batch, height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device),
                          torch.arange(width, device=device), indexing='ij')
    return torch.stack((x, y), 0).float()[None].expand(batch, -1, -1, -1)


def sample_reference(image, mask, uv):
    """Mask-normalized bilinear lookup, deterministic gradient to UV only.

    Reference images are constants. Gather does not need a scatter backward;
    only the bilinear weights depend on UV. Pixel centers use integer indices.
    """
    image = image.detach().float(); mask = mask.detach().bool()
    b, channels, height, width = image.shape
    finite = torch.isfinite(uv).all(1, keepdim=True)
    u, v = uv.float().nan_to_num().unbind(1)
    x0, y0 = u.floor().detach(), v.floor().detach()
    fx, fy = u-x0, v-y0
    value = image.new_zeros(b, channels, *u.shape[-2:])
    mass = image.new_zeros(b, 1, *u.shape[-2:])
    flat = torch.where(mask, image, 0.).flatten(2)
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        x, y = x0+dx, y0+dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        index = (y.clamp(0, height-1).long()*width+x.clamp(0, width-1).long()).flatten(1)
        good = mask.flatten(2).gather(2, index[:, None]).reshape_as(mass)
        w = weight[:, None]*inside[:, None]*good*finite
        samples = flat.gather(2, index[:, None].expand(-1, channels, -1)).reshape_as(value)
        value = value + w*samples
        mass = mass + w
    return value/mass.clamp_min(1e-6), mass


class CADTransport(nn.Module):
    def __init__(self, enabled=True, reference_conditioned=False, surface_locked=False, mandatory_lookup=False):
        super().__init__()
        self.enabled = enabled
        self.reference_conditioned = reference_conditioned
        self.surface_locked = surface_locked
        self.mandatory_lookup = mandatory_lookup
        self.head = nn.Sequential(nn.Conv2d(25 if reference_conditioned else 16, 32, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 7, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[6] = -4.

    def forward(self, dense, fallback, geometry, cad_valid):
        if self.reference_conditioned:
            b = len(dense)
            bounds = cad_valid.reshape(b,1,16,16).repeat_interleave(14,-2).repeat_interleave(14,-1)
            available = (geometry[:,3:4] > 0) & bounds & torch.isfinite(geometry[:,2:7]).all(1,keepdim=True)
            reference = torch.cat((geometry[:,4:7],geometry[:,2:3]),1).detach().float()
            # Give the query explicit reference geometry and its signed error.
            # No RGB/FP feature or alternative pose path is introduced.
            reference = torch.where(available,reference,0.)
            residual = torch.where(available,fallback[:,:4].float()-reference,0.)
            condition = torch.cat((reference,available.float(),residual),1)
            dense = torch.cat((dense,condition.to(dense.dtype)),1)
        raw = self.head(dense).float()
        with torch.autocast(raw.device.type, enabled=False):
            b, _, h, w = fallback.shape
            flow = raw[:, :2].tanh()*112.
            uv = pixel_grid(b, h, w, raw.device)+flow
            bounds = cad_valid.reshape(b, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
            valid = (geometry[:, 3:4] > 0) & bounds & torch.isfinite(geometry[:, 2:7]).all(1, keepdim=True)
            reference = torch.cat((geometry[:, 4:7], geometry[:, 2:3]), 1)
            warped, mass = sample_reference(reference, valid, uv)
            correction = torch.cat((.05*raw[:, 2:5].tanh(), .25*raw[:, 5:6].tanh()), 1)
            if self.surface_locked:
                correction = torch.cat((torch.zeros_like(correction[:,:3]),correction[:,3:4]),1)
            transported = warped+correction
            gate = raw[:, 6:7].sigmoid()*(mass >= .999).detach()
            confidence = gate
            if self.mandatory_lookup:
                gate = (mass >= .999).detach().float()
            if not self.enabled:
                gate = gate*0.
            surface = torch.cat((fallback[:, :4]+gate*(transported-fallback[:, :4]), fallback[:, 4:5]), 1)
        return surface, dict(transport_flow=flow, transport_gate_logits=raw[:, 6:7],
                             transport_gate=gate, transport_surface=transported,
                             transport_confidence=confidence,transport_lookup_mass=mass, transport_fallback=fallback)


@torch.no_grad()
def transport_targets(xyz, reference_geometry, base, diameter, crop_k, cad_valid):
    """Teacher-only exact projection and z-buffer/surface-identity eligibility.

    Canonical targets retain their annotated gauge, including textured objects;
    no geometry-only symmetry orbit is substituted for surface identity.
    """
    b, _, h, w = xyz.shape
    camera = torch.einsum('bij,bjhw->bihw', base[:, :3, :3].float(), xyz.float())
    camera = camera*diameter[:, None, None, None]+base[:, :3, 3, None, None]
    projected = torch.einsum('bij,bjhw->bihw', crop_k.float(), camera)
    uv = projected[:, :2]/projected[:, 2:3].clamp_min(.001)
    bounds = cad_valid.reshape(b, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
    valid = (reference_geometry[:, 3:4] > 0) & bounds
    reference = torch.cat((reference_geometry[:, 4:7], reference_geometry[:, 2:3]), 1)
    sampled, mass = sample_reference(reference, valid, uv)
    flow = uv-pixel_grid(b, h, w, xyz.device)
    supported = ((mass >= .999) & (camera[:, 2:3] > .001)
                 & ((sampled[:, :3]-xyz).norm(dim=1, keepdim=True) < .03)
                 & (flow.abs().amax(1, keepdim=True) < 110.))
    return dict(flow=flow, supported=supported, reference=sampled)

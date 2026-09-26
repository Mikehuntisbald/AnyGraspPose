"""Template-to-image flow as an INTERMEDIATE JEPA input, not a pose bypass.

Two rounds share parameters. Round two also compares the template's canonical
surface identity with the first round's DPT recovery. Each round keeps immutable
observed RGB-D evidence. GT is accepted by the separate target/loss functions
only. Rendered depth remains estimated-pose depth, never relabelled as recovered
camera depth. Full-CAD atlas decoding remains downstream of this interaction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def patch_grid(device):
    y, x = torch.meshgrid(torch.arange(16, device=device), torch.arange(16, device=device), indexing='ij')
    return torch.stack((x, y), -1).float().reshape(256, 2)*14+6.5


@torch.no_grad()
def template_points(geometry, cad_valid):
    """One ACTUAL raster sample per patch, nearest its center; no XYZ averaging."""
    b = len(geometry)
    valid = (geometry[:, 3] > 0) & torch.isfinite(geometry[:, 2:7]).all(1)
    cells = valid.reshape(b, 16, 14, 16, 14).permute(0, 1, 3, 2, 4).reshape(b, 256, 196)
    yy, xx = torch.meshgrid(torch.arange(14, device=geometry.device), torch.arange(14, device=geometry.device), indexing='ij')
    distance = ((xx-6.5)**2+(yy-6.5)**2).flatten()
    index = distance[None, None].expand_as(cells).masked_fill(~cells, float('inf')).argmin(-1)
    offset = torch.stack((index % 14, index // 14), -1)
    uv = patch_grid(geometry.device)[None]-6.5+offset
    flat_index = uv[..., 1].long()*224+uv[..., 0].long()
    reference = geometry[:, 2:7].flatten(2).gather(2, flat_index[:, None].expand(-1, 5, -1)).transpose(1, 2)
    available = cells.any(-1) & cad_valid
    reference = torch.where(available[..., None], reference, 0.)
    return dict(uv=uv, xyz=reference[..., 2:5], depth=reference[..., :1], available=available)


class FlowReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.descriptor = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 64))
        self.endpoint = nn.Sequential(nn.Linear(128+3+2, 128), nn.GELU(), nn.Linear(128, 4))
        nn.init.zeros_(self.endpoint[-1].weight)
        nn.init.zeros_(self.endpoint[-1].bias)
        self.write = nn.Sequential(nn.LayerNorm(72), nn.Linear(72, 256), nn.GELU(), nn.Linear(256, 256))
        nn.init.normal_(self.write[-1].weight, std=.001)
        nn.init.zeros_(self.write[-1].bias)
        # Inference-only causal interventions; not confidence-based eval masks.
        self.disable_transport = False
        self.transport_gain = 1.
        self.disable_recovery_feedback = False

    @torch.autocast('cuda', enabled=False)
    def forward(self, patch, observed, cad, geometry, valid, reference, recovered=None):
        grid = patch_grid(patch.device)
        source = F.normalize(self.descriptor(cad.float()), dim=-1)
        current = F.normalize(self.descriptor(patch.float()), dim=-1)
        measured = F.normalize(self.descriptor(observed.float()), dim=-1)
        appearance = (source @ (.5*current+.5*measured).transpose(-1, -2))/.1
        # A broad estimated-coordinate prior, not a correspondence label. Global
        # feature CE below excludes this prior and the predicted-geometry term.
        distance = (reference['uv'][:, :, None]-grid[None, None]).square().sum(-1)
        scores = appearance-distance/(2*56.**2)
        feedback = torch.zeros_like(scores)
        if recovered is not None and not self.disable_recovery_feedback:
            recovered_xyz = F.avg_pool2d(recovered[:, :3].float(), 14).flatten(2).transpose(1, 2)
            # Inferred geometry is weaker than observed descriptors. Detached
            # trust cannot reduce its own supervised loss or manufacture support.
            trust = .5*F.avg_pool2d(recovered[:, 4:5].float().sigmoid(), 14).flatten(1).detach()
            d2 = (reference['xyz'][:, :, None]-recovered_xyz[:, None]).square().sum(-1)
            feedback = -(d2/(2*.1**2)).clamp_max(4.)*trust[:, None]
            scores = scores+feedback
        scores = scores.masked_fill(~valid[:, None], -1e4)
        # Local soft argmax avoids averaging two far-away ambiguous matches.
        peak = scores.detach().argmax(-1)
        near = (grid[None, None]-grid[peak][:, :, None]).abs().amax(-1) <= 14
        probability = scores.masked_fill(~near, -1e4).softmax(-1)
        coarse_uv = probability @ grid
        matched = probability @ current
        endpoint_features = torch.cat((source, matched, reference['xyz'], reference['uv']/224.), -1)
        raw = self.endpoint(endpoint_features)
        uv = coarse_uv+14*raw[..., :2].tanh()
        local_metrics = {}
        if getattr(self, 'capture_local_flow', False) or hasattr(self, 'local_flow_head'):
            from .local_flow import local_flow_features
            features = local_flow_features(endpoint_features,coarse_uv,reference,measured,scores,peak)
            local_metrics['local_flow_features'] = features.detach()
            if hasattr(self, 'local_flow_head'):
                uv = coarse_uv+self.local_flow_head(features)
        logits = raw[..., 2:]
        logp = scores.log_softmax(-1)
        entropy = -(logp.exp()*logp).sum(-1)/math.log(256)
        evidence_metrics = {}
        if getattr(self, 'capture_point_evidence', False) or hasattr(self, 'point_evidence_head'):
            from .point_evidence import point_evidence_features
            features = point_evidence_features(source,current,measured,reference,uv,geometry,recovered,entropy,logits)
            evidence_metrics['point_evidence_features'] = features
            evidence_metrics['legacy_visible_logits'] = logits[..., 1]
            if hasattr(self, 'point_evidence_head'):
                evidence = self.point_evidence_head(features)
                logits = torch.stack((logits[..., 0],evidence[..., 0]),-1)
                evidence_metrics['point_quality_logits'] = evidence[..., 1]

        # Forward soft splatting: template endpoints -> observation patches.
        # This is not backward grid_sample with a forward-flow field.
        weights = (-(uv[:, :, None]-grid[None, None]).square().sum(-1)/(2*14.**2)).exp()
        trust = (.1+.9*logits[..., 0].sigmoid())*reference['available']
        weights = weights*trust[..., None]*valid[:, None]
        mass = weights.sum(1)
        values = torch.cat((source, reference['xyz'], reference['depth']), -1)
        aligned = (weights.transpose(1, 2) @ values)/mass[..., None].clamp_min(1e-6)
        observed_valid = geometry[:, 1:2].float().clamp(0, 1)
        observed_mass = F.avg_pool2d(observed_valid, 14).flatten(1)
        observed_depth = (F.avg_pool2d(geometry[:, :1].float()*observed_valid, 14).flatten(1)
                          /observed_mass.clamp_min(1e-6))
        residual = (aligned[..., -1]-observed_depth)*observed_mass
        context = torch.cat((aligned, mass[..., None].clamp_max(4), observed_depth[..., None],
                             observed_mass[..., None], residual[..., None]), -1)
        write = (.1*self.transport_gain)*self.write(context)*(mass/(mass+1))[..., None]*valid[..., None]
        if self.disable_transport: write = write*0
        return patch+write.to(patch.dtype), dict(
            uv=uv, flow=uv-reference['uv'], scores=appearance.masked_fill(~valid[:, None], -1e4),
            coarse_uv=coarse_uv, endpoint_delta=uv-coarse_uv, peak_uv=grid[peak],
            support_logits=logits[..., 0], visible_logits=logits[..., 1], entropy=entropy,
            aligned_mass=mass, geometry_feedback=feedback, write=write, **evidence_metrics, **local_metrics)


@torch.no_grad()
def flow_labels(output, target, crop_k, diameter, visible):
    from .cad_image_correspondence import image_correspondence_targets
    reference = output['flow_reference']
    # Reuse the audited GT projection, front-surface/edge checks and disjoint
    # visible / real-hidden / CAD-proxy ownership; no labels enter the model.
    return image_correspondence_targets(dict(cad_image_xyz=reference['xyz'],
        cad_image_available=reference['available']), target, crop_k, diameter, visible)


@torch.autocast('cuda', enabled=False)
def flow_reconstruction_loss(output, labels):
    from .cad_image_correspondence import balanced_binary_loss
    grid = patch_grid(labels['uv'].device)
    heat = (-(labels['uv'][:, :, None]-grid[None, None]).square().sum(-1)/(2*7.**2)).softmax(-1)
    def mean(value, mask):
        count = mask.sum(-1)
        per = (value*mask).sum(-1)/count.clamp_min(1)
        return (per*(count > 0)).sum()/(count > 0).sum().clamp_min(1)
    loss = output['surface_xyz'].sum()*0
    metrics = {}
    for stage, result in enumerate(output['flow_rounds']):
        ce = -(heat*result['scores'].float().log_softmax(-1)).sum(-1)
        endpoint = F.smooth_l1_loss(result['uv'].float()/224, labels['uv']/224, beta=1/224, reduction='none').sum(-1)
        error = (result['uv'].detach()-labels['uv']).norm(dim=-1)
        stage_loss = endpoint.sum()*0
        for name, factor in [('observed', 1.), ('real', 1.), ('proxy', .5)]:
            mask = labels[name]
            stage_loss = stage_loss+factor*mean(10*endpoint+.05*ce, mask)
            metrics[f'flow{stage}_{name}_epe'] = mean(error, mask).detach()
            metrics[f'flow_{name}_count'] = mask.sum().detach()
        for name in ('support', 'visible'):
            stage_loss = stage_loss+.05*balanced_binary_loss(result[name+'_logits'], labels[name], labels['known_'+name])
        loss = loss+(.5 if stage == 0 else 1.)*stage_loss
    return loss, metrics

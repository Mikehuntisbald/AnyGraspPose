"""V21: decoded completion is the only visual input to the pose readout.

The pose branch has no access to JEPA patch latents, FP features or teacher
tensors. Measured camera depth has priority at predicted visible object pixels.
Appearance routing is explicit: legacy observed-visible or all JEPA-decoded. Canonical surface coordinates are always predicted: observed
camera XYZ transformed by the *estimated* pose is not a correspondence target.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .cad_surface import CADSurfaceTracker


def pack_completion(mid, last, surface, geometry, rays, base, diameter,
                    visibility, support, observed_mid, observed_last, valid, measured_depth_m,
                    feature_source='observed_visible', measurement_probability=None,
                    completion_mass_ratio=None):
    """Prediction-only dense correspondences, camera points centered at base t/d."""
    if feature_source not in ('observed_visible','decoded'):
        raise ValueError('Unknown serial readout feature source')
    with torch.autocast(surface.device.type, enabled=False):
        bounds = valid.reshape(-1, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
        expand = lambda x: x.reshape(-1, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
        vis = visibility.float().sigmoid()
        obj = expand(support.float().sigmoid()) * bounds
        if measurement_probability is None:
            measurement_confidence = expand(vis)
        else:
            if measurement_probability.shape != measured_depth_m.shape:
                raise ValueError('Dense measurement probability must match the depth image')
            # A separately validated sensor-ownership decision. Pose gradients
            # must not train the selector to call an occluder a measurement.
            measurement_confidence = measurement_probability.detach().float().nan_to_num(nan=0.,posinf=0.,neginf=0.).clamp(0,1)
        measured_mask = (measured_depth_m > 0) & torch.isfinite(measured_depth_m) & bounds & (measurement_confidence >= .7)
        # Hard source ownership; confidence remains soft. No GT visibility.
        measured = measurement_confidence * measured_mask
        completed = .5 * (~measured_mask) * surface[:, 4:5].float().sigmoid().nan_to_num() * obj
        if completion_mass_ratio is not None:
            if completion_mass_ratio <= 0:
                raise ValueError('Completion mass ratio must be positive')
            measured_mass=measured.sum((1,2,3),keepdim=True)
            completed_mass=completed.sum((1,2,3),keepdim=True)
            # Per-pixel confidence alone does not limit aggregate influence:
            # thousands of hypotheses can overwhelm a few real measurements.
            # If depth is completely absent, retain the RGB/CAD completion path.
            factor=(completion_mass_ratio*measured_mass/completed_mass.clamp_min(1e-8)).clamp_max(1.)
            completed=completed*torch.where(measured_mass>0,factor,1.)
        weight = measured + completed
        measured_residual = (measured_depth_m.float() - base[:,2,3,None,None,None].float()) / diameter[:,None,None,None]
        depth = torch.where(measured_mask, measured_residual, surface[:, 3:4].float())
        absolute_d = depth + base[:, 2, 3, None, None, None].float() / diameter[:, None, None, None]
        camera = absolute_d * rays.float() - base[:, :3, 3, None, None].float() / diameter[:, None, None, None]
        finite = torch.isfinite(camera).all(1, keepdim=True) & torch.isfinite(surface[:, :3]).all(1, keepdim=True) & (absolute_d > 0)
        weight = weight * finite
        xyz = torch.nan_to_num(surface[:, :3].float())
        camera = torch.nan_to_num(camera)
        # Feature routing never introduces a teacher or a direct render feature.
        if feature_source == 'decoded':
            # Every visual feature has passed through the unified JEPA. Keeping
            # real RGB input does not require overwriting its fused output.
            feature = torch.where(valid[...,None],torch.cat((mid,last),-1),0.)
        else:
            # Legacy path retained only for exact older-experiment reproduction.
            vm = (vis[..., None] >= .7) & valid[..., None]
            feature = torch.cat((torch.where(vm, observed_mid, mid),
                                 torch.where(vm, observed_last, last)), -1)
    return dict(feature=feature, xyz=xyz, camera=camera, weight=weight,
                measured_weight=measured, completed_weight=completed)


class CompletionRelations(nn.Module):
    """Local correspondences plus signed rotation/translation moment token.

Products are formed before pooling; averaging XYZ before forming cross
products would discard the within-patch rotation signal. There is no solver
or delta-pose shortcut: object attention and the learned head read these tokens.
"""
    def __init__(self, conditioning='legacy'):
        super().__init__()
        if conditioning not in ('legacy', 'unit_gate', 'shape_conditioned'):
            raise ValueError('Unknown completion relation conditioning')
        self.conditioning = conditioning
        self.appearance = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.relation = nn.Sequential(nn.Linear(17, 128), nn.GELU(), nn.Linear(128, 128))
        self.combine = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU())
        self.moments = nn.Sequential(nn.Linear(24, 256), nn.GELU(), nn.Linear(256, 256))
        self.norm = nn.LayerNorm(256)
        if conditioning == 'shape_conditioned':
            self.precondition = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 256))
            nn.init.zeros_(self.precondition[-1].weight)
            nn.init.zeros_(self.precondition[-1].bias)

    def forward(self, packet, base):
        with torch.autocast(base.device.type, enabled=False):
            x = packet['xyz'].float().flatten(2).transpose(1, 2)
            y = packet['camera'].float().flatten(2).transpose(1, 2)
            w = packet['weight'].float().flatten(1)
            p = x @ base[:, :3, :3].float().transpose(-1, -2)
            residual = y - p
            cross = torch.linalg.cross(p, residual)
            local = torch.cat((x, p, residual, cross, y, w[..., None],
                               packet['measured_weight'].flatten(1)[..., None]), -1)
            mass = F.avg_pool2d(packet['weight'].float(), 14, 14).flatten(1)
            grid = local.transpose(1, 2).reshape(len(x), 17, 224, 224)
            pooled = F.avg_pool2d(grid * packet['weight'], 14, 14).flatten(2).transpose(1, 2)
            pooled = pooled / mass[..., None].clamp_min(1e-6)
            wn = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
            mean_p = (p * wn[..., None]).sum(1)
            mean_y = (y * wn[..., None]).sum(1)
            pc, yc = p - mean_p[:, None], y - mean_y[:, None]
            scatter = pc.transpose(1, 2) @ (pc * wn[..., None])
            cov = pc.transpose(1, 2) @ (yc * wn[..., None])
            # Centering decouples translation from rotation. Scaling by object
            # spread prevents object size from suppressing the signed signal.
            scale = scatter.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-4)
            moments = torch.cat((scatter.flatten(1) / scale[:, None],
                (cov - scatter).flatten(1) / scale[:, None], mean_p, mean_y - mean_p), -1)
            residual_scale = ((residual.square().sum(-1) * wn).sum(-1) + 1e-12).sqrt().div(.1).clamp_max(3.)
        tokens = self.combine(torch.cat((self.appearance(packet['feature']), self.relation(pooled)), -1))
        global_token = self.moments(moments)[:, None]
        if self.conditioning == 'shape_conditioned':
            from .shape_conditioned import normalized_relation
            signal = normalized_relation(scatter, cov, mean_p, mean_y)
            global_token = global_token + self.precondition(signal)[:, None]
        if self.conditioning != 'legacy':
            # An evidence/zero-residual gate, not a shape-dependent multiplier
            # of every correction. Keep the legacy graph exact by default.
            residual_scale = ((residual_scale - 1e-5).clamp_min(0) * 100).tanh()
        tokens = self.norm(torch.cat((tokens, global_token), 1))
        token_valid = torch.cat((mass > 1e-5, (w.sum(-1) > 1e-5)[:, None]), 1)
        return tokens, token_valid, moments, residual_scale


class SerialCompletionTracker(CADSurfaceTracker):
    architecture_id = 'stream_serial_completion_jepa_v21'
    model_version = 'serial-decoded-completion-pose-v21'
    cache_contract = 'observed-only-serial-completion-v21'

    def __init__(self, encoder, cached_utonia):
        super().__init__(encoder, cached_utonia)
        self.geometry_readout = CompletionRelations()
        # Reuse neither the ineffective V20 readout nor its small geometry gate.
        self.migration.update(readout='decoded appearance + canonical/camera correspondences -> object query -> pose; no patch skip')

    def extra_frame_inputs(self, obs, memory=None, history_enabled=None):
        state, history, _, cad = super().extra_frame_inputs(obs, memory, history_enabled)
        if obs.crop_rays is None:
            raise ValueError('Serial pose requires measured crop rays')
        return state, history, (obs.crop_rays, obs.diameter, obs.mid, obs.last, obs.measured_depth_m), cad

    def read_completion(self, packet, base):
        tokens, mask, moments, scale = self.geometry_readout(packet, base)
        obj = self.query.expand(len(base), -1, -1)
        obj = obj + self.object_attn(self.object_norm(obj), tokens, mask)
        return obj, dict(serial_moments=moments, serial_token_valid=mask,pose_evidence_available=mask.any(-1),serial_residual_scale=scale,
            serial_measured_weight=packet['measured_weight'], serial_completed_weight=packet['completed_weight'])

    def read_object(self, patch, valid, surface, geometry_image, base, cad_valid,
                    evidence_logits, readout_inputs):
        return self.read_decoded_object(patch, valid, surface, geometry_image, base, cad_valid,
            evidence_logits, readout_inputs, self.core.feature_mid(patch), self.core.feature_last(patch))

    def read_decoded_object(self, patch, valid, surface, geometry_image, base, cad_valid,
                            evidence_logits, readout_inputs, decoded_mid, decoded_last):
        rays, diameter, mid, last, measured_depth = readout_inputs
        packet = pack_completion(decoded_mid, decoded_last,
            surface, geometry_image, rays, base, diameter, evidence_logits,
            self.core.support(patch).squeeze(-1), mid, last, valid, measured_depth,
            feature_source=getattr(self,'readout_feature_source','observed_visible'))
        return self.read_completion(packet, base)


def migrate_serial(model, source):
    """Keep the trained completion backbone; reset the entire learned pose readout."""
    from .reconstruction_only import is_pose_parameter
    shared = {k: v for k, v in source.items() if not is_pose_parameter(k)}
    result = model.load_state_dict(shared, strict=False)
    if result.unexpected_keys or any(not is_pose_parameter(k) for k in result.missing_keys):
        raise ValueError(f'Unexpected serial migration: {result}')
    return list(result.missing_keys)

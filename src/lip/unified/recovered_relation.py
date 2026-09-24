"""Recovered geometry participates in the single object-query pose readout.

Observed points and predicted completion retain separate types and confidence.
Canonical CAD correspondences search all rendered patches, not the same pixel.
No teacher, FP feature, pose solver, or completed geometry enters the writer.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .supported_history import SupportedHistoryTracker


def patch_pool(value, weight):
    mass = F.avg_pool2d(weight.float(), 14, 14)
    pooled = F.avg_pool2d(torch.nan_to_num(value.float()) * weight, 14, 14)
    pooled = pooled / mass.clamp_min(1e-6)
    return pooled.flatten(2).transpose(1, 2), mass.flatten(1)


def cad_correspondence(source, reference, reference_depth, valid):
    """Soft surface association in canonical coordinates, normalized by diameter."""
    with torch.autocast(source.device.type, enabled=False):
        source, reference = source.float(), reference.float()
        distance = (source[:, :, None] - reference[:, None]).square().sum(-1)
        logits = (-distance / (2 * .1 ** 2)).masked_fill(~valid[:, None], float('-inf'))
        # No finite sentinel can safely mask a key for arbitrary depth errors.
        # Empty references use a finite all-zero row before masking all values.
        logits = torch.where(valid.any(-1)[:, None, None], logits, 0.)
        weights = logits.softmax(-1) * valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        xyz = weights @ reference
        depth = (weights @ reference_depth[..., None]).squeeze(-1)
        error = (weights * distance).sum(-1)
    return xyz, depth, error, valid.any(-1)[:, None].expand_as(depth)


class RecoveredGeometryRelation(nn.Module):
    def __init__(self):
        super().__init__()
        self.measured = nn.Sequential(nn.Linear(12, 128), nn.GELU(), nn.Linear(128, 256))
        self.completed = nn.Sequential(nn.Linear(12, 128), nn.GELU(), nn.Linear(128, 256))
        # Nonzero from the first update: pose gradients can train the recovery heads.
        self.gain = nn.Parameter(torch.tensor(-3.))

    def forward(self, patch, surface, geometry, base, valid, cad_valid,
                visibility, observed_xyz, observed_valid):
        with torch.autocast(patch.device.type, enabled=False):
            bounds = valid[:, None].reshape(-1, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
            reference_mask = ((geometry[:, 3:4] > 0) & bounds &
                              torch.isfinite(geometry[:, 2:7]).all(1, keepdim=True))
            reference, ref_mass = patch_pool(geometry[:, 4:7], reference_mask)
            ref_depth, _ = patch_pool(geometry[:, 2:3], reference_mask)
            ref_valid = (ref_mass > 0) & cad_valid & valid
            # A dropped CAD must be unable to leak through either values or masks.
            reference = torch.where(ref_valid[..., None], reference, 0.)
            ref_depth = torch.where(ref_valid[..., None], ref_depth, 0.)[..., 0]

            finite = torch.isfinite(surface[:, :4]).all(1, keepdim=True)
            probability = torch.nan_to_num(surface[:, 4:5], nan=-100.).sigmoid() * finite * bounds
            restored, completion_mass = patch_pool(surface[:, :4], probability)
            recovered_xyz, recovered_depth = restored[..., :3], restored[..., 3]
            measured_mask = (geometry[:, 1:2] > 0) & bounds & torch.isfinite(geometry[:, :1])
            measured_depth, measured_mass = patch_pool(geometry[:, :1], measured_mask)
            observed_valid = observed_valid & valid & (measured_mass > 0) & torch.isfinite(observed_xyz).all(-1)
            observed_xyz = torch.where(observed_valid[..., None], observed_xyz.float(), 0.)
            measured_depth = torch.where(observed_valid, measured_depth[..., 0], 0.)
            measured_weight = visibility.float().sigmoid() * measured_mass * observed_valid
            # Completion is a bounded hypothesis, subordinate to visible measurements.
            completed_weight = .5 * completion_mass * (1 - measured_weight) * valid

            def relation(xyz, depth, confidence):
                match, match_z, distance, available = cad_correspondence(xyz, reference, ref_depth, ref_valid)
                # Both source types are canonical XYZ; this is the camera-Z
                # implied by applying the current estimated rotation to that XYZ.
                implied_z = (xyz * base[:, None, 2, :3].float()).sum(-1)
                residual = (xyz - match) * available[..., None]
                delta_depth = (depth - match_z) * available
                return torch.cat((xyz, residual, depth[..., None], delta_depth[..., None],
                                  (depth - implied_z)[..., None], distance[..., None],
                                  available.float()[..., None], confidence[..., None]), -1)

            measured_relation = relation(observed_xyz, measured_depth, measured_weight)
            completed_relation = relation(recovered_xyz, recovered_depth, completed_weight)
        evidence = (self.measured(measured_relation) * measured_weight[..., None] +
                    self.completed(completed_relation) * completed_weight[..., None])
        pose_patch = patch + self.gain.sigmoid() * evidence.to(patch.dtype)
        return pose_patch, dict(pose_relation_latent=pose_patch,
            relation_measured_weight=measured_weight, relation_completed_weight=completed_weight,
            relation_recovered_xyz=recovered_xyz, relation_recovered_depth=recovered_depth,
            relation_cad_valid=ref_valid)


class RecoveredRelationTracker(SupportedHistoryTracker):
    architecture_id = 'stream_recovered_relation_jepa_v11'
    model_version = 'recovered-geometry-observed-constraint-v11'
    cache_contract = 'observed-only-supported128-recovered-readout-v11'

    def __init__(self, encoder, cached_utonia):
        super().__init__(encoder, cached_utonia)
        self.geometry_readout = RecoveredGeometryRelation()
        self.weights_version = self.model_version + '/initial'
        self.migration.update(architecture=self.architecture_id,
            readout='shared JEPA patch -> recovered geometry with measured constraints -> object query -> pose')

    def extra_frame_inputs(self, obs, memory=None, history_enabled=None):
        return obs.state, (), (obs.object_xyz, obs.depth_valid)

    def read_object(self, patch, valid, surface, geometry_image, base, cad_valid,
                    evidence_logits, readout_inputs):
        pose_patch, diagnostics = self.geometry_readout(patch, surface, geometry_image, base,
            valid, cad_valid, evidence_logits, *readout_inputs)
        obj = self.query.expand(len(patch), -1, -1)
        return obj + self.object_attn(self.object_norm(obj), pose_patch, valid), diagnostics


def initialize_from_supported(model, source_state):
    """Explicit candidate initialization. New readout gets no inherited Adam state."""
    old = {k: v for k, v in model.state_dict().items()
           if not k.startswith(('geometry_readout.', 'encoder.'))}
    supplied = {k: v for k, v in source_state.items() if not k.startswith('encoder.')}
    if old.keys() != supplied.keys():
        raise ValueError('Expected exact supported-history core; no loose checkpoint migration')
    result = model.load_state_dict(source_state, strict=False)
    if result.unexpected_keys or any(not k.startswith(('geometry_readout.', 'encoder.')) for k in result.missing_keys):
        raise ValueError('Unexpected recovered-readout migration keys')
    model.migration.update(previous_experiment_weights_loaded=True,
        initialization='supported-history core exact; geometry_readout random, nonzero small gain',
        optimizer_reset=True, memory_reset=True)

"""V10: spatial observation/reference encoders and appearance-geometry attention.

The paired CNN design follows the shared-then-joint encoding idea in
FoundationPose. These modules are independently initialized, use GroupNorm,
and do not load FoundationPose code or weights. The V9 teacher, memory writer,
four JEPA blocks and final-patch-only pose interface are retained.
"""
import torch
from torch import nn
from torch.nn import functional as F

from lip.geometry.so3 import original_pose, update
from lip.jepa.predictor import SafeAttention
from .two_stream import TwoStreamTracker


class ConvNormAct(nn.Sequential):
    def __init__(self, incoming, outgoing, kernel=3, stride=1):
        super().__init__(
            nn.Conv2d(incoming, outgoing, kernel, stride, kernel // 2, bias=False),
            nn.GroupNorm(8, outgoing), nn.GELU())


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels))

    def forward(self, value):
        return F.gelu(value + self.branch(value))


class AppearancePair(nn.Module):
    """Shared spatial processing before joint observation/render processing."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            ConvNormAct(256, 128), ResidualConv(128), ResidualConv(128))
        self.joint = nn.Sequential(ResidualConv(256), ResidualConv(256))

    def forward(self, real, cad, valid, cad_valid):
        batch = len(real)
        mask = torch.cat((valid, cad_valid), 0)[:, None].reshape(2 * batch, 1, 16, 16)
        value = torch.cat((real, cad), 0).transpose(1, 2).reshape(2 * batch, 256, 16, 16)
        # Mask again after spatial processing: biases/norms must not turn a
        # dropped reference into a valid reference feature map.
        observed, reference = (self.shared(value * mask) * mask).chunk(2)
        return self.joint(torch.cat((observed, reference), 1)).flatten(2).transpose(1, 2)


class GeometryPair(nn.Module):
    """Same nine physical channels as V9, with separate stems and a shared CNN.

    Observation: normalized depth and validity (2 channels).
    Reference: normalized depth, silhouette and object XYZ (5 channels).
    Relation: depth difference and joint validity (2 channels).
    """
    def __init__(self):
        super().__init__()
        self.observed = ConvNormAct(2, 32, kernel=7, stride=2)
        self.reference = ConvNormAct(5, 32, kernel=7, stride=2)
        self.shared = nn.Sequential(
            ConvNormAct(32, 64, stride=2), ResidualConv(64), ResidualConv(64))
        self.relation = ConvNormAct(2, 16, kernel=7, stride=4)
        self.joint = nn.Sequential(
            ConvNormAct(144, 128, stride=2), ResidualConv(128), ResidualConv(128),
            ConvNormAct(128, 256, kernel=1))

    def forward(self, image, valid, cad_valid, pool):
        batch = len(image)
        bounds = valid.reshape(batch, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
        cad_bounds = cad_valid.reshape(batch, 1, 16, 16).repeat_interleave(14, -2).repeat_interleave(14, -1)
        observation = image[:, :2] * bounds
        reference = image[:, 2:7] * cad_bounds
        relation = image[:, 7:9] * bounds * cad_bounds
        both = self.shared(torch.cat((self.observed(observation), self.reference(reference)), 0))
        observed, rendered = both.chunk(2)
        observed = observed * F.max_pool2d(bounds.float(), 4)
        rendered = rendered * F.max_pool2d(cad_bounds.float(), 4)
        paired = self.joint(torch.cat((observed, rendered, self.relation(relation)), 1))
        tokens = (pool @ paired @ pool.T).flatten(2).transpose(1, 2)
        evidence = (observation[:, 1:2] > 0) | (reference[:, 1:2] > 0)
        token_valid = (F.max_pool2d(evidence.float(), 14).flatten(1) > 0) & valid
        return tokens, token_valid


class AppearanceGeometryAttention(nn.Module):
    """Appearance-grid queries read geometry at any valid grid position.

    A learned null key/value is always available. The appearance residual is
    retained; invalid geometry is masked while the state remains available.
    Spatial positions are used on both sides, not a same-index attention mask.
    """
    def __init__(self):
        super().__init__()
        self.appearance_norm = nn.LayerNorm(256)
        self.geometry_norm = nn.LayerNorm(256)
        self.state = nn.Sequential(nn.Linear(24, 256), nn.GELU(), nn.Linear(256, 256))
        self.state_type = nn.Parameter(torch.zeros(1, 1, 256))
        self.state_norm = nn.LayerNorm(256)
        self.attention = SafeAttention(256, 8)
        self.null_geometry = nn.Parameter(torch.zeros(1, 1, 256))
        self.gate = nn.Linear(256, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.ff = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024),
                                nn.GELU(), nn.Linear(1024, 256))

    def forward(self, appearance, geometry, position, geometry_valid, state):
        query = self.appearance_norm(appearance + position)
        source = torch.cat((self.geometry_norm(geometry + position),
                            self.state_norm(self.state(state)[:, None] + self.state_type),
                            self.null_geometry.expand(len(appearance), -1, -1)), 1)
        valid = torch.cat((geometry_valid, torch.ones_like(geometry_valid[:, :2])), 1)
        update = self.attention(query, source, valid)
        fused = appearance + self.gate(query).sigmoid() * update
        return fused + self.ff(fused)


class ConvCrossTracker(TwoStreamTracker):
    architecture_id = 'stream_conv_cross_jepa_v10'
    model_version = 'conv-pair-cross-modal-state-clean-v10'
    cache_contract = 'observed-local128-conv-cross-state-v10'

    def __init__(self, encoder, cached_utonia):
        super().__init__(encoder, cached_utonia)
        del self.appearance, self.geometry, self.fusion
        self.appearance_pair = AppearancePair()
        self.geometry_pair = GeometryPair()
        self.cross_fusion = AppearanceGeometryAttention()
        self.migration.update(
            architecture=self.architecture_id,
            pair_encoding='independent trainable shared/joint residual CNNs',
            fusion='appearance queries, spatial geometry plus 24D state keys/values, residual and null option',
            foundationpose_weights_loaded=False)

    def extra_frame_inputs(self, obs, memory=None, history_enabled=None):
        return (obs.state,)

    def fuse_history(self, patch, base, diameter, state, valid, geometry_image, cad_valid, history_inputs):
        return patch, {}

    def history_sources(self, mem, mv, mb):
        return mem, mv, mb

    def read_surface(self,patch,valid,inputs,layer):
        return patch,{}

    def prepare_surface_read(self,patch,valid,inputs,levels,layer):
        return inputs,{}

    def refine_surface(self,patch,valid,inputs,levels,before_last,mem,mv,mb):
        return patch,levels,{}

    def read_object(self, patch, valid, surface, geometry_image, base, cad_valid,
                    evidence_logits, readout_inputs):
        obj = self.query.expand(len(patch), -1, -1)
        return obj + self.object_attn(self.object_norm(obj), patch, valid), {}

    def read_decoded_object(self, patch, valid, surface, geometry_image, base, cad_valid,
                            evidence_logits, readout_inputs, decoded_mid, decoded_last):
        return self.read_object(patch, valid, surface, geometry_image, base, cad_valid,
                                evidence_logits, readout_inputs)

    def tensor_frame(self, mid, last, cad_mid, cad_last, cad_valid, geometry_image, static_cad,
                     base, diameter, center, position, prompt, dt, valid, mem, mv, mb, state,
                     history_inputs=(), readout_inputs=(), cad_inputs=()):
        batch = len(mid)
        real = self.core.src_proj(torch.cat((mid, last), -1))
        cad = self.core.src_proj(torch.cat((cad_mid, cad_last), -1))
        appearance = self.appearance_pair(real, cad, valid, cad_valid)
        geometry, geometry_valid = self.geometry_pair(geometry_image, valid, cad_valid, self.geometry_pool)
        geometry = geometry + self.cad_prior(static_cad)[:, None] * cad_valid.any(-1)[:, None, None]
        patch = self.cross_fusion(appearance, geometry, position, geometry_valid, state)
        patch, history_metrics = self.fuse_history(patch, base, diameter, state, valid, geometry_image, cad_valid, history_inputs)
        mem, mv, mb = self.history_sources(mem, mv, mb)
        patch = patch + position + prompt[:, None] + dt[:, None]
        surface_metrics={}
        dense_levels=[]
        use_dpt=getattr(self,'surface_decoder_kind','mlp')=='dpt'
        before_last=patch
        for layer,block in enumerate(self.core.blocks):
            if layer==3:before_last=patch
            read_inputs,stage_metrics=self.prepare_surface_read(patch,valid,cad_inputs,dense_levels,layer)
            surface_metrics.update(stage_metrics)
            patch,read_metrics=self.read_surface(patch,valid,read_inputs,layer)
            surface_metrics.update(read_metrics)
            patch = block(patch, valid, mem, mv, mb)
            if use_dpt:dense_levels.append(patch)
        patch,dense_levels,refined_metrics=self.refine_surface(patch,valid,cad_inputs,dense_levels,before_last,mem,mv,mb)
        surface_metrics.update(refined_metrics)
        patch = self.core.final_norm(patch)
        if use_dpt:
            dense_levels[-1]=patch
            if hasattr(self,'cad_atlas_decoder'):
                dense=self.surface_head.dense_features(dense_levels,valid)
                surface=self.surface_head.output[-1](dense).float()
                surface,atlas_metrics=self.cad_atlas_decoder(dense,surface,*cad_inputs[-3:])
                surface_metrics.update(atlas_metrics)
            elif hasattr(self,'cad_transport'):
                dense=self.surface_head.dense_features(dense_levels,valid)
                surface=self.surface_head.output[-1](dense).float()
                surface,transport_metrics=self.cad_transport(dense,surface,geometry_image,cad_valid)
                surface_metrics.update(transport_metrics)
            else:
                surface=self.surface_head(dense_levels,valid)
        else:
            surface = self.surface_head(patch).reshape(batch, 16, 16, 14, 14, 5)
            surface = surface.permute(0, 5, 1, 3, 2, 4).reshape(batch, 5, 224, 224).float()
        surface_xyz, surface_depth, surface_validity = surface[:, :3], surface[:, 3:4], surface[:, 4:5]
        logits = self.visibility(real).squeeze(-1)
        decoded_mid, decoded_last = self.core.feature_mid(patch), self.core.feature_last(patch)
        obj, relation_metrics = self.read_decoded_object(
            patch, valid, torch.cat((surface_xyz, surface_depth, surface_validity), 1),
            geometry_image, base, cad_valid, logits, readout_inputs, decoded_mid, decoded_last)
        latent = F.layer_norm(obj[:, 0], (256,))
        delta = self.head(latent).float()
        if 'pose_evidence_available' in relation_metrics:
            delta = delta * relation_metrics['pose_evidence_available'][:, None]
        if 'serial_residual_scale' in relation_metrics:
            delta = delta * relation_metrics['serial_residual_scale'][:, None]
        with torch.autocast(delta.device.type, enabled=False):
            pose = update(base.float(), delta[:, :3], delta[:, 3:], diameter.float())
            original = original_pose(pose, center.float())
        return dict(
            latent=latent, latent_object=latent, latent_context=real.mean(1), patch_latent=patch,
            pose_centered=pose, pose_original=original, delta_rotvec=delta[:, :3], delta_center_norm=delta[:, 3:],
            f_predicted=decoded_last, f_mid_predicted=decoded_mid,
            evidence_logits=logits, support_logits=self.core.support(patch).squeeze(-1),
            log_feature_error=self.core.log_error(patch).squeeze(-1).clamp(-14, 5),
            surface_xyz=surface_xyz, surface_depth_residual=surface_depth, geometry_valid_logits=surface_validity,
            surface_depth_m=base[:, 2, 3, None, None, None].float() + surface_depth * diameter[:, None, None, None].float(),
            writer_h=real, writer_p=logits.sigmoid() * valid, **history_metrics, **relation_metrics, **surface_metrics)

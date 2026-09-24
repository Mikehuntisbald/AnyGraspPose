"""FP online geometry, cached whole-CAD prior, one compilable JEPA pose core."""
import torch
from torch import nn
from torch.nn import functional as F
from lip.geometry.so3 import update, original_pose
from .model import UnifiedTracker


class FPUnifiedTracker(UnifiedTracker):
    architecture_id = 'stream_dino_fp_staticutonia_jepa_rgbd_v3'
    model_version = 'unified-fp-staticcad-rgbd-v3'
    cache_contract = 'observed-local128-dino-fp-staticutonia-v3'
    online_geometry = 'foundationpose'

    def __init__(self, encoder, cached_utonia, parent, fp):
        super().__init__(encoder, cached_utonia, parent)
        self.fp = fp
        self.fp_observation_proj = nn.Linear(128, 256)
        self.fp_pair_proj = nn.Linear(512, 256)
        nn.init.zeros_(self.fp_pair_proj.weight); nn.init.zeros_(self.fp_pair_proj.bias)
        self.compiled_frame = None
        self.pose_relation_fusion = False
        self.capture_pose_inputs = False
        self.pose_patch_only = False
        self.pose_final_patch_only = False
        self.jepa_pose_geometry = False
        self.pair_into_patch = False
        self.fp_observation_only = False

    def disable_pair_features(self):
        """Remove paired FP modules; retain observation encoding and JEPA geometry."""
        if self.pose_relation_fusion or self.pair_into_patch:
            raise ValueError('Observation-only FP cannot enable paired feature fusion')
        del self.fp_pair_proj
        if hasattr(self.fp, 'encoder') and 'encodeAB' in self.fp.encoder:
            del self.fp.encoder['encodeAB']
        self.fp_observation_only = True

    def enable_jepa_pose_geometry(self):
        if not self.pose_patch_only or self.pose_relation_fusion:
            raise ValueError('Internal pose geometry requires a strictly patch-only readout')
        device=self.fp_observation_proj.weight.device
        self.pose_dense_proj=nn.Linear(15,256).to(device)
        self.pose_cad_proj=nn.Linear(15,256).to(device)
        self.pose_state_proj=nn.Linear(24,256).to(device)
        self.pose_error_head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,6)).to(device)
        for m in (self.pose_dense_proj,self.pose_cad_proj,self.pose_state_proj,self.pose_error_head[-1]):
            nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
        self.jepa_pose_geometry=True

    def enable_pose_relation(self):
        if self.fp_observation_only:
            raise ValueError('Observation-only FP cannot restore a paired readout bypass')
        if not hasattr(self,'pose_relation_proj'):
            self.pose_relation_proj=nn.Linear(512,256).to(self.fp_pair_proj.weight.device)
            nn.init.zeros_(self.pose_relation_proj.weight);nn.init.zeros_(self.pose_relation_proj.bias)
        self.pose_relation_fusion=True

    def pose_readout(self, patches, pair, cad_valid, state, valid, ctx, readout_valid):
        """Keep pose-dependent paired evidence inside the object-query readout.

        Completion targets describe the object appearance, not the error of the
        rendered pose. Their latent must not be the only carrier of that error.
        No second pose predictor or temporal path is introduced.
        """
        b=len(patches)
        obj=self.readout.query[:,:1].expand(b,-1,-1)
        relation=(self.pose_relation_proj(pair)*cad_valid.any(-1)[:,None,None]) if self.pose_relation_fusion else None
        object_valid=torch.cat((valid,torch.ones(b,1,device=valid.device,dtype=torch.bool)),1)
        for i in range(4):
            source=patches[:,-1] if self.pose_final_patch_only else patches[:,i]
            if self.pose_relation_fusion:
                source=source+relation
            keys=source if self.pose_patch_only else torch.cat((source,state),1)
            mask=valid if self.pose_patch_only else object_valid
            obj=obj+self.object_attn[i](self.object_norm[i](obj),keys,mask)
        if self.pose_patch_only:
            latent=torch.where(readout_valid[:,None],F.layer_norm(obj[:,0],(256,)),0.)
            return dict(latent=latent,latent_object=latent,latent_context=torch.zeros_like(latent),
                        context_gate=latent.new_zeros(b,1),cross_attention_output=torch.zeros_like(latent),latent_parent=latent)
        return self.single_context_readout(F.layer_norm(obj,(256,)),F.layer_norm(ctx,(256,)),readout_valid)

    def train(self, mode=True):
        super().train(mode); self.fp.eval(); return self

    def enable_compilation(self, enabled=True):
        # Function wrapper, not a registered child: checkpoint names stay stable.
        if enabled and torch.cuda.is_available() and torch.cuda.device_count()!=1:
            raise ValueError('Compiled FP v3 requires one visible GPU per worker; use tools/fp_worker.py')
        self.compiled_frame = torch.compile(self.tensor_frame, fullgraph=True, dynamic=False) if enabled else None

    def single_context_readout(self, obj, ctx, valid):
        """Exact specialization of the original readout with previous=().

        Its attention has one key. Softmax is exactly one; Q/K projections
        and temporal masking cannot affect its result. Keep the original
        parameter layout, including zero gradients in unused Q/K weight rows.
        """
        r=self.readout; w=r.cross_attn.in_proj_weight; bias=r.cross_attn.in_proj_bias
        value=F.linear(torch.where(valid[:,None,None],ctx,0.),w[512:],bias[512:])
        h=r.cross_attn.out_proj(value)
        h=torch.where(valid[:,None,None],h,0.)
        gate=torch.sigmoid(r.cross_gate(torch.cat((obj,h),-1)))
        base=r.parent_latent(obj[:,0],ctx[:,0])
        latent=torch.where(valid[:,None],base+gate[:,0]*h[:,0],0.)
        return dict(latent=latent,latent_object=obj[:,0],latent_context=ctx[:,0],
                    context_gate=gate[:,0],cross_attention_output=h[:,0],latent_parent=base)

    def tensor_frame(self, mid,last,cad_mid,cad_last,cad_valid,static_geo,geo_position,geo_valid,
                     fp_observed,fp_pair,state_input,base,diameter,center,position,prompt,dt,
                     valid,readout_valid,mem,mv,mb,dense_relation=None,cad_relation=None):
        b=len(mid)
        h=self.core.src_proj(torch.cat((mid,last),-1))
        past=(mem[:,:64]*mv[:,:64,None]).sum(1)/mv[:,:64].sum(-1).clamp_min(1)[:,None]
        logits,p,current,cv,cb=self.core.evidence(h,position,prompt,valid,past)
        cad=self.core.src_proj(torch.cat((cad_mid,cad_last),-1))+position+self.cad_type
        observed=self.fp_observation_proj(fp_observed)
        pair=0. if self.fp_observation_only else self.fp_pair_proj(fp_pair)*cad_valid.any(-1)[:,None,None]
        observed_geo=observed+self.geo_position(torch.cat((geo_position[:,:256,:5],torch.zeros_like(geo_position[:,:256,5:])), -1))+self.geo_type.weight[0]
        geo=torch.cat((observed if self.pair_into_patch else observed+pair,self.geo_proj(static_geo)),1)+self.geo_position(geo_position)
        geo=geo+torch.cat((self.geo_type.weight[0].expand(256,-1),self.geo_type.weight[1].expand(128,-1)))[None]
        if self.jepa_pose_geometry:
            geo=torch.cat((geo[:,:256],geo[:,256:]+self.pose_cad_proj(cad_relation)),1)
        state=self.state(state_input)[:,None]
        q=self.core.mask_query+position+prompt[:,None]+dt[:,None]
        if self.pair_into_patch:
            # Same 16x16 crop grid: comparison evidence enters BEFORE all four
            # JEPA blocks, never the object readout. Missing depth stays masked.
            q=q+pair*geo_valid[:,:256,None]
        if self.jepa_pose_geometry:
            q=q+self.pose_dense_proj(dense_relation)+self.pose_state_proj(state_input)[:,None]
        ctx=self.readout.query[:,1:].expand(b,-1,-1)
        patches=[]
        context_source=torch.cat((h+position,observed_geo,mem[:,64:]),1)
        context_valid=torch.cat((valid,geo_valid[:,:256],mv[:,64:]),1)
        object_valid=torch.cat((valid,torch.ones(b,1,device=valid.device,dtype=torch.bool)),1)
        for i,block in enumerate(self.core.blocks):
            q=block(q,current,cv,cb,mem,mv,mb,geo,geo_valid,geo_valid.any(-1).float())
            q=q+self.cad_attn[i](self.cad_norm[i](q),cad,cad_valid)
            patch=self.core.final_norm(q) if i==3 else q
            patches.append(patch)
            ctx=ctx+self.context_attn[i](self.context_norm[i](ctx),context_source,context_valid)
        stacked=torch.stack(patches,1)
        readout=self.pose_readout(stacked,fp_pair,cad_valid,state,valid,ctx,readout_valid)
        if self.capture_pose_inputs:
            self.captured_pose_inputs=tuple(v.detach() if v is not None else None for v in (stacked,fp_pair,cad_valid,state,valid,ctx,readout_valid))
        delta=self.head(readout['latent']).float()
        with torch.autocast(delta.device.type,enabled=False):
            pose=update(base.float(),delta[:,:3],delta[:,3:],diameter.float())
            original=original_pose(pose,center.float())
        surface=self.surface_head(patch).reshape(b,16,16,14,14,5).permute(0,5,1,3,2,4).reshape(b,5,224,224).float()
        extra={}
        if self.jepa_pose_geometry:
            extra=dict(pose_error_prediction=self.pose_error_head(patch).mean(1).float(),pose_base=base.float())
        return dict(**readout,**extra,patch_latent=patch,f_predicted=self.core.feature_last(patch),
            f_mid_predicted=self.core.feature_mid(patch),evidence_logits=logits,
            support_logits=self.core.support(patch).squeeze(-1),
            log_feature_error=self.core.log_error(patch).squeeze(-1).clamp(-14,5),
            pose_centered=pose,pose_original=original,delta_rotvec=delta[:,:3],delta_center_norm=delta[:,3:],
            surface_xyz=surface[:,:3],surface_depth_residual=surface[:,3:4],geometry_valid_logits=surface[:,4:5],
            surface_depth_m=base[:,2,3,None,None,None].float()+surface[:,3:4]*diameter[:,None,None,None].float(),
            writer_h=h,writer_p=p)

    def forward(self,obs,memory=None,history_enabled=None):
        packet=obs.packet;b=len(obs.mid)
        if not getattr(self,'trusted_training_inputs',False): packet.validate()
        memory=self.empty_memory(obs) if memory is None else memory
        if memory.stream!=packet.stream_id or memory.version!=self.weights_version:
            raise ValueError('Stale/cross-stream memory')
        if not getattr(self,'trusted_training_inputs',False) and memory.timestamp is not None and not torch.all(packet.timestamp_s>memory.timestamp):
            raise ValueError('Noncausal memory')
        enabled=torch.ones(b,device=obs.mid.device,dtype=torch.bool) if history_enabled is None else history_enabled
        position,prompt,dt,_=self.core.positions(packet,memory.timestamp)
        valid=F.avg_pool2d(packet.pixel_valid.float(),14,14).flatten(1)>=.999
        features=[];masks=[];bias=[]
        for kind,records in enumerate((memory.objects,memory.contexts)):
            for i in range(4):
                if i<len(records):
                    r=records[i];spatial=torch.cat((r.xyz*r.xyz_valid[...,None],r.xy,r.xyz_valid[...,None].float()),-1)
                    features.append(r.features+self.core.memory_types.weight[kind]+self.memory_position(spatial))
                    age=packet.timestamp_s-r.timestamp
                    masks.append(r.valid&(age>0)[:,None]&enabled[:,None])
                    bias.append(-F.softplus(self.core.memory_age_rate)*age.float().log1p()[:,None].expand(-1,16))
                else:
                    features.append(self.core.memory_types.weight[kind][None,None].expand(b,16,-1)*0)
                    masks.append(valid.new_zeros(b,16))
                    bias.append(self.core.memory_age_rate.float().expand(b,16)*0)
        arguments=(obs.mid,obs.last,obs.cad_mid,obs.cad_last,obs.cad_valid,obs.geo,obs.geo_position,obs.geo_valid,
                   obs.fp_observed,obs.fp_pair,obs.state,obs.base,obs.diameter,obs.center,position,prompt,dt,
                   valid,obs.metadata.key_valid.any(-1),torch.cat(features,1),torch.cat(masks,1),torch.cat(bias,1),
                   obs.dense_relation,obs.cad_relation)
        result=(self.compiled_frame or self.tensor_frame)(*arguments)
        next_memory=self.writer(result.pop('writer_h'),position,result.pop('writer_p'),valid,obs,memory)
        result['read_memory_tokens']=sum(len(rs)*16 for rs in (memory.objects,memory.contexts))
        return result,next_memory

from dataclasses import replace
import torch
from torch import nn
from lip.unified.model import UnifiedTracker
from lip.unified.features import Observation,Scene,build_teachers
from lip.jepa.model import DinoJEPA
from lip.jepa.contracts import FramePacket
from lip.engine.stream_state import FrameMeta
from lip.models.stream_cross_readout import StreamResidualCrossReadout


def test_frozen_attention_preserves_bias_only_gradient():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA efficient SDPA regression')
    from lip.jepa.predictor import SafeAttention
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.manual_seed(71)
    model=SafeAttention().cuda().requires_grad_(False)
    query=torch.randn(2,32,256,device='cuda');source=torch.randn(2,40,256,device='cuda')
    bias=torch.randn(2,40,device='cuda',requires_grad=True)
    results=[]
    for backend in [SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]:
        bias.grad=None
        with sdpa_kernel(backend):output=model(query,source,bias=bias)
        output.square().mean().backward();results.append((output.detach(),bias.grad.clone()))
    assert torch.allclose(results[0][0],results[1][0],atol=1e-5,rtol=1e-4)
    assert torch.allclose(results[0][1],results[1][1],atol=1e-6,rtol=1e-4)
    assert results[0][1].norm()>0 and all(p.grad is None for p in model.parameters())


def make_model():
    core=DinoJEPA(None,memory_enabled=True,geometry_enabled=False)
    parent={'core.'+k:v for k,v in core.state_dict().items() if not k.startswith('writer.')}
    modules={'state':nn.Sequential(nn.Linear(24,256),nn.GELU(),nn.Linear(256,256)),
             'readout':StreamResidualCrossReadout(8),'head':nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,6))}
    for name,module in modules.items():parent.update({'lip.'+name+'.'+k:v for k,v in module.state_dict().items()})
    point=nn.Identity();point.feature_dim=12
    return UnifiedTracker(nn.Identity(),point,parent).eval()


def observation(time=0.,stream='a'):
    packet=FramePacket(torch.randn(1,3,224,224),torch.ones(1,1,224,224,dtype=torch.bool),torch.eye(3)[None],
        torch.tensor([time],dtype=torch.float64),(stream,),torch.tensor([[32.,32.,192.,192.]]),torch.tensor([[640.,480.]]))
    meta=FrameMeta(packet.timestamp_s,torch.tensor([int(time*30)]),torch.tensor([0]),torch.ones(1,17,dtype=torch.bool),torch.zeros(1,17))
    base=torch.eye(4)[None];base[:,2,3]=1.
    return Observation(packet,torch.randn(1,256,384),torch.randn(1,256,384),torch.randn(1,256,384),torch.randn(1,256,384),
        torch.ones(1,256,dtype=torch.bool),torch.randn(1,384,12),torch.randn(1,384,6),torch.ones(1,384,dtype=torch.bool),
        (torch.arange(384)<256)[None],torch.randn(1,256,3),torch.ones(1,256,dtype=torch.bool),torch.randn(1,24),
        base,torch.tensor([.1]),torch.zeros(1,3),meta)


def test_pose_and_feature_loss_reach_same_patch_latent():
    torch.manual_seed(4);m=make_model();out,_=m(observation())
    a=torch.autograd.grad(out['pose_centered'].square().sum(),out['patch_latent'],retain_graph=True)[0]
    b=torch.autograd.grad(out['f_predicted'].square().mean(),out['patch_latent'],retain_graph=True)[0]
    c=torch.autograd.grad(out['surface_xyz'].square().mean()+out['surface_depth_residual'].square().mean()+out['geometry_valid_logits'].square().mean(),out['patch_latent'])[0]
    assert a.abs().sum()>0 and b.abs().sum()>0 and c.abs().sum()>0
    assert out['surface_xyz'].shape==(1,3,224,224) and out['surface_depth_m'].shape==(1,1,224,224)


def test_context_and_written_observation_ignore_cad_appearance():
    torch.manual_seed(4);m=make_model();o=observation()
    with torch.no_grad():
        for module in m.cad_attn:nn.init.normal_(module.out.weight,std=.02)
        left,lmem=m(o);right,rmem=m(replace(o,cad_mid=o.cad_mid+3,cad_last=o.cad_last-2))
    assert torch.equal(left['latent_context'],right['latent_context'])
    assert not torch.equal(left['latent_object'],right['latent_object'])
    assert torch.equal(lmem.contexts[0].features,rmem.contexts[0].features)


def test_history_disabled_has_no_feature_dependency_but_retains_time():
    m=make_model();first=observation();_,memory=m(first);current=observation(1/30)
    changed=replace(memory,objects=tuple(replace(r,features=r.features+10) for r in memory.objects),
                    contexts=tuple(replace(r,features=r.features-10) for r in memory.contexts))
    off=torch.tensor([False])
    with torch.no_grad():a,_=m(current,memory,off);b,_=m(current,changed,off)
    assert torch.equal(a['pose_centered'],b['pose_centered'])


def test_memory_writer_gets_later_loss_gradient_and_local_positions():
    m=make_model();o=observation();_,memory=m(o)
    later,_=m(observation(1/30),memory)
    grad=torch.autograd.grad(later['f_predicted'].square().mean(),m.writer.proj.weight,retain_graph=True)[0]
    geometry_grad=torch.autograd.grad(later['surface_xyz'].square().mean()+later['surface_depth_residual'].square().mean(),m.writer.proj.weight)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    assert torch.isfinite(geometry_grad).all() and geometry_grad.norm()>0
    assert memory.contexts[0].xy.unique(dim=1).shape[1]>1


def test_cross_stream_and_future_memory_rejected():
    import pytest
    m=make_model();o=observation();_,memory=m(o)
    with pytest.raises(ValueError):m(o,memory)
    with pytest.raises(ValueError):m(observation(1/30,'b'),memory)


def test_fullcad_teacher_covers_visible_and_hidden_and_preserves_background():
    class Encoder(nn.Module):
        def forward(self,x):
            f=torch.nn.functional.avg_pool2d(x,14,14).flatten(2).transpose(1,2).repeat(1,1,128)
            return f,f
    rgb=torch.rand(1,3,224,224);mask=torch.zeros(1,224,224);mask[:,40:180,40:110]=1
    silhouette=torch.zeros(224,224,dtype=torch.bool);silhouette[40:180,40:180]=True
    def render(*args):return {'rgb':torch.full((3,224,224),.8),'mask':silhouette,
        'depth':silhouette[None].float()*1.2,'xyz':silhouette[None].float().expand(3,-1,-1)*.02}
    scene=Scene(rgb,torch.ones(1,1,224,224),{},torch.eye(3),torch.eye(3),torch.ones(1,1,224,224,dtype=torch.bool),
        torch.eye(4),.1,torch.zeros(3),torch.zeros(24),0.,'a',(224,224),{'appearance':{}})
    result=build_teachers(Encoder(),[scene],[torch.eye(4)],[mask],[torch.zeros_like(scene.bounds)],render)
    assert torch.equal(result.proxy_rgb[0,:,~silhouette],rgb[0,:,~silhouette])
    assert torch.all(result.proxy_rgb[0,:,silhouette]==.8)
    assert torch.all(result.proxy_rgb[0,:,mask[0].bool()]==.8)
    assert result.proxy_weight.sum()>0 and result.visible_weight.sum()>0
    assert result.proxy_visible_weight.sum()==0 and result.proxy_hidden_weight.sum()>0
    assert torch.allclose(result.surface_xyz[result.geometry_weight.expand_as(result.surface_xyz)],torch.tensor(.2))
    assert torch.all(result.surface_depth_m[result.geometry_weight]==1.2)
    assert not result.geometry_weight[0,0,50:170,50:100].any()
    assert torch.isnan(result.geometry_valid_label[0,0,50:170,50:100]).all()


def test_rgbd_occluder_preserves_outside_and_places_depth_in_front():
    from lip.unified.occlusion import OcclusionPlan
    from lip.unified.features import observation_cloud
    rgb=torch.rand(1,3,224,224);depth=torch.full((1,1,224,224),1.)
    support=torch.zeros(224,224,dtype=torch.bool);support[60:165,60:165]=True
    base=torch.eye(4);base[2,3]=1.
    scene=Scene(rgb,depth,{'mask':support},torch.eye(3),torch.tensor([[150.,0,112],[0,150.,112],[0,0,1.]]),
        torch.ones(1,1,224,224,dtype=torch.bool),base,.15,torch.zeros(3),torch.zeros(24),0.,'a',(224,224),{})
    # Nonconstant donor texture and a missing-depth patch exercise both modalities.
    donor=dict(rgb=torch.rand(3,64,64),alpha=torch.ones(1,64,64,dtype=torch.bool),relative_depth=torch.zeros(1,64,64),
        depth_valid=torch.ones(1,64,64,dtype=torch.bool),kind='hand',object_id=None,source_dir='train/a',frame=1)
    donor['depth_valid'][:,25:35,25:35]=False
    plan=OcclusionPlan((donor,),8,16,.7,((0.,0.,.2,.1,0.,0.,1.),))
    clean=plan.render(scene,7);assert not clean.mask.any() and torch.equal(clean.rgb,rgb)
    altered=plan.render(scene,8);mask=altered.mask
    assert torch.equal(altered.rgb[~mask.expand_as(rgb)],rgb[~mask.expand_as(rgb)])
    assert torch.equal(altered.depth[~mask],depth[~mask])
    assert altered.rgb[mask.expand_as(rgb)].std()>.05
    assert (altered.depth[mask]<depth[mask]).all() and (altered.depth[mask]==0).any()
    cloud,_,_,_,valid,observed=observation_cloud(scene,mask,altered)
    assert torch.equal(observed,altered.depth) and torch.equal(valid,altered.depth[0,0]>0)
    assert torch.isfinite(cloud['coord']).all()
    later=plan.render(scene,9);assert not torch.equal(later.rgb,altered.rgb)
    assert torch.equal(plan.render(scene,24).rgb,rgb)


def test_real_hidden_supervision_is_disjoint_from_cad_and_visible_real_is_diagnostic():
    from lip.unified.losses import reconstruction_loss
    from lip.unified.features import TeacherTargets
    m=make_model();out,_=m(observation())
    features=torch.randn(1,256,384);weights=torch.ones(1,256)
    dense=torch.ones(1,1,224,224,dtype=torch.bool)
    real_mask=weights.clone();real_mask[:,128:]=0;proxy_mask=1-real_mask
    target=TeacherTargets(features,features,features,features,proxy_mask,real_mask,proxy_mask,proxy_mask,weights*0,
        weights,weights,torch.rand(1,3,224,224),torch.rand(1,3,224,224),torch.zeros(1,3,224,224),
        dense.float(),dense.float(),dense,dense,~dense,dense.float(),~dense,dense)
    first,parts=reconstruction_loss(out,target)
    modified=features.clone();modified[:,128:]=torch.randn_like(modified[:,128:])
    changed=replace(target,real_mid=modified,real_last=modified)
    second,_=reconstruction_loss(out,changed)
    assert torch.equal(first,second)
    modified=features.clone();modified[:,:128]=torch.randn_like(modified[:,:128])
    third,_=reconstruction_loss(out,replace(target,real_mid=modified,real_last=modified))
    assert not torch.equal(first,third)
    first.backward();assert m.surface_head[-1].weight.grad.norm()>0
    assert not parts['real_visible_eval'].requires_grad


def test_artificial_occlusion_uses_original_real_depth_and_excludes_missing_depth():
    class Encoder(nn.Module):
        def forward(self,x):
            f=torch.nn.functional.avg_pool2d(x,14,14).flatten(2).transpose(1,2).repeat(1,1,128)
            return f,f
    rgb=torch.rand(1,3,224,224);depth=torch.full((1,1,224,224),.9);depth[:,:,80:100,80:100]=0
    visible=torch.zeros(1,224,224);visible[:,30:195,30:125]=1
    silhouette=torch.zeros(224,224,dtype=torch.bool);silhouette[30:195,30:195]=True
    added=torch.zeros(1,1,224,224,dtype=torch.bool);added[:,:,40:185,40:110]=True
    pose=torch.eye(4);pose[2,3]=1.
    def render(*args):return dict(rgb=torch.full((3,224,224),.8),mask=silhouette,depth=silhouette[None].float()*1.2,xyz=torch.zeros(3,224,224))
    scene=Scene(rgb,depth,{},torch.eye(3),torch.tensor([[150.,0,112],[0,150.,112],[0,0,1.]]),
        torch.ones(1,1,224,224,dtype=torch.bool),pose,.1,torch.zeros(3),torch.zeros(24),0.,'a',(224,224),{'appearance':{}})
    target=build_teachers(Encoder(),[scene],[pose],[visible],[added],render)
    assert target.hidden_real_weight.sum()>0 and target.proxy_weight.sum()>0
    assert not (target.hidden_real_weight.bool()&target.proxy_weight.bool()).any()
    assert not (target.geometry_real_weight&target.geometry_proxy_weight).any()
    assert torch.all(target.surface_depth_m[target.geometry_real_weight]==.9)
    assert torch.all(target.surface_depth_m[target.geometry_proxy_weight]==1.2)
    assert not target.geometry_weight[:,:,80:100,80:100].any()
    assert torch.allclose(target.surface_xyz[:,2:3][target.geometry_real_weight],torch.tensor(-1.))


def test_saturated_depth_and_unrepresentable_sparse_cloud_are_not_fabricated():
    from lip.unified.features import usable_depth
    from lip.unified.utonia import FrozenUtonia
    depth=torch.tensor([0.,float('nan'),float('inf'),-1.,65.535,1.2,8.5])
    torch.testing.assert_close(usable_depth(depth),torch.tensor([0.,0.,0.,0.,0.,1.2,8.5]))
    encoder=FrozenUtonia.__new__(FrozenUtonia);nn.Module.__init__(encoder);encoder.feature_dim=12
    # This is the observed failing 68002-cell extent, without a CUDA dependency.
    bad=dict(coord=torch.tensor([[0.,0.,0.],[0.,680.02,0.]]),color=torch.ones(2,3),normal=torch.ones(2,3))
    empty={k:v[:0] for k,v in bad.items()}
    assert encoder([bad,empty])==[None,None]
    assert encoder.last_unrepresentable_clouds[0]['lane']==0


def test_cache_rebinds_changed_texture_and_rejects_mutated_weights(tmp_path):
    from PIL import Image
    import pytest
    from lip.unified.cad import CADStore
    class Point(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(1),requires_grad=False)
            self.checkpoint_sha256='weight-a';self.source_sha256='source-a'
        def forward(self,clouds):return [c['coord']+c['color']*self.weight for c in clouds]
    obj=tmp_path/'mesh.obj'
    obj.write_text('mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl material\nf 1/1 2/2 3/3\n')
    (tmp_path/'material.mtl').write_text('newmtl material\nmap_Kd texture.png\n')
    Image.new('RGB',(4,4),(255,0,0)).save(tmp_path/'texture.png')
    encoder=Point();store=CADStore(tmp_path/'cache',encoder,'cpu');mesh={'diameter':1.,'center':[0.,0.,0.]}
    before=store.get(obj,mesh)
    Image.new('RGB',(4,4),(0,0,255)).save(tmp_path/'texture.png')
    after=store.get(obj,mesh)
    assert before['key']!=after['key'] and not torch.equal(before['features'],after['features'])
    encoder.checkpoint_sha256='weight-b';changed=store.get(obj,mesh);assert changed['key']!=after['key']
    encoder.weight.add_(1)
    with pytest.raises(ValueError,match='weights changed'):store.get(obj,mesh)


def test_context_excludes_cad_geometry_and_reference_depth_residual():
    m=make_model();o=observation();geo=o.geo.clone();geo[:,256:]+=5
    position=o.geo_position.clone();position[:,:,-1]+=9;position[:,256:,:3]-=3
    with torch.no_grad():a,_=m(o);b,_=m(replace(o,geo=geo,geo_position=position))
    assert torch.equal(a['latent_context'],b['latent_context'])
    assert not torch.equal(a['latent_object'],b['latent_object'])


def test_real_geometry_teacher_is_identical_under_outer_mixed_precision():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA autocast required')
    class Encoder(nn.Module):
        def forward(self,x):
            f=torch.nn.functional.avg_pool2d(x,14,14).flatten(2).transpose(1,2).repeat(1,1,128)
            return f,f
    device='cuda';rgb=torch.rand(1,3,224,224,device=device);depth=torch.full((1,1,224,224),1.123456,device=device)
    mask=torch.ones(1,224,224,device=device);silhouette=torch.ones(224,224,device=device,dtype=torch.bool)
    pose=torch.eye(4,device=device);pose[:3,3]=torch.tensor([.0113,-.037,1.017],device=device)
    def render(*args):return dict(rgb=torch.full((3,224,224),.8,device=device),mask=silhouette,
        depth=silhouette[None].float()*1.2,xyz=torch.zeros(3,224,224,device=device))
    scene=Scene(rgb,depth,{},torch.eye(3,device=device),torch.tensor([[151.234,0,111.32],[0,147.583,113.42],[0,0,1.]],device=device),
        torch.ones(1,1,224,224,device=device,dtype=torch.bool),pose,.12345,torch.zeros(3,device=device),torch.zeros(24,device=device),0.,'a',(224,224),{'appearance':{}})
    reference=build_teachers(Encoder(),[scene],[pose],[mask],[scene.bounds],render)
    with torch.autocast('cuda',dtype=torch.bfloat16):actual=build_teachers(Encoder(),[scene],[pose],[mask],[scene.bounds],render)
    assert torch.equal(reference.surface_xyz,actual.surface_xyz)
    assert torch.equal(reference.surface_depth_m,actual.surface_depth_m)
    assert torch.equal(reference.surface_depth_residual,actual.surface_depth_residual)


def test_unmasked_visible_regions_have_no_completion_gradient():
    from lip.unified.losses import reconstruction_loss
    class Encoder(nn.Module):
        def forward(self,x):
            f=torch.nn.functional.avg_pool2d(x,14,14).flatten(2).transpose(1,2).repeat(1,1,128)
            return f,f
    rgb=torch.rand(1,3,224,224);visible=torch.zeros(1,224,224);visible[:,28:196,28:112]=1
    silhouette=torch.zeros(224,224,dtype=torch.bool);silhouette[28:196,28:196]=True
    added=torch.zeros(1,1,224,224,dtype=torch.bool);added[:,:,28:112,28:112]=True
    pose=torch.eye(4);pose[2,3]=1.
    def render(*args):return dict(rgb=torch.full((3,224,224),.8),mask=silhouette,depth=silhouette[None].float()*1.2,xyz=torch.zeros(3,224,224))
    scene=Scene(rgb,torch.ones(1,1,224,224),{},torch.eye(3),torch.eye(3),torch.ones(1,1,224,224,dtype=torch.bool),
        pose,.1,torch.zeros(3),torch.zeros(24),0.,'a',(224,224),{'appearance':{}})
    target=build_teachers(Encoder(),[scene],[pose],[visible],[added],render)
    m=make_model();out,_=m(observation());loss,_=reconstruction_loss(out,target)
    keys=('f_predicted','surface_xyz','surface_depth_residual','geometry_valid_logits')
    grads=torch.autograd.grad(loss,[out[k] for k in keys])
    unmasked_visible=visible[None].bool()&~added
    assert target.hidden_real_weight.sum()>0 and target.proxy_weight.sum()>0
    assert not (target.geometry_weight&unmasked_visible).any()
    for gradient in grads[1:]:assert gradient[unmasked_visible.expand_as(gradient)].count_nonzero()==0
    no_feature_target=(target.hidden_real_weight+target.proxy_weight)==0
    assert grads[0][no_feature_target].count_nonzero()==0

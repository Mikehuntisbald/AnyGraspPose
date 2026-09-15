import math
import pytest
import torch
from lip.losses_spatial_alignment import correspondence_targets,spatial_alignment_loss
from lip.models.rotation_alignment import RotationAlignment


def plane_fixture():
    yy,xx=torch.meshgrid(torch.arange(224),torch.arange(224),indexing='ij')
    geom=torch.zeros(1,9,224,224);geom[:,3]=1
    geom[:,4]=(xx-111.5)/200;geom[:,5]=(yy-111.5)/200
    d=torch.ones(1);target=torch.eye(4)[None];target[:,2,3]=1
    k=torch.tensor([[[200.,0.,111.5],[0.,200.,111.5],[0.,0.,1.]]])
    a=torch.eye(3)[None];depth=torch.ones(1,1,224,224)
    return geom,d,target,k,a,depth,depth.clone(),torch.ones(1,dtype=torch.bool)


def test_identity_geometry_correspondence_and_translation_direction():
    values=list(plane_fixture());dist,valid,count=correspondence_targets(*values)
    assert valid.all() and count['visible_keys']==196
    assert torch.equal(dist.argmax(-1),torch.arange(196)[None])
    torch.testing.assert_close(dist.sum(-1),torch.ones(1,196))
    values[2]=values[2].clone();values[2][:,0,3]=16/200
    shifted,valid,_=correspondence_targets(*values)
    # Camera +x moves each CAD point to a query one cell to its right.
    assert shifted[0,8*14+8].argmax()==8*14+7


def test_camera_rotation_and_nonidentity_crop_depth_lookup():
    values=list(plane_fixture())
    values[2][:,:3,:3]=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    dist,_,_=correspondence_targets(*values)
    assert dist[0,8*14+9].argmax()==4*14+8
    values=list(plane_fixture());values[4]=torch.tensor([[[.5,0.,10.],[0.,.5,10.],[0.,0.,1.]]])
    values[5]=torch.ones(1,1,448,448)
    # Query/CAD (8,8) projects near crop (136,136), raw (252,252).
    values[5][:,:,240:270,240:270]=0
    dist,_,_=correspondence_targets(*values)
    assert not dist[:,:,8*14+8].count_nonzero()
    assert dist[:,:,5*14+5].sum()>0


def test_checkpoint_rejects_unrecorded_latent_route_change(tmp_path):
    from lip.engine.stream_checkpoint import load_init
    path=tmp_path/'state.pt'
    torch.save(dict(config=dict(rotation_alignment=True,alignment_use_parent_latent=True)),path)
    with pytest.raises(ValueError,match='latent route'):
        load_init(path,object(),{},dict(rotation_alignment=True,alignment_use_parent_latent=False))


@pytest.mark.parametrize('change',['observed_occlusion','self_occlusion','invalid_depth','symmetric','behind','outside'])
def test_correspondence_rejects_invalid_targets(change):
    x=list(plane_fixture())
    if change=='observed_occlusion':x[5]*=.5
    if change=='self_occlusion':x[6]*=.5
    if change=='invalid_depth':x[5].zero_()
    if change=='symmetric':x[7].fill_(False)
    if change=='behind':x[2][:,2,3]=-1
    if change=='outside':x[2][:,0,3]=100
    dist,valid,_=correspondence_targets(*x)
    assert not valid.any() and torch.isfinite(dist).all() and not dist.count_nonzero()
    logits=torch.randn(1,196,196,requires_grad=True)
    loss,_=spatial_alignment_loss(logits.log_softmax(-1),dist,valid)
    loss.backward();assert loss==0 and not logits.grad.count_nonzero()


def test_actual_attention_probabilities_and_aux_gradient_at_zero_rotation_start():
    torch.set_num_threads(2);torch.manual_seed(4);m=RotationAlignment()
    dense=torch.randn(1,196,256,requires_grad=True);geom=torch.randn(1,9,224,224);latent=torch.randn(1,256)
    correction,(q,k)=m(dense,geom,latent,return_matching=True)
    assert not correction.count_nonzero()
    probability=m.correspondence_log_probability(q,k).exp()
    _,actual=m.match.attn(q,k,k,need_weights=True)
    torch.testing.assert_close(probability,actual,atol=1e-7,rtol=1e-5)
    target=torch.eye(196)[None];supported=torch.ones(1,196,dtype=torch.bool)
    loss,_=spatial_alignment_loss(probability.log(),target,supported);loss.backward()
    assert m.cad[0].weight.grad.norm()>0 and m.match.attn.in_proj_weight.grad.norm()>0
    assert dense.grad.norm()>0


def test_removed_latent_is_invariant_after_nonzero_head():
    torch.set_num_threads(2);torch.manual_seed(5);m=RotationAlignment(use_parent_latent=False).eval()
    with torch.no_grad():m.output[-1].weight.normal_(std=.01)
    dense=torch.randn(2,196,256);geom=torch.randn(2,9,224,224)
    a=m(dense,geom,torch.randn(2,256));b=m(dense,geom,torch.randn(2,256)*100)
    assert torch.equal(a,b)
    m.use_parent_latent=True
    assert not torch.equal(m(dense,geom,torch.zeros(2,256)),m(dense,geom,torch.ones(2,256)))

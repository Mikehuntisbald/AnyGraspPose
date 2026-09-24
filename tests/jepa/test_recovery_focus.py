from types import SimpleNamespace
from dataclasses import replace
import torch
from lip.unified.recovery_focus import spatial_terms, camera_disagreement, focus_loss
from lip.unified.recovery_metrics import scramble_history
from lip.unified.model import Record, Memory


def test_spatial_object_mean_and_wrong_locations_cannot_match_perfect_recovery():
    torch.manual_seed(5)
    target = torch.randn(2, 12, 24)
    weight = torch.ones(2, 12)
    exact = spatial_terms(target, target, weight, weight)
    mean = spatial_terms(target.mean(1, keepdim=True).expand_as(target), target, weight, weight)
    wrong = spatial_terms(target.roll(3, 1), target, weight, weight)
    assert all(abs(float(x)) < 1e-6 for x in exact)
    assert all(x > .1 for x in mean) and all(x > .1 for x in wrong)


def test_spatial_loss_supervises_only_missing_queries_and_detaches_teacher():
    torch.manual_seed(6)
    p = torch.randn(2, 12, 24, requires_grad=True)
    t = torch.randn_like(p, requires_grad=True)
    query = torch.zeros(2, 12); query[:, 2:7] = 1
    sum(spatial_terms(p, t, query, torch.ones_like(query))).backward()
    assert p.grad[:, 2:7].abs().sum() > 0
    assert not p.grad[:, :2].any() and not p.grad[:, 7:].any()
    assert t.grad is None


def test_empty_singleton_and_ambiguous_teacher_are_finite_without_fake_negatives():
    p = torch.randn(2, 12, 24, requires_grad=True)
    for query in (torch.zeros(2, 12), torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 12)):
        terms = spatial_terms(p, p.detach(), query, query)
        assert all(torch.isfinite(x) and x == 0 for x in terms)
    t = torch.randn(2, 1, 24).expand_as(p)
    assert abs(float(spatial_terms(t, t, torch.ones(2, 12), torch.ones(2, 12))[1])) < 1e-6


def camera_example():
    rays = torch.tensor([[[[-.2, .2], [-.2, .2]], [[-.1, -.1], [.1, .1]], [[1., 1.], [1., 1.]]]])
    rot = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    translation = torch.tensor([[.3, -.2, 4.]])
    depth = torch.ones(1, 1, 2, 2)*4.1
    xyz = torch.einsum('bji,bjhw->bihw', rot, rays*depth-translation[:, :, None, None])
    target = SimpleNamespace(camera_rotation=rot, camera_translation_d=translation,
                             camera_rays=rays, base_depth_d=torch.tensor([3.9]))
    out = dict(surface_xyz=xyz.clone().requires_grad_(), surface_depth_residual=(depth-3.9).clone().requires_grad_())
    return out, target


def test_full_camera_constraint_uses_xy_and_both_heads_not_just_depth():
    output, target = camera_example()
    assert camera_disagreement(output, target).abs().max() < 1e-6
    with torch.no_grad(): output['surface_xyz'][:, 0] += .1
    residual = camera_disagreement(output, target)
    assert residual[:, :2].abs().sum() > 0 and residual[:, 2].abs().max() < 1e-6
    residual.square().mean().backward()
    assert output['surface_xyz'].grad.abs().sum() > 0
    # Perturb the independent depth as well: both recovered outputs are trainable.
    output['surface_xyz'].grad = None; output['surface_depth_residual'].grad = None
    with torch.no_grad(): output['surface_depth_residual'].add_(.1)
    camera_disagreement(output, target).square().mean().backward()
    assert output['surface_xyz'].grad.abs().sum() > 0 and output['surface_depth_residual'].grad.abs().sum() > 0


def test_camera_loss_keeps_float32_geometry_inside_autocast():
    output,target=camera_example()
    with torch.autocast('cpu',dtype=torch.bfloat16):error=camera_disagreement(output,target)
    assert error.dtype==torch.float32 and error.abs().max()<1e-6


def test_camera_constraint_has_no_confidence_escape_or_visible_pixel_gradient():
    output, target = camera_example()
    torch.manual_seed(7)
    f = torch.randn(1, 4, 24)
    output.update(f_predicted=f.clone().requires_grad_(), f_mid_predicted=f.clone().requires_grad_(),
                  geometry_valid_logits=torch.full((1, 1, 2, 2), -100., requires_grad=True))
    mask = torch.zeros(1, 1, 2, 2); mask[:, :, 0, 0] = 1
    target.__dict__.update(real_last=f, real_mid=f, proxy_last=f, proxy_mid=f,
                          hidden_real_weight=torch.zeros(1, 4), visible_weight=torch.ones(1, 4),
                          proxy_weight=torch.zeros(1, 4), support_label=torch.ones(1, 4),
                          geometry_real_weight=mask, geometry_proxy_weight=mask*0)
    with torch.no_grad(): output['surface_xyz'].add_(.1)
    value, _ = focus_loss(output, target, dict(cad_feature=.5, camera_consistency=.25))
    value.backward()
    assert value > 0 and output['geometry_valid_logits'].grad is None
    assert not output['surface_xyz'].grad[:, :, 1].any()
    assert not output['surface_xyz'].grad[:, :, 0, 1].any()


def test_scrambled_history_changes_associations_without_changing_support_or_source():
    values=torch.arange(48).reshape(1,16,3).float()
    valid=torch.zeros(1,16,dtype=torch.bool);valid[:,[1,4,7]]=True
    record=Record(values,valid,values[:,:,:2].clone(),values.clone(),valid.clone(),torch.tensor([0.]),torch.tensor([.8]))
    memory=Memory(('a',),'v1',(record,),(record,),torch.tensor([0.]))
    changed=scramble_history(memory)
    assert torch.equal(record.features,values) and changed.stream==memory.stream
    for result in (changed.objects[0],changed.contexts[0]):
        assert not torch.equal(result.features,record.features)
        assert torch.equal(result.features[:,~valid[0]],record.features[:,~valid[0]])
        assert torch.equal(result.features[:,valid[0]].sort(1).values,record.features[:,valid[0]].sort(1).values)
        for name in ('xy','xyz','valid','xyz_valid','timestamp','quality'):
            assert torch.equal(getattr(result,name),getattr(record,name))


def test_spatial_and_camera_losses_reach_the_same_jepa_patch():
    from test_recovered_relation import sample
    from lip.unified.reconstruction_only import configure_reconstruction_only
    _,model,obs=sample();configure_reconstruction_only(model)
    output,_=model(obs)
    target=torch.randn_like(output['f_predicted']);q=torch.zeros(1,256);q[:,32:96]=1
    spatial=sum(spatial_terms(output['f_predicted'],target,q,torch.ones_like(q)))
    gs=torch.autograd.grad(spatial,output['patch_latent'],retain_graph=True)[0]
    rays=torch.zeros(1,3,224,224);rays[:,2]=1
    teacher=SimpleNamespace(camera_rotation=torch.eye(3)[None],camera_translation_d=torch.tensor([[0.,0.,4.]]),
                            camera_rays=rays,base_depth_d=torch.tensor([4.]))
    camera=camera_disagreement(output,teacher).square().mean()
    gc=torch.autograd.grad(camera,output['patch_latent'])[0]
    assert gs.isfinite().all() and gc.isfinite().all() and gs.abs().sum()>0 and gc.abs().sum()>0


def test_history_overlap_requires_past_visible_depth_not_only_projection():
    from lip.unified.history_training import observed_history_support
    from lip.unified.features import camera_points
    pose=torch.eye(4);pose[2,3]=1.
    k=torch.tensor([[400.,0.,112.],[0.,400.,112.],[0.,0.,1.]])
    depth=torch.ones(1,1,224,224)
    scene=SimpleNamespace(diameter=.2,k_crop=k,affine=torch.eye(3),depth=depth,
                          bounds=torch.ones_like(depth,dtype=torch.bool))
    xyz=((camera_points(depth,k)-pose[:3,3])/.2).permute(2,0,1)[None]
    target=SimpleNamespace(surface_xyz=xyz)
    visible=torch.ones(1,1,224,224)
    assert observed_history_support([scene],pose[None],visible,target).all()
    visible[:,:,:,112:]=0
    known=observed_history_support([scene],pose[None],visible,target)
    assert known[:,:,:,:112].all() and not known[:,:,:,112:].any()
    # A foreground occluder at the old pixel is not object-surface support.
    scene.depth=depth*.8
    assert not observed_history_support([scene],pose[None],visible,target).any()

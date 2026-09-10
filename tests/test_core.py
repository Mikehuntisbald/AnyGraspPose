import inspect
import numpy as np
import pytest
import torch
from lip.geometry.so3 import exp,log,angle,update,center_pose,original_pose
from lip.geometry.crop import project,crop_matrix,crop_images,geometry_channels
from lip.geometry.renderer import Renderer
from lip.data.index import target,split_of
from lip.data.perturb import noisy_history
from lip.models.tracker import Tracker
from lip.engine.features import build_features,stack_features
from lip.engine.runtime import batch_step
from lip.losses import pose_loss
from lip.integrations.foundationpose import FoundationPoseAdapter
from lip.evaluation.metrics import constant_velocity,errors,summarize

torch.set_num_threads(2)


def mesh():
    import trimesh
    m=trimesh.creation.box(extents=[.1,.08,.06])
    return dict(vertices=np.array(m.vertices,dtype='f4'),faces=np.array(m.faces,dtype='i4'),center=np.array([.03,-.02,.01],dtype='f4'),
                diameter=np.float32(np.linalg.norm([.1,.08,.06])),points=np.tile(np.asarray(m.vertices,dtype='f4'),(64,1)))


def base_pose():
    t=torch.eye(4);t[2,3]=.6;return t


def test_target_mapping_no_hand():
    meta=dict(ycb_ids=[5,19,2],ycb_grasp_ind=1)
    a=base_pose()[:3].numpy();b=a.copy();b[0,3]=.11
    oid,j,t=target(meta,dict(pose_y=np.stack([a,b,a])))
    assert (oid,j)==(19,1) and t[0,3]==np.float32(.11)
    with pytest.raises(ValueError):target(meta,dict(pose_y=np.zeros((3,3,4))))


def test_center_roundtrip_projection():
    t=base_pose().double();t[:3,:3]=exp(torch.tensor([.1,.2,-.4],dtype=torch.double));c=torch.tensor([.1,-.2,.3],dtype=torch.double)
    tc=center_pose(t,c);assert torch.allclose(original_pose(tc,c),t,atol=1e-12)
    v=torch.rand(20,3,dtype=torch.double);k=torch.tensor([[600.,0,320],[0,600,240],[0,0,1]],dtype=torch.double)
    assert torch.allclose(project(v@t[:3,:3].T+t[:3,3],k),project((v-c)@tc[:3,:3].T+tc[:3,3],k),atol=1e-10)


def test_crop_projection_padding_pixel_centers():
    m=mesh();t=base_pose();t[0,3]=-.25;k=torch.tensor([[400.,0,30],[0,410,25],[0,0,1]])
    a,kc=crop_matrix(torch.tensor(m['vertices']),t,k)
    cam=torch.tensor(m['vertices'])@t[:3,:3].T+t[:3,3];uv=project(cam,k)
    mapped=torch.cat((uv,torch.ones(len(uv),1)),-1)@a.T
    assert torch.allclose(project(cam,kc),mapped[:,:2],atol=3e-5)
    x=torch.arange(64*64).float().reshape(1,1,64,64)
    assert torch.allclose(crop_images(x,torch.eye(3),64,'nearest'),x)


@pytest.mark.parametrize('v',[[0.,0,0],[1e-9,-1e-8,1e-7],[3.141592-1e-5,0,0],[0,2.221441,2.221441]])
def test_so3_stable_gradients(v):
    v=torch.tensor(v,dtype=torch.double,requires_grad=True);r=exp(v);w=log(r)
    assert torch.allclose(r.T@r,torch.eye(3,dtype=torch.double),atol=1e-10)
    assert abs(float(torch.det(r).detach())-1)<1e-10
    assert torch.allclose(exp(w),r,atol=2e-6)
    (w.square().sum()+angle(r)).backward();assert torch.isfinite(v.grad).all()


def test_decoupled_update():
    t=base_pose();t[0,3]=.2
    r=update(t,torch.tensor([0.,0,1.]),torch.zeros(3),torch.tensor(.2))
    assert torch.equal(r[:3,3],t[:3,3])
    r=update(t,torch.zeros(3),torch.tensor([1.,0,0]),torch.tensor(.2))
    assert torch.equal(r[:3,:3],t[:3,:3]) and r[0,3]==.4


def test_cpu_renderer_geometry_validity():
    m=mesh();t=base_pose();k=torch.tensor([[150.,0,31.5],[0,150,31.5],[0,0,1.]])
    rd,xyz=Renderer('cpu')(m,t,k,64)
    assert abs(float(rd[0,32,32])-.57)<1e-5
    assert xyz[1,40,32]>xyz[1,25,32]
    for depth in [torch.zeros(2,1,64,64),rd[None].expand(2,-1,-1,-1)]:
        g=geometry_channels(depth,rd[None],xyz[None],.15,.6)
        assert g.shape==(2,9,64,64) and torch.isfinite(g).all()
    depth=torch.zeros(2,1,64,64);depth[0,0,1,1]=float('nan')
    assert torch.isfinite(geometry_channels(depth,rd[None],xyz[None],.15,.6)).all()


def test_split_disjoint_physical_sequences():
    buckets={k:set() for k in ['train','val','test']}
    for s in range(10):
        for q in range(100):buckets[split_of(s,q)].add((s,q))
    assert [len(buckets[k]) for k in buckets]==[800,40,160]
    assert not buckets['train']&buckets['val'] and not buckets['train']&buckets['test']
    assert split_of(6,0,'s1')=='val' and split_of(9,99,'s1')=='train'


def test_available_subjects_keep_global_split_positions(tmp_path):
    from lip.data.index import subject_listings,SUBJECTS
    selected=[SUBJECTS[1],SUBJECTS[2],SUBJECTS[9]]
    for subject in selected:
        for q in range(100):
            d=tmp_path/subject/f'20200101_{q:06d}';d.mkdir(parents=True);(d/'meta.yml').touch()
    listings,missing=subject_listings(tmp_path,selected)
    assert not missing
    counts=dict(train=0,val=0,test=0)
    for si,subject in enumerate(SUBJECTS):
        for qi,_ in enumerate(listings[subject]):counts[split_of(si,qi)]+=1
    assert counts==dict(train=240,val=20,test=40)
    assert not listings[SUBJECTS[0]]
    (tmp_path/selected[0]/'20200101_000099'/'meta.yml').unlink()
    _,missing=subject_listings(tmp_path,selected)
    assert missing==[dict(subject=selected[0],found=99)]


def test_quick_evaluation_selection_is_frozen_and_object_balanced():
    from lip.evaluation.selection import fixed_balanced_subset
    rows=[dict(object_id=o,frame=f) for o in range(20) for f in range(o+1)]
    chosen=fixed_balanced_subset(rows,32,lambda x:x['object_id'],42)
    assert chosen==fixed_balanced_subset(rows,32,lambda x:x['object_id'],42)
    assert len(chosen)==32 and len({r['object_id'] for r in chosen})==20
    assert len({(r['object_id'],r['frame']) for r in chosen})==32


def inputs(length=3,size=32):
    m=mesh();base=base_pose();k=torch.tensor([[100.,0,31.5],[0,100.,31.5],[0,0,1.]])
    renderer=Renderer('cpu');depth,_=renderer(m,base,k,64)
    hist=base[None].repeat(length,1,1);valid=torch.ones(length,dtype=torch.bool);pv=valid.clone();pv[-1]=False
    f,_=build_features(torch.rand(length,3,64,64),depth[None].repeat(length,1,1,1),hist,torch.arange(length)/30,
                       valid,pv,base,k,m,renderer,size)
    return stack_features([f]),m


def test_cpu_forward_backward_no_hand_and_padding():
    f,m=inputs();model=Tracker(False,dropout=0.);model.train()
    assert all(not module.training for module in model.modules() if isinstance(module,torch.nn.BatchNorm2d))
    f['frame_valid'][0,:2]=False;f['geometry'].zero_();f['history_pose_valid'].zero_()
    out=model(**f);assert torch.isfinite(out['latent']).all();assert out['latent'].shape==(1,256)
    gt=f['T_base_centered'].clone();gt[:,0,3]+=.02;gt[:,:3,:3]=exp(torch.tensor([[.1,.05,0.]]))
    loss,_=pose_loss(out['pose_centered'],gt,torch.tensor(m['points'])[None],f['object_diameter_m'])
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.head[-1].weight.grad.abs().sum()>0
    assert not any('mano' in k or 'pose_m' in k for k in inspect.signature(model.forward).parameters)


def test_temporal_causality_with_nonzero_head():
    f,_=inputs();model=Tracker(False,dropout=0.).eval();torch.nn.init.normal_(model.head[-1].weight,std=.01)
    captured=[];hook=model.temporal.register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
    with torch.no_grad():
        model(**f);f['rgb'][:,2]=torch.randn_like(f['rgb'][:,2]);f['geometry'][:,2]=torch.randn_like(f['geometry'][:,2]);model(**f)
    hook.remove()
    assert torch.allclose(captured[0][:,:34],captured[1][:,:34],atol=2e-5)
    assert not torch.allclose(captured[0][:,-1],captured[1][:,-1])


def test_perturbation_bounded_and_target_relative():
    poses=base_pose()[None].repeat(100,1,1);noisy,stats=noisy_history(poses,.15,torch.Generator().manual_seed(4))
    assert (angle(noisy[:,:3,:3])<=torch.pi/4+1e-5).all()
    assert ((noisy[:,:3,3]-poses[:,:3,3]).norm(dim=-1)<=.25*.15+1e-5).all()
    delta=log(poses[:,:3,:3]@noisy[:,:3,:3].transpose(-1,-2));dt=(poses[:,:3,3]-noisy[:,:3,3])/.15
    assert torch.allclose(update(noisy,delta,dt,torch.full((100,),.15)),poses,atol=1e-6)


def test_fp_nonzero_center_rejection():
    class FP:
        pose_last=torch.eye(4)
        def get_tf_to_centered_mesh(self):
            c=torch.eye(4);c[:3,3]=torch.tensor([-.1,.2,-.3]);return c
        def track_one(self,rgb,depth,K,iteration):
            self.pose_last=self.pose_last.clone();self.pose_last[0,3]+=.2
            return self.pose_last@self.get_tf_to_centered_mesh()
    fp=FP();adapter=FoundationPoseAdapter(fp,'cpu');prior=base_pose()
    adapter.accept(prior);assert torch.allclose(fp.pose_last@fp.get_tf_to_centered_mesh(),prior)
    result=adapter.refine(prior,None,None,None,accept_result=False)
    assert torch.equal(result,prior) and torch.allclose(fp.pose_last@fp.get_tf_to_centered_mesh(),prior)


def test_constant_velocity_real_dt_and_identity_loss():
    a=base_pose();b=a.clone();b[0,3]=.1
    cv=constant_velocity(torch.stack([a,b]),torch.tensor([0.,.2]),torch.tensor(.6))
    assert abs(float(cv[0,3])-.3)<1e-6
    e=errors(a,a,mesh()['vertices'],.15);assert e['add_m']==0 and e['rotation_deg']==0


def test_current_future_gt_cannot_enter_forward_and_rollout_owns_state():
    from lip.data.synthetic import synthetic_item
    item=synthetic_item(length=2,size=64);model=Tracker(False,dropout=0.).eval()
    torch.nn.init.normal_(model.head[-1].weight,std=.002)
    c=dict(clip_length=2,image_size=32,precision='fp32',crop_expansion=2.)
    captured=[]
    hook=model.register_forward_pre_hook(lambda m,a,k:captured.append({n:v.detach().clone() for n,v in k.items()}),with_kwargs=True)
    with torch.no_grad():
        _,out1=batch_step(model,[item],Renderer('cpu'),c,rollout=4,noise=False,backward=False)
        item['poses'][2:,:3,3]+=.2
        _,out2=batch_step(model,[item],Renderer('cpu'),c,rollout=4,noise=False,backward=False)
    hook.remove()
    for i in range(4):
        for name in captured[i]:assert torch.equal(captured[i][name],captured[4+i][name]),name
        assert torch.equal(out1[i],out2[i])
        if i:assert torch.equal(captured[i]['T_base_centered'],out1[i-1])


def test_checkpoint_resume_scheduler_and_rng(tmp_path):
    from lip.engine.checkpoint import save,resume
    from lip.engine.config import optimizer_and_scheduler
    c=dict(lr_rgb_backbone=1e-5,lr_new_modules=1e-4,weight_decay=.05,warmup_optimizer_steps=2,max_optimizer_steps=10,min_lr_ratio=.1)
    model=torch.nn.Linear(3,3);opt,sched=optimizer_and_scheduler(model,c);audit=dict(mesh_hash='m',split_hash='s')
    for _ in range(3):
        opt.zero_grad();model(torch.ones(1,3)).sum().backward();opt.step();sched.step()
    save(tmp_path/'last.pt',model,opt,sched,3,c,audit,6)
    expected=torch.rand(4);expectedlr=[g['lr'] for g in opt.param_groups]
    clone=torch.nn.Linear(3,3);o,s=optimizer_and_scheduler(clone,c);ck=resume(tmp_path/'last.pt',clone,o,s,audit)
    assert ck['global_step']==3 and ck['sampler_position']==6 and s.last_epoch==3
    assert [g['lr'] for g in o.param_groups]==expectedlr and torch.equal(torch.rand(4),expected)
    for _ in range(2):
        for m,optimizer,scheduler in [(model,opt,sched),(clone,o,s)]:
            optimizer.zero_grad();m(torch.ones(1,3)).sum().backward();optimizer.step();scheduler.step()
    for a,b in zip(model.parameters(),clone.parameters()):assert torch.allclose(a,b,atol=1e-7)


def test_best_selection_survives_resume(tmp_path):
    from lip.engine.checkpoint import record_best
    last=tmp_path/'last.pt';best=tmp_path/'best.pt'
    torch.save(dict(global_step=5000,best=-1.,model={'weight':torch.tensor([1.,2.])}),last)
    record_best(last,best,.6,'validation/metrics.json')
    for path in (last,best):
        ck=torch.load(path,weights_only=False)
        assert ck['global_step']==5000 and ck['best']==.6
        assert torch.equal(ck['model']['weight'],torch.tensor([1.,2.]))


def test_incomplete_inventory_fails_closed(tmp_path):
    from lip.data.index import build_index
    raw=tmp_path/'raw';raw.mkdir()
    with pytest.raises(RuntimeError,match='complete sequence inventory'):build_index(raw,tmp_path/'cache')
    assert not (tmp_path/'cache'/'streams.jsonl').exists()


def test_official_calibration_tuple_without_unsafe_constructor(tmp_path):
    from lip.data.index import read_yaml
    p=tmp_path/'intr.yml';p.write_text('color: {fx: 600, fy: 600, ppx: 320, ppy: 240}\nextrinsics: !!python/tuple [1, 2, 3]\n')
    assert read_yaml(p)['color']['ppx']==320
    p.write_text('bad: !!python/object/apply:os.system [false]')
    import yaml
    with pytest.raises(yaml.constructor.ConstructorError):read_yaml(p)


def test_closed_loop_never_resets_and_keeps_lost_rows(tmp_path,monkeypatch):
    import cv2,json
    import lip.evaluate as evaluation
    from lip.data.synthetic import synthetic_item
    item=synthetic_item(length=4)
    raw=tmp_path/'raw';index=tmp_path/'index';index.mkdir()
    stream=dict(stream_id='subject/sequence/camera',relative_dir='subject/sequence/camera',object_id=1,
                split='train',intrinsics=item['k'].tolist(),mesh_cache='mesh.npz',pose_cache='poses.npz')
    path=raw/stream['relative_dir'];path.mkdir(parents=True)
    for i,(r,d) in enumerate(zip(item['rgb'],item['depth'])):
        cv2.imwrite(str(path/f'color_{i:06d}.jpg'),(r.permute(1,2,0).numpy()[...,::-1]*255).astype('uint8'))
        cv2.imwrite(str(path/f'aligned_depth_to_color_{i:06d}.png'),(d[0].numpy()*1000).astype('uint16'))
        np.savez(path/f'labels_{i:06d}.npz',seg=(d[0].numpy()>0).astype('uint8'))
    np.savez(index/'mesh.npz',**item['mesh'])
    orig=original_pose(item['poses'][1:],torch.tensor(item['mesh']['center'])).numpy()
    np.savez(index/'poses.npz',frames=np.arange(len(orig)),poses=orig)
    (index/'streams.jsonl').write_text(json.dumps(stream)+'\n')
    audit=dict(depth_scale_to_m=.001,fps=30.,split_hash='unit_fixture',mesh_hash='unit_fixture',complete=False)
    (index/'audit.json').write_text(json.dumps(audit))
    monkeypatch.setattr(evaluation,'check_data_gate',lambda *a,**kw:audit)
    c=dict(seed=42,clip_length=3,image_size=32,crop_expansion=2.,precision='fp32',moving_center_per_sec=.2,moving_rotation_rad_per_sec=.3)
    m=Tracker(False,dropout=0.);torch.nn.init.normal_(m.head[-1].weight,std=.001)
    evaluation.evaluate(m,c,raw,index,tmp_path/'before',split='train',allow_verified_subset=True)
    orig[1:,0,3]+=1.
    np.savez(index/'poses.npz',frames=np.arange(len(orig)),poses=orig)
    evaluation.evaluate(m,c,raw,index,tmp_path/'after',split='train',allow_verified_subset=True)
    before=[json.loads(x) for x in (tmp_path/'before/predictions.jsonl').read_text().splitlines()]
    after=[json.loads(x) for x in (tmp_path/'after/predictions.jsonl').read_text().splitlines()]
    assert len(before)==len(after)==7
    assert [r['pose_centered'] for r in before]==[r['pose_centered'] for r in after]
    assert after[-1]['lost'] and after[-1]['adds_m']>.1


@pytest.mark.cuda
def test_cuda_cpu_renderer_agreement():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('nvdiffrast.torch')
    m=mesh();k=torch.tensor([[150.,0,31.5],[0,150,31.5],[0,0,1.]])
    cpu=Renderer('cpu')(m,base_pose(),k,64);gpu=Renderer('cuda')(m,base_pose().cuda(),k.cuda(),64)
    mask=(cpu[0]>0)&(gpu[0].cpu()>0)
    assert mask.sum()>300
    assert (cpu[0][mask]-gpu[0].cpu()[mask]).abs().max()<1e-5
    assert (cpu[1][:,mask[0]]-gpu[1].cpu()[:,mask[0]]).abs().max()<1e-5


@pytest.mark.foundationpose
def test_real_fp_requires_dependencies():
    import os
    if not os.getenv('FOUNDATIONPOSE_DIR'):pytest.skip('FOUNDATIONPOSE_DIR and weights not supplied; adapter unit tests are separate')
    pytest.skip('Real FP requires an explicit RGB-D/mesh fixture and loaded estimator; not a fake inference test')

"""Real zero-start parity and same-parameter spatial-supervision gradients."""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_config import make_model,load_stream_config,optimizer_and_scheduler
from lip.engine.stream_checkpoint import sha,source_hash,load_init
from lip.engine.config import check_data_gate
from lip.engine.stream_training import StreamTrainingModule
from lip.geometry.renderer import Renderer
from lip.data.stream_clips import StreamClips


def main():
    root=Path(__file__).resolve().parents[1];r=root/'runs/factorial';e=json.loads((r/'experiment.json').read_text());torch.set_num_threads(2)
    assert source_hash()==e['source_sha256'];assert sha(e['models_info'])==e['models_info_sha256']
    audit=check_data_gate(e['index_root'])
    cohort=json.loads(Path('/mnt/why/dexycb_lip/alignment_diagnosis_20260915/runs/train_diagnostic/spec.json').read_text())
    data=StreamClips(e['data_root'],e['index_root'],8,48,fixed=json.loads(Path(e['training_manifest']).read_text()),decode_threads=2,
        external_initializers=e['train_initializers'],external_initializers_sha256=e['train_initializers_sha256'],real_initialization_probability=.5,include_initial_observation=True)
    chosen=cohort['cohort']['fit'][:4];samples=[data[v['manifest_index']] for v in chosen]
    assert all('real_initial_pose' in s for s in samples)
    c=load_stream_config(e['arms']['s0_l1']['config']);parities=[];pilots={}
    for device,precision,count in [('cpu','fp32',1),('cuda','fp32',8),('cuda','bf16',8)]:
        base_c=dict(c,rotation_alignment=False,alignment_spatial_weight=0.,alignment_use_parent_latent=True)
        parent=make_model(base_c).to(device).eval();load_init(e['parent'],parent,audit,base_c)
        renderer=Renderer(device)
        short=dict(base_c,precision=precision,burn_in_frames=0,supervised_unroll_frames=count,startup_supervision_frames=0,
            temporal_occlusion_probability=0.,batch_current_features=False)
        with torch.no_grad():expected=StreamTrainingModule(parent,short,renderer)(samples[:1],True)['predictions']
        del parent
        for name,arm in e['arms'].items():
            cfg=load_stream_config(arm['config']);model=make_model(cfg).to(device).eval();load_init(arm['init'],model,audit,cfg)
            with torch.no_grad():actual=StreamTrainingModule(model,dict(short,rotation_alignment=True),renderer)(samples[:1],True)['predictions']
            assert torch.equal(expected,actual),(name,device,precision)
            parities.append(dict(arm=name,device=device,precision=precision,updates=count,bitwise_equal=True))
            del model
        del renderer,expected
        if device=='cuda':torch.cuda.empty_cache()
    for name,arm in e['arms'].items():
        cfg=load_stream_config(arm['config']);model=make_model(cfg).cuda().train();load_init(arm['init'],model,audit,cfg);model.rgb.requires_grad_(False)
        # Non-candidate three-step optimization only validates this exact graph.
        pilot=dict(cfg,warmup_steps=1);optimizer,scheduler=optimizer_and_scheduler(model,pilot)
        runner=StreamTrainingModule(model,pilot,Renderer('cuda'));rows=[]
        for step in range(3):
            optimizer.zero_grad(set_to_none=True);result=runner(samples);result['loss'].backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            cad=float(model.rotation_alignment.cad[0].weight.grad.norm());qk=float(model.rotation_alignment.match.attn.in_proj_weight.grad[:512].norm())
            if arm['spatial']:assert cad>0 and qk>0 and result['spatial_alignment_diagnostics'][3]>0
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip_norm']);assert torch.isfinite(grad)
            optimizer.step();scheduler.step()
            rows.append(dict(step=step+1,loss=float(result['loss'].detach()),cad_gradient_l2=cad,match_qk_gradient_l2=qk,
                spatial=result.get('spatial_alignment_diagnostics',torch.zeros(5)).cpu().tolist(),real_initializations=result['real_initializations']))
        pilots[name]=rows;print(name,json.dumps(rows),flush=True)
        del model,runner,optimizer,scheduler,result;torch.cuda.empty_cache()
    report=dict(passed=True,source_sha256=source_hash(),parent_sha256=e['parent_sha256'],parities=parities,pilots=pilots,
        samples=chosen,pilot_discarded=True,auxiliary_nonzero_cad_and_qk_gradient_at_zero_rotation_head=True,
        scope='Actual train RGB-D, real PoseCNN priors; no val/test data or candidate weights. Pixel target conventions separately tested analytically. Original parent trajectories preserved at each zero-start.')
    (r/'equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()

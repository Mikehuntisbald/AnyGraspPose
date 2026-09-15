"""Real primed-training/online parity, supervised gradients and a discarded pilot."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash,load_init
from lip.engine.stream_config import make_model,load_stream_config,optimizer_and_scheduler
from lip.engine.config import check_data_gate
from lip.engine.stream_training import StreamTrainingModule
from lip.data.stream_clips import StreamClips
from lip.geometry.renderer import Renderer
from standalone_bop_state import prime_initial_observation


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);p.add_argument('--arm',default='real_mix');a=p.parse_args()
    e=json.loads((a.experiment/'experiment.json').read_text());assert source_hash()==e['source_sha256'];torch.set_num_threads(2)
    c=load_stream_config(e['arms'][a.arm]['config']);audit=check_data_gate(e['index_root']);samples_manifest=json.loads(Path(e['training_manifest']).read_text())
    data=StreamClips(e['data_root'],e['index_root'],8,48,fixed=samples_manifest,decode_threads=2,
        external_initializers=e['train_initializers'],external_initializers_sha256=e['train_initializers_sha256'],
        real_initialization_probability=.5,include_initial_observation=True)
    samples=[data[i] for i in e['probe_indices'][:2]];assert all('real_initial_pose_original' in s for s in samples)
    results=[]
    for device,precision,count in [('cpu','fp32',8),('cuda','fp32',16),('cuda','bf16',16)]:
        model=make_model(c).to(device).eval();load_init(e['arms'][a.arm]['init'],model,audit,c);renderer=Renderer(device)
        probe=dict(c,precision=precision,burn_in_frames=0,supervised_unroll_frames=count,startup_supervision_frames=0,temporal_occlusion_probability=0.,batch_current_features=False)
        s=samples[0]
        with torch.no_grad():
            training=StreamTrainingModule(model,probe,renderer)([s],True)['predictions'][0]
            def advance(state,rgb,depth,timestamp):
                proposal,next_state=model.step(rgb,depth,timestamp,state,renderer=renderer,precision=precision,image_size=c['image_size'],crop_expansion=c['crop_expansion'])
                assert proposal['status']=='ok'
                return model.commit(proposal,next_state),proposal
            def observe(state):
                state,pred=advance(state,s['initial_rgb'],s['initial_depth'].float()*s['depth_scale'],float(s['timestamps'][0]))
                return state,pred['pose_original'],'ok',0
            state,_=prime_initial_observation(model,s['real_initial_pose_original'],s['mesh'],s['k'],s['stream']['stream_id'],
                float(s['timestamps'][0]),1/s['nominal_frame_interval'],tuple(s['rgb'].shape[-2:]),s['stream']['object_id'],s['stream']['camera_serial'],observe)
            actual=[]
            for i in range(count):
                state,pred=advance(state,s['rgb'][i],s['depth'][i].float()*s['depth_scale'],float(s['timestamps'][i+1]));actual.append(state.pose_centered.clone())
            actual=torch.stack(actual);torch.testing.assert_close(training,actual,atol=3e-5,rtol=1e-5)
            results.append(dict(device=device,precision=precision,updates=count,initial_observations=1,max_abs=float((training-actual).abs().max()),
                final_frame_id=state.frame_id,bitwise_equal=torch.equal(training,actual)))
            assert state.frame_id==count+1
        del model,renderer
        if device=='cuda':torch.cuda.empty_cache()
    model=make_model(c).cuda().train();load_init(e['arms'][a.arm]['init'],model,audit,c);model.rgb.requires_grad_(False);renderer=Renderer('cuda')
    initial={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
    # FP32 pilot checks learnability without BF16 weight-rounding plateaus during
    # the formal small-LR warmup. Its optimizer and weights are discarded.
    pilot_c=dict(c,precision='fp32',warmup_steps=1);optimizer,scheduler=optimizer_and_scheduler(model,pilot_c);pilot=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);result=StreamTrainingModule(model,pilot_c,renderer)(samples)
        result['loss'].backward();assert all(p.grad is None for p in model.rgb.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip_norm']);assert torch.isfinite(norm)
        optimizer.step();scheduler.step();pilot.append(dict(step=step+1,loss=float(result['loss'].detach()),grad_norm=float(norm),
            real_initializations=result['real_initializations'],primed_observations=result['primed_observations'],supervised_frames=result['supervised_frames']))
        assert result['real_initializations']==result['primed_observations']==2 and result['supervised_frames']==96
    changed=[n for n,t in model.state_dict().items() if not torch.equal(t.cpu(),initial[n])]
    assert changed and not any(n.startswith('rgb.') for n in changed)
    receipt=dict(passed=True,source_sha256=source_hash(),parent_sha256=e['parent_sha256'],
        train_initializers_sha256=e['train_initializers_sha256'],online_parity=results,pilot=pilot,pilot_discarded=True,
        rgb_unchanged=True,all_trainable_gradients_finite=True,changed_tensors=changed,
        scope='Actual train RGB-D and real train PoseCNN poses; CPU/CUDA online parity and discarded optimization pilot. No new val/test score or training claim.')
    (a.experiment/'equivalence.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

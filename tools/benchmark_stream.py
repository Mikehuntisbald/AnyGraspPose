"""Single-GPU batch-one timing on identical real RGB-D sequences, without GT metrics."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.data.index import read_frame
from lip.engine.features import build_features,stack_features
from lip.engine.stream_features import build_current_features,stack_current
from lip.engine.stream_config import load_stream_config,make_model
from lip.engine.stream_checkpoint import load_init,sha
from lip.engine.stream_state import FrameMeta
from lip.engine.config import check_data_gate,environment
from lip.geometry.renderer import Renderer
from lip.geometry.so3 import update,original_pose
from lip.models.tracker import Tracker


class Events:
    def __init__(self):self.pending={}
    @contextmanager
    def section(self,name):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();yield;end.record();self.pending.setdefault(name,[]).append((start,end))
    def read(self):
        return {k:sum(a.elapsed_time(b) for a,b in events) for k,events in self.pending.items()}


class CountRenderer:
    def __init__(self,renderer):self.renderer=renderer;self.calls=0;self.events=None
    def __call__(self,*args):
        self.calls+=1
        if self.events is None:return self.renderer(*args)
        with self.events.section('render'):return self.renderer(*args)


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)))


def benchmark(model,kind,config,stream,frames,times,initial,mesh,root,audit,predecoded,warmup,repeats,core):
    renderer=CountRenderer(Renderer('cuda'));counts=dict(rgb=0,geometry=0)
    def hook(name):
        def count(_,args):counts[name]+=len(args[0])
        return count
    compiled=getattr(model,'_stream_compiled',False)
    handles=[] if compiled else [model.rgb.register_forward_pre_hook(hook('rgb')),model.geometry.register_forward_pre_hook(hook('geometry'))]
    if compiled:model._encoded_images_count=0
    active={'events':None};starts={}
    def start_stage(name):
        def begin(module,args):
            event=torch.cuda.Event(enable_timing=True);event.record();starts[name]=event
        return begin
    def end_stage(name,next_name=None):
        def end(module,args,output):
            event=torch.cuda.Event(enable_timing=True);event.record()
            active['events'].pending.setdefault(name,[]).append((starts.pop(name),event))
            if next_name:starts[next_name]=event
        return end
    if kind=='legacy':
        for first,last,name in [(model.rgb,model.rgb_proj,'rgb_encoder'),(model.geometry,model.geometry,'geometry_encoder'),
                (model.fusion[0],model.fusion[-1],'spatial_fusion'),(model.temporal,model.head,'temporal_readout')]:
            handles.extend((first.register_forward_pre_hook(start_stage(name)),last.register_forward_hook(end_stage(name,'pose_update' if name=='temporal_readout' else None))))
        handles.append(model.register_forward_hook(end_stage('pose_update')))
    model.eval();k=torch.tensor(stream['intrinsics'],device='cuda');latency=[];parts=[];ages=[];cache_bytes=0
    torch.cuda.reset_peak_memory_stats();initial_allocated=torch.cuda.memory_allocated()
    with torch.no_grad():
        for repeat in range(repeats):
            if kind=='legacy':
                from lip.engine.stream_features import mesh_to_device
                resident=mesh_to_device(mesh,'cuda');base=initial.cuda();accepted=[];rgbs=[];depths=[];history_times=[]
            else:
                init_original=original_pose(initial,torch.as_tensor(mesh['center']))
                state=model.initialize(init_original,mesh,k,stream['stream_id'],times[0],object_id=stream['object_id'],camera_id=stream['camera_serial'],mesh_hash=stream['mesh_sha256'])
                resident=state.mesh;sources=[];metas=[];last_pose=state.pose_centered;previous=None
            finish=0.;start_timestamp=float(times[1]);
            for j,frame in enumerate(frames[1:],1):
                events=Events();renderer.events=events;active['events']=events;torch.cuda.synchronize();started=time.perf_counter()
                if core:rgb,depth=predecoded[j-1]
                else:
                    color,dep=read_frame(root,stream,int(frame),audit['depth_scale_to_m'])
                    rgb=torch.from_numpy(color).cuda();depth=torch.from_numpy(dep).cuda()
                if kind=='legacy':
                    rgbs.append(rgb);depths.append(depth);history_times.append(float(times[j]));n=min(j,8);pad=8-n
                    images=torch.stack([rgbs[-n]]*pad+rgbs[-n:]);dep=torch.stack([depths[-n]]*pad+depths[-n:])
                    past=accepted[-(n-1):] if n>1 else [];history=torch.stack([base]*pad+past+[base])
                    valid=torch.tensor([False]*pad+[True]*n,device='cuda');pv=valid.clone();pv[-1]=False
                    ts=torch.tensor([history_times[-n]]*pad+history_times[-n:],device='cuda')
                    with events.section('preprocess_including_render'):
                        f,_=build_features(images,dep,history,ts,valid,pv,base,k,resident,renderer,224,2.)
                    with events.section('legacy_network'),torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                        out=model(**stack_features([f]))
                    base=out['pose_centered'][0];accepted.append(base);pose=out['pose_original'][0]
                elif kind=='reference':
                    with events.section('preprocess_including_render'):
                        f,d=build_current_features(rgb,depth,last_pose,k,resident,renderer,times[j],times[j-1],previous,None if j<2 else times[j-2])
                    meta=FrameMeta(torch.tensor([times[j]],device='cuda',dtype=torch.float64),torch.tensor([j],device='cuda'),torch.zeros(1,device='cuda',dtype=torch.long),torch.ones(1,17,device='cuda',dtype=torch.bool),d['role_bias'][None])
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                        sources.append(model.encode_current(stack_current([f]),events));metas.append(meta)
                        q=(model.readout.query+model.token_type[2])[:,None].expand(1,len(sources),-1,-1)
                        with events.section('temporal_readout'):
                            zs,_=model.temporal.full_reference(torch.stack(sources,1),q,metas)
                            z=model.readout(zs[:,-1]);delta=model.head(z['latent']).float()
                    with events.section('pose_update'):
                        next_pose=update(last_pose[None],delta[:,:3],delta[:,3:],f['object_diameter_m'][None])[0]
                        previous=last_pose;last_pose=next_pose;pose=original_pose(last_pose,resident['center'])
                else:
                    # Feature construction contains its own event-timed render. The
                    # total core wall clock includes transactional state checks.
                    with events.section('tracker_total'):
                        proposal,next_state=model.step(rgb,depth,times[j],state,renderer=renderer,precision=config['precision'],profiler=events)
                        if proposal['status']!='ok':raise RuntimeError('Timing sequence needs external reinit: '+proposal['status'])
                        state=model.commit(proposal,next_state);pose=proposal['pose_original'];cache_bytes=state.cache.kv_bytes
                if not core:pose=pose.cpu() # Explicit output boundary included in E2E.
                torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
                values=events.read()
                if 'preprocess_including_render' in values:values['preprocess']=max(0.,values['preprocess_including_render']-values.get('render',0.))
                elif kind not in ('legacy','reference'):
                    accounted=sum(values.get(k,0.) for k in ['render','rgb_encoder','geometry_encoder','spatial_fusion','compiled_encoder_fusion','temporal_readout','pose_update'])
                    values['preprocess_and_state_checks']=max(0.,values['tracker_total']-accounted)
                arrival=(float(times[j])-start_timestamp)*1000;finish=max(finish,arrival)+elapsed
                if j>warmup:latency.append(elapsed);parts.append(values);ages.append(finish-arrival)
    for h in handles:h.remove()
    n=(len(frames)-1)*repeats
    if compiled:counts={name:model._encoded_images_count for name in counts}
    return dict(kind=kind,scope='predecoded GPU input tracker core wall-clock' if core else 'decode + H2D + tracker + final pose D2H wall-clock',
        latency_ms=distribution(latency),components_gpu_ms={k:distribution([r[k] for r in parts if k in r]) for k in sorted(set(k for r in parts for k in r))},
        encoder_images=counts,encoder_images_per_update={k:v/n for k,v in counts.items()},render_calls=renderer.calls,render_calls_per_update=renderer.calls/n,
        fp_calls=0,critic_calls=0,cache_bytes=cache_bytes,initial_allocated=initial_allocated,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        simulated_queue=dict(policy='process every frame, no drop; measured service times and original dt; no live camera claim',dropped=0,age_ms=distribution(ages)),
        warmup_frames_each_repeat=warmup,repeats=repeats,observed_updates=n)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dual-checkpoint');p.add_argument('--dual-config',default='configs/stream_lip_v2_dual.yaml');p.add_argument('--legacy-checkpoint')
    p.add_argument('--data-root',required=True);p.add_argument('--index-root',default='cache/dexycb_s0');p.add_argument('--out',required=True)
    p.add_argument('--frames',type=int,default=48);p.add_argument('--warmup',type=int,default=8);p.add_argument('--repeats',type=int,default=3);p.add_argument('--reference',action='store_true')
    p.add_argument('--compile-stream',action='store_true');a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('A free CUDA GPU is required for timing')
    torch.cuda.set_device(0);torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    c=load_stream_config(a.config);audit=check_data_gate(a.index_root)
    streams=sorted([s for s in map(json.loads,(Path(a.index_root)/'streams.jsonl').read_text().splitlines()) if s['split']=='val'],key=lambda s:s['stream_id'])
    stream=dict(streams[0]);stream['mesh_sha256']=sha(Path(a.index_root)/stream['mesh_cache'])
    with np.load(Path(a.index_root)/stream['pose_cache']) as z:frames=z['frames'][:a.frames+1].copy();pose=torch.from_numpy(z['poses'][0].copy());times=z['timestamps'][:len(frames)].copy() if 'timestamps' in z else frames.astype('f8')/audit['fps']
    with np.load(Path(a.index_root)/stream['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
    from lip.geometry.so3 import center_pose
    initial=center_pose(pose,torch.tensor(mesh['center']));data=[]
    for frame in frames[1:]:
        rgb,depth=read_frame(a.data_root,stream,int(frame),audit['depth_scale_to_m']);data.append((torch.from_numpy(rgb).cuda(),torch.from_numpy(depth).cuda()))
    reports={};configs=[('stream_single',a.checkpoint,c)]
    if a.legacy_checkpoint:configs.insert(0,('legacy',a.legacy_checkpoint,c))
    if a.dual_checkpoint:configs.append(('stream_dual',a.dual_checkpoint,load_stream_config(a.dual_config)))
    if a.reference:configs.append(('reference',a.checkpoint,c))
    compilation={}
    for kind,path,config in configs:
        if kind=='legacy':
            model=Tracker(False,dropout=0.).cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        else:model=make_model(config).cuda();load_init(path,model,audit,config)
        if a.compile_stream and kind not in ('legacy','reference'):
            from lip.engine.stream_features import mesh_to_device
            model.eval();original_encoder=model.encode_current
            resident=mesh_to_device(mesh,'cuda')
            features,_=build_current_features(data[0][0],data[0][1],initial.cuda(),torch.tensor(stream['intrinsics'],device='cuda'),resident,Renderer('cuda'),times[1],times[0])
            features=stack_current([features])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                expected=original_encoder(features)
                optimized=torch.compile(original_encoder,mode='reduce-overhead',fullgraph=True,dynamic=False)
                started=time.perf_counter()
                for _ in range(3):actual=optimized(features)
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                error=(actual-expected).abs().max().item()
                # BF16 compilation can choose a different fused reduction order;
                # record the actual discrepancy without weakening FP32 cache tests.
                if not torch.isfinite(actual).all():raise FloatingPointError('Compiled encoder produced nonfinite tokens')
            compilation[kind]=dict(mode='reduce-overhead',region='encode_current only; temporal and state API remain eager',
                warmup_seconds=seconds,warmup_images=3,precision=config['precision'],encoder_max_abs_error=error)
            model._stream_compiled=True;model._encoded_images_count=0
            def wrapped(features,profiler=None,compiled_fn=optimized,owner=model):
                owner._encoded_images_count+=features['rgb'].shape[0]
                if profiler is None:return compiled_fn(features)
                with profiler.section('compiled_encoder_fusion'):return compiled_fn(features)
            model.encode_current=wrapped
        # Reference is intentionally bounded to a separate short correctness/timing scope.
        length=min(len(frames),17) if kind=='reference' else len(frames)
        warm=min(a.warmup,length-2);repeat=1 if kind=='reference' else a.repeats
        reports[kind]=dict(checkpoint_sha256=sha(path),frames=frames[:length].tolist(),
            core=benchmark(model,kind,config,stream,frames[:length],times[:length],initial,mesh,a.data_root,audit,data[:length-1],warm,repeat,True),
            end_to_end=benchmark(model,kind,config,stream,frames[:length],times[:length],initial,mesh,a.data_root,audit,data[:length-1],warm,repeat,False))
        del model;torch.cuda.empty_cache()
        (out/'progress.json').write_text(json.dumps(dict(completed=list(reports)),indent=2))
    report=dict(completed=True,environment=environment(),stream_id=stream['stream_id'],split_hash=audit['split_hash'],mesh_hash=audit['mesh_hash'],
        initial_pose_source='GT first frame only',GT_metrics=False,overlay=False,compile=a.compile_stream,compilation=compilation,mode='stream encoder compiled' if a.compile_stream else 'eager',batch_size=1,
        attention_backend='PyTorch SDPA automatic dispatch; explicit float mask and is_causal=False',timing='CUDA events for GPU sections, synchronize once at each frame boundary; wall-clock latency includes normal API validation',
        configurations=reports,memory_frames_1='Not trained; intentionally not reported as an accuracy ablation')
    (out/'benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{s:v[s]['latency_ms'] for s in ('core','end_to_end')} for k,v in reports.items()},indent=2))

if __name__=='__main__':main()

"""Same-model CUDA proof: detached AMP preview changes gradients, not predictions."""
import argparse,importlib.util,json,sys,types
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes
from lip.unified.cad_surface import CADSurfaceTracker
from lip.engine.jepa_checkpoint import load_core,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--legacy-source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    record=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(m,record['model']);del record
    m.requires_grad_(True).eval();m.fast_geometry=m.vector_geometry=True
    factory=Factory(c,m,make_store(c,m));e,targets=factory.sample(19000000,frames=12);frame=8
    scene=prepare_scene(e.rgb[frame],e.depth[frame],e.initial,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,fast=True)
    occ=e.occlusion_plan.render(scene,frame);obs=encode_scenes(m,[scene],occlusions=[occ])
    spec=importlib.util.spec_from_file_location('lip.unified.legacy_preview_for_diagnostic',a.legacy_source);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    params=dict(m.named_parameters());names=('core.src_proj.weight','core.src_proj.bias','visibility.1.weight','visibility.1.bias');result={};outputs={}
    for name,fn in [('legacy',legacy.CADSurfaceTracker.extra_frame_inputs),('fixed',CADSurfaceTracker.extra_frame_inputs)]:
        def serial_inputs(self,observation,memory=None,history_enabled=None):
            state,history,_,cad=fn(self,observation,memory,history_enabled)
            return state,history,(observation.crop_rays,observation.diameter,observation.mid,observation.last,observation.measured_depth_m),cad
        m.extra_frame_inputs=types.MethodType(serial_inputs,m)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output,_=m(obs)
            loss=output['f_predicted'][...,0].float().mean()+output['evidence_logits'].float().square().mean()
        grads=torch.autograd.grad(loss,[params[k] for k in names],allow_unused=True,retain_graph=True)
        result[name]={k:None if g is None else float(g.norm()) for k,g in zip(names,grads)}
        outputs[name]={k:output[k].detach().clone() for k in ('patch_latent','f_predicted','surface_xyz','evidence_logits','pose_centered')}
    differences={k:float((outputs['fixed'][k]-v).abs().max()) for k,v in outputs['legacy'].items()}
    assert all(v is None for v in result['legacy'].values())
    assert all(v is not None and v>0 for v in result['fixed'].values())
    assert all(v==0 for v in differences.values()),differences
    receipt=dict(completed=True,device=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint_sha256=sha(a.checkpoint),legacy_source_sha256=sha(a.legacy_source),
        fixed_source_sha256=sha(Path(__file__).resolve().parents[1]/'src/lip/unified/cad_surface.py'),gradient_norms=result,forward_max_abs=differences,optimizer_updates=0,training_data_only=True)
    Path(a.out).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

"""Real train-development examples: restored features/depth, never invented RGB."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.unified.features import prepare_scene,encode_scenes,build_teachers
from lip.engine.jepa_checkpoint import load_core,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);c=yaml.safe_load(Path(a.config).read_text());m=build_model(c)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);load_core(m,saved['model']);del saved
    m.requires_grad_(False).eval();m.fast_geometry=m.vector_geometry=True
    factory=Factory(c,m,make_store(c,m));cache=torch.load(Path(a.cache)/'rank0/packets.pt',weights_only=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);records=[]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with torch.no_grad():
        for i,record in enumerate([x for x in cache if x['split']=='holdout'][:4]):
            e,(truth,visible)=factory.sample(record['seed'],frames=12);frame=8
            scene=prepare_scene(e.rgb[frame],e.depth[frame],e.initial,e.mesh,e.k,e.times[frame],e.stream,e.cad,factory.renderer,fast=True)
            occ=e.occlusion_plan.render(scene,frame);obs=encode_scenes(m,[scene],occlusions=[occ])
            target=build_teachers(m.ema_teacher,[scene],truth[frame:frame+1],visible[frame:frame+1],[occ.mask],factory.renderer,real_geometry_max_radius_d=1.,fast=True,batch_render=True,vectorized=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction,_=m(obs)
            hidden=target.geometry_weight[0,0].cpu().numpy();d=scene.diameter
            xyz=(prediction['surface_xyz']-target.surface_xyz).norm(dim=1)[0].cpu().numpy()*d*1000
            depth=(prediction['surface_depth_m']-target.surface_depth_m).abs()[0,0].cpu().numpy()*1000
            feature=np.full((16,16),np.nan)
            for teacher,weight in ((target.real_mid,target.hidden_real_weight),(target.proxy_mid,target.proxy_weight)):
                pred=torch.nn.functional.normalize(prediction['f_mid_predicted'].float(),dim=-1)
                teacher=torch.nn.functional.normalize(teacher.float(),dim=-1)
                # Spatial centering removes each map's common object component.
                pred=pred-pred.mean(1,keepdim=True);teacher=teacher-teacher.mean(1,keepdim=True)
                value=(pred-teacher).norm(dim=-1)[0].reshape(16,16).cpu().numpy()
                selected=weight[0].reshape(16,16).cpu().numpy()>0;feature[selected]=value[selected]
            image=lambda x:x[0].permute(1,2,0).cpu().numpy().clip(0,1)
            obj=prediction['support_logits'][0].sigmoid().reshape(16,16).repeat_interleave(14,0).repeat_interleave(14,1).cpu().numpy()>.5
            panels=[('Student RGB-D occluded RGB',image(occ.rgb),'rgb'),('Estimated-pose CAD RGB',image(scene.render['rgb'][None]),'rgb'),
                ('Full CAD teacher overlay',image(target.proxy_rgb),'rgb'),('Predicted depth (m)',np.where(obj,prediction['surface_depth_m'][0,0].cpu().numpy(),np.nan),'depth'),
                ('Hidden target depth (m)',np.where(hidden,target.surface_depth_m[0,0].cpu().numpy(),np.nan),'depth'),
                ('Hidden depth error (mm)',np.where(hidden,depth,np.nan),'error'),('Hidden canonical XYZ error (mm)',np.where(hidden,xyz,np.nan),'error'),
                ('Layer4 centered feature error',feature,'feature')]
            fig,axes=plt.subplots(2,4,figsize=(14,7))
            for ax,(title,values,kind) in zip(axes.flat,panels):
                im=ax.imshow(values) if kind=='rgb' else ax.imshow(values,cmap='viridis' if kind=='depth' else 'magma')
                ax.set_title(title,fontsize=9);ax.axis('off')
                if kind!='rgb':fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
            fig.suptitle('V21: train-development frame, native initializer; teacher/error panels use GT\nTexture recovery is DINO features, not generated RGB',fontsize=11)
            fig.tight_layout();fig.savefig(out/f'example{i}.png',dpi=160);plt.close(fig)
            records.append(dict(seed=record['seed'],stream=e.stream,frame=frame,hidden_pixels=int(hidden.sum()),
                xyz_mm=float(xyz[hidden].mean()) if hidden.any() else None,depth_mm=float(depth[hidden].mean()) if hidden.any() else None))
    (out/'receipt.json').write_text(json.dumps(dict(checkpoint_sha256=sha(a.checkpoint),examples=records,training=False,official_test_access=False,
        teacher='checkpoint EMA4/11; oracle evaluation cache supplies selection seeds only'),indent=2)+'\n')

if __name__=='__main__':main()

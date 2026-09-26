"""Check actual late-frame bytes and empirical error transport before optimization."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.unified.build import build_model,make_store
from lip.unified.training import Factory
from lip.engine.jepa_checkpoint import load_core,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    configs={arm:yaml.safe_load(Path('configs/jepa/full_window_v33_'+arm+'.yaml').read_text()) for arm in ('prefix','full')}
    model=build_model(configs['full']);path=configs['full']['serial_completion']['source_checkpoint']
    saved=torch.load(path,map_location='cpu',weights_only=False);load_core(model,saved['model']);del saved
    model.requires_grad_(False).eval();factories={arm:Factory(c,model,make_store(c,model)) for arm,c in configs.items()}
    rows=[]
    for seed in range(330000,330064):
        ep,(gt,mask)=factories['full'].sample(seed,frames=10)
        if ep.training_window['first_frame']<=28:continue
        before,(oldgt,_)=factories['prefix'].sample(seed,frames=10)
        assert before.stream==ep.stream
        torch.testing.assert_close(ep.initial[:3,:3]@gt[0,:3,:3].T,before.initial[:3,:3]@oldgt[0,:3,:3].T,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(ep.initial[:3,3]-gt[0,:3,3],before.initial[:3,3]-oldgt[0,:3,3],atol=2e-6,rtol=2e-6)
        first=ep.training_window['first_frame'];sid=ep.stream.split('|')[0];stream=factories['full'].streams[sid];directory=factories['full'].root/stream['relative_dir']
        for offset in (0,9):
            frame=first+offset
            rgb=cv2.cvtColor(cv2.imread(str(directory/f'color_{frame:06d}.jpg')),cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()
            depth=cv2.imread(str(directory/f'aligned_depth_to_color_{frame:06d}.png'),-1).astype('f4')[None]
            torch.testing.assert_close(ep.rgb[offset],torch.from_numpy(rgb).cuda().float()/255,rtol=0,atol=0)
            torch.testing.assert_close(ep.depth[offset],torch.from_numpy(depth).cuda()*factories['full'].audit['depth_scale_to_m'],rtol=0,atol=0)
            with np.load(directory/f'labels_{frame:06d}.npz',allow_pickle=False) as z:visible=z['seg']==stream['object_id']
            assert torch.equal(mask[offset],torch.from_numpy(visible[None].copy()).cuda())
            assert abs(ep.times[offset]-frame/factories['full'].audit['fps'])<1e-10
        rows.append(dict(seed=seed,stream=sid,**ep.training_window))
        if len(rows)==8:break
    assert len(rows)==8 and sum(r['raw_native_frames'] for r in rows)>0
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(dict(completed=True,optimizer_updates=0,checkpoint_sha256=sha(path),rows=rows,
        native_rgb_depth_mask_exact=True,empirical_error_preserved=True,training_only_gt_augmentation=True,official_test_access=False),f,indent=2)
    print(json.dumps(dict(completed=True,windows=len(rows),raw_frames=sum(r['raw_native_frames'] for r in rows))),flush=True)

if __name__=='__main__':main()

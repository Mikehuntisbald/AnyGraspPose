"""Audit nonempty-target metrics; saved fit curves include zero-valued empty lanes."""
import argparse,hashlib,json,sys,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fit_serial_geometry_decoder import stack
from lip.unified.dpt_surface import DPTSurfaceHead
from lip.unified.cad_coordinate_decoder import CADCoordinateDecoder

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    rows=[]
    for f in sorted((a.root/'cache').glob('rank*/decoder_cache.pt')):rows+=torch.load(f,weights_only=False)
    physical=lambda r:'/'.join(r['stream'].split('|')[0].split('/')[:2])
    rows=[r for r in rows if int(hashlib.sha256(physical(r).encode()).hexdigest()[:8],16)%4==0]
    results={}
    with torch.no_grad():
        for arm in ('parent','dpt','cad'):
            model=(CADCoordinateDecoder(rows[0]['cad_features'].shape[-1]) if arm=='cad' else DPTSurfaceHead()).cuda().eval()
            file=a.root/arm/'decoder_diagnostic.pt' if arm!='parent' else a.root.parent/'serial_completion_v21/runs/seed42/last.pt'
            saved=torch.load(file,map_location='cpu',weights_only=False)
            weights=saved['model'] if arm!='parent' else {k.removeprefix('surface_head.'):v for k,v in saved['model'].items() if k.startswith('surface_head.')}
            model.load_state_dict(weights);del saved,weights;values={k:[] for k in ('real','proxy','visible')}
            for start in range(0,len(rows),8):
                part=rows[start:start+8];t=stack(part)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(t['levels'],t['valid'],t['cad_features'],t['cad_geometry'],t['cad_available'])[0] if arm=='cad' else model(t['levels'],t['valid'])
                for name,mask,target in [('real',t['real_mask'],t['xyz']),('proxy',t['proxy_mask'],t['xyz']),('visible',t['observed_mask'],t['observed_xyz'])]:
                    xyz=(pred[:,:3]-target).norm(dim=1,keepdim=True)*t['diameter'][:,None,None,None]*1000
                    depth=(pred[:,3:4]-t['depth']).abs()*t['diameter'][:,None,None,None]*1000
                    for i,r in enumerate(part):
                        count=int(mask[i].sum())
                        if count:values[name].append(dict(physical=physical(r),pixels=count,xyz_mm=float(xyz[i][mask[i]].mean()),**({} if name=='visible' else dict(depth_mm=float(depth[i][mask[i]].mean())))))
            results[arm]={}
            for name,v in values.items():
                keys=['xyz_mm']+([] if name=='visible' else ['depth_mm'])
                sequences=sorted({r['physical'] for r in v})
                results[arm][name]=dict(nonempty_frames=len(v),physical_sequences=len(sequences),
                    frame_mean={k:statistics.mean(r[k] for r in v) for k in keys},
                    sequence_mean={k:statistics.mean(statistics.mean(r[k] for r in v if r['physical']==s) for s in sequences) for k in keys})
            del model
    result=dict(heldout_records=len(rows),empty_targets_excluded=True,source='training-split physical holdout only; frozen JEPA cache; not native pose',results=results)
    (a.root/'nonempty_metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

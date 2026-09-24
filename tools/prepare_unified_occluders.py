"""Extract real hand/object RGB-D cutouts exclusively from the train split."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.jepa_checkpoint import sha
from lip.engine.object_jepa_checkpoint import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--per-kind',type=int,default=96);a=p.parse_args()
    cv2.setNumThreads(0);torch.set_num_threads(1)
    audit=json.loads((a.index/'audit.json').read_text());streams=list(map(json.loads,(a.index/'streams.jsonl').read_text().splitlines()))
    legal={s['relative_dir']:s for s in streams if s['split']=='train'}
    heldout={s['relative_dir'] for s in streams if s['split']!='train'};assert not set(legal)&heldout
    directories=sorted(legal);rng=np.random.default_rng(42);rng.shuffle(directories)
    cutouts=[];counts={'hand':0,'object':0};source_hashes={};physical=set();objects=set()
    for directory in directories:
        s=legal[directory];physical_id='/'.join(directory.split('/')[:2])
        if physical_id in physical:continue
        physical.add(physical_id)
        for fraction in (.25,.5,.75):
            frame=min(s['num_frames']-1,int(s['num_frames']*fraction));d=a.root/directory
            rp=d/f'color_{frame:06d}.jpg';dp=d/f'aligned_depth_to_color_{frame:06d}.png';mp=d/f'labels_{frame:06d}.npz'
            with np.load(mp,allow_pickle=False) as z:seg=z['seg'].copy()
            rgb=cv2.cvtColor(cv2.imread(str(rp)),cv2.COLOR_BGR2RGB);depth=cv2.imread(str(dp),-1).astype('f4')*audit['depth_scale_to_m']
            labels=[255]+[int(i) for i in np.unique(seg) if i not in (0,255)];rng.shuffle(labels)
            for label in labels:
                kind='hand' if label==255 else 'object'
                if counts[kind]>=a.per_kind:continue
                mask=(seg==label).astype('uint8');n,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
                if n<=1:continue
                component=1+int(stats[1:,cv2.CC_STAT_AREA].argmax());area=int(stats[component,cv2.CC_STAT_AREA])
                if area<800:continue
                x,y,w,h=[int(v) for v in stats[component,:4]]
                if min(w,h)<20:continue
                alpha=(components[y:y+h,x:x+w]==component).astype('uint8')
                # RGB outside donor support is filled by the nearest supported pixel
                # before interpolation, preventing black/background fringes.
                import scipy.ndimage
                _,nearest=scipy.ndimage.distance_transform_edt(alpha==0,return_indices=True)
                color=rgb[y:y+h,x:x+w].copy();color=color[nearest[0],nearest[1]]
                dep=depth[y:y+h,x:x+w].copy()*alpha
                scale=min(1.,256/max(w,h));size=(max(1,round(w*scale)),max(1,round(h*scale)))
                color=cv2.resize(color,size,interpolation=cv2.INTER_AREA);alpha=cv2.resize(alpha,size,interpolation=cv2.INTER_NEAREST).astype(bool)
                dep=cv2.resize(dep,size,interpolation=cv2.INTER_NEAREST);good=alpha&(dep>0)
                if good.sum()<.5*alpha.sum():continue
                median=float(np.median(dep[good]));relative=np.where(good,dep-median,0.).astype('f4')
                entry=dict(rgb=torch.from_numpy(color.transpose(2,0,1).copy()),alpha=torch.from_numpy(alpha[None].copy()),
                    relative_depth=torch.from_numpy(relative[None]),depth_valid=torch.from_numpy(good[None]),
                    kind=kind,object_id=label if kind=='object' else None,source_split='train',source_dir=directory,
                    frame=frame,source_bbox=[x,y,w,h],median_depth_m=median)
                cutouts.append(entry);counts[kind]+=1
                if kind=='object':objects.add(label)
                for path in (rp,dp,mp):source_hashes[str(path.relative_to(a.root))]=sha(path)
            if min(counts.values())>=a.per_kind:break
        if min(counts.values())>=a.per_kind:break
    if min(counts.values())<a.per_kind:raise RuntimeError(f'Insufficient train donors: {counts}')
    a.out.mkdir(parents=True,exist_ok=False)
    bank=a.out/'bank.pt';torch.save(dict(version='train-real-rgbd-cutouts-v1',cutouts=cutouts),bank)
    receipt=dict(completed=True,version='train-real-rgbd-cutouts-v1',split='train',seed=42,counts=counts,object_ids=sorted(objects),
        physical_sequences=len({'/'.join(x['source_dir'].split('/')[:2]) for x in cutouts}),
        split_hash=audit['split_hash'],bank_sha256=sha(bank),preprocessing_sha256=sha(__file__),source_files_sha256=source_hashes,
        hand_label=255,max_cutout_side=256,official_test_access=False,validation_donors=False)
    atomic_json(a.out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='source_files_sha256'},indent=2))

if __name__=='__main__':main()

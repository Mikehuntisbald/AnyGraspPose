"""CPU-only scoring audit: base rotation/translation and fixed40 geometry error."""
import argparse,json,statistics,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path('/mnt/why/dexycb_lip');index=root/'cache/dexycb_s0'
    streams={x['stream_id']:x for x in map(json.loads,(index/'streams.jsonl').read_text().splitlines())}
    refpath=root/'smooth_val_evaluation_20260915/runs/full/residual/scored/predictions.jsonl'
    reference={(x['stream_id'],x['frame_index']):x for x in map(json.loads,refpath.read_text().splitlines())}
    initial=json.loads((root/'posecnn_val_20260915/runs/full/initializers.json').read_text())['initializers']
    info=json.loads((root/'cache/raw_full_20260910/bop/models/models_info.json').read_text())
    targets={};centers={};rows=[]
    for path in sorted(Path(a.run).glob('rank*/frames.jsonl')):
        manifest=json.loads(path.with_name('manifest.json').read_text());assert manifest['completed']
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==manifest['frames_sha256']
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['history']!='off' or r['phase']!='occlusion' or not r['case'].startswith('heavy'):continue
            sid=r['stream_id'];frame=r['frame_index'];s=streams[sid]
            if sid not in targets:
                with np.load(index/s['pose_cache']) as z:targets[sid]=z['poses'].copy()
                with np.load(index/s['mesh_cache']) as z:centers[sid]=z['center'].copy()
            gt=targets[sid][frame];base=np.asarray(initial[sid]['pose_original'] if r['relative_frame']==0 else reference[sid,frame-1]['pose_original'])
            angle=float(np.degrees(np.arccos(np.clip((np.trace(base[:3,:3]@gt[:3,:3].T)-1)/2,-1,1))))
            center=centers[sid]
            offset=float(np.linalg.norm(base[:3,3]+base[:3,:3]@center-gt[:3,3]-gt[:3,:3]@center)*1000)
            metadata=info[str(s['object_id'])];symmetric=bool(metadata.get('symmetries_discrete') or metadata.get('symmetries_continuous'))
            rows.append(dict(physical=r['physical_sequence'],stream=sid,frame=frame,case=r['case'],object_id=s['object_id'],symmetric=symmetric,
                base_rotation_deg=angle,base_center_mm=offset,metrics={k:r[k] for k in ('geometry_canonical_real','geometry_canonical_proxy')}))
    tables={}
    for sym in ('all','declared_symmetric','nonsymmetric'):
        for name,low,high in [('all',-1,181),('le15',-1,15),('15to45',15,45),('gt45',45,181)]:
            subset=[r for r in rows if low<r['base_rotation_deg']<=high and (sym=='all' or r['symmetric']==(sym=='declared_symmetric'))]
            result=dict(frames_and_cases=len(subset),physical_sequences=len({r['physical'] for r in subset}),objects=sorted({r['object_id'] for r in subset}))
            for kind in ('real','proxy'):
                metrics={}
                for r in subset:
                    d=r['metrics']['geometry_canonical_'+kind]
                    if d is None:continue
                    for k,v in dict(xyz_mm=d['xyz_mm'],depth_mm=d['depth_mm'],base_rotation_deg=r['base_rotation_deg'],base_center_mm=r['base_center_mm']).items():
                        metrics.setdefault(k,{}).setdefault(r['physical'],{}).setdefault(r['case'],[]).append(v)
                result[kind]={k:statistics.mean(statistics.mean(statistics.mean(v) for v in cases.values()) for cases in groups.values()) for k,groups in metrics.items()}
            tables[sym+'/'+name]=result
    Path(a.out).write_text(json.dumps(dict(completed=True,training=False,optimizer_updates=0,rows=len(rows),tables=tables,
        reference_sha256=hashlib.file_digest(refpath.open('rb'),'sha256').hexdigest(),
        reduction='Mean within case, equal heavy-case mass within sequence, equal sequence mass',
        scope='Descriptive subsets of fixed40 heavy cases; populations vary by bin. Geodesic rotation is not symmetry reduced; declared symmetry alone does not establish texture equivalence.'),indent=2)+'\n')


if __name__=='__main__':main()

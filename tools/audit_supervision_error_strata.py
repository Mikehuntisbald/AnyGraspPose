"""Paired saved-example error strata; primary unfiltered metrics stay intact."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--v40',type=Path,required=True);p.add_argument('--v41',type=Path,required=True);a=p.parse_args()
    rows=[]
    for rank in range(8):
        examples={key:np.load(root/'probe'/f'step{step}'/f'rank{rank}'/'heavy_example.npz') for key,root,step in [('source',a.v41,0),('v40',a.v40,100),('v41',a.v41,100)]}
        ref=examples['source']
        for x in examples.values():
            for key in ('target_depth','target_xyz','cad_target_xyz','cad_target_depth','real_mask','proxy_mask','diameter'):
                assert np.array_equal(x[key],ref[key]),key
        gap=np.abs(ref['target_depth']-ref['cad_target_depth'])*1000
        regions={'all_real':ref['real_mask'],'real_gap_le10mm':ref['real_mask']&(gap<=10),
                 'real_gap_10to30mm':ref['real_mask']&(gap>10)&(gap<=30),'real_gap_gt30mm':ref['real_mask']&(gap>30),'all_proxy':ref['proxy_mask']}
        for region,mask in regions.items():
            if not mask.any():continue
            for model,x in examples.items():
                xyz=np.linalg.norm(x['predicted_xyz']-ref['cad_target_xyz'],axis=1,keepdims=True)*float(ref['diameter'])*1000
                depth=np.abs(x['predicted_depth']-ref['target_depth'])*1000
                rows.append(dict(rank=rank,region=region,model=model,pixels=int(mask.sum()),cad_xyz_mm=float(xyz[mask].mean()),depth_mm=float(depth[mask].mean())))
    tables={}
    for region in sorted({r['region'] for r in rows}):
        tables[region]={}
        for model in ('source','v40','v41'):
            selected=[r for r in rows if r['region']==region and r['model']==model]
            tables[region][model]={key:statistics.mean(r[key] for r in selected) for key in ('cad_xyz_mm','depth_mm')}
            tables[region][model]['examples']=len(selected)
    result=dict(completed=True,training=False,scope='Eight saved heavy examples only; equal example mass; paired within each fixed target stratum. Secondary diagnostic, never replaces full64 unfiltered metrics.',tables=tables,rows=rows)
    (a.v41/'error_strata.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(tables,indent=2))


if __name__=='__main__':main()

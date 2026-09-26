"""Paired physical-sequence means for the conditional native-frame diagnostic."""
import argparse,json
from pathlib import Path
from report_cad_image_v45 import reduce


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arms',nargs='+',default=['source','parent','trained']);a=p.parse_args()
    data={}
    for arm in a.arms:
        rows={}
        for path in sorted((a.root/arm).glob('rank*/frames.jsonl')):
            assert json.loads((path.parent/'receipt.json').read_text())['completed']
            for line in path.read_text().splitlines():
                r=json.loads(line);assert r['seed'] not in rows;rows[r['seed']]=r
        assert len({r['physical_sequence'] for r in rows.values()})==40
        data[arm]=rows
    reference=next(iter(data.values()))
    for rows in data.values():
        assert rows.keys()==reference.keys()
        for key,r in rows.items():
            other=reference[key]
            assert all(r[k]==other[k] for k in ('stream','physical_sequence','frame_index','heavy'))
            assert r['correspondence']['pose']['base']==other['correspondence']['pose']['base']
            for kind in ('real','proxy'):
                x,y=r['metrics'][kind],other['metrics'][kind];assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels']
    results={arm:{name:reduce({key:r for key,r in rows.items() if r['heavy']==heavy},False) for name,heavy in [('natural',False),('heavy',True)]} for arm,rows in data.items()}
    result=dict(completed=True,physical_sequences=40,frames=len(reference)//2,records_per_model=len(reference),
        paired_masks_and_bases=True,scope='Conditional one-step, previous sealed LIP pose/crop, not native closed-loop or official test',
        reduction='equal physical-sequence means',metrics=results,default_model_changed=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2))
    lines=['# Fixed40 conditional native-frame correspondence evaluation','','Previous-frame sealed LIP estimates define the common crop/base.119 frames and238 natural/heavy records per model. No GT student input; no closed-loop claim. Raw rotation is not symmetry reduced.','',
           '|Model/input|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---:|---:|---:|---:|']
    for arm,settings in results.items():
        for setting,m in settings.items():
            vals=[m.get('/geometry/'+region+'/'+metric) for region,metric in [('real','canonical_xyz_mm'),('real','depth_mm'),('proxy','canonical_xyz_mm'),('proxy','depth_mm')]]
            lines.append('|'+arm+'/'+setting+'|'+'|'.join('—' if v is None else f'{v:.3f}' for v in vals)+'|')
    lines+=['','|Model/input/readout|ADD-S mm|ADD-S<0.05d %|Accepted %|','|---|---:|---:|---:|']
    methods=['base','frozen_head','cad_to_image','cad_to_image_visible','cad_to_image_rgbd_mixed','image_to_cad_rgbd_mixed','anchored_flow','anchored_flow_rgbd_mixed']
    for arm,settings in results.items():
        for setting,m in settings.items():
            for method in methods:
                prefix='/correspondence/pose/'+method
                if prefix+'/adds_mm' not in m:continue
                accepted=m.get('/correspondence/solvers/'+method+'/accepted')
                lines.append(f"|{arm}/{setting}/{method}|{m[prefix+'/adds_mm']:.3f}|{m[prefix+'/adds_005d']*100:.3f}|"+('—' if accepted is None else f'{accepted*100:.1f}')+'|')
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

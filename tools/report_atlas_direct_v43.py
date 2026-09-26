"""Paired100-update result; direct CE vs assisted CE, identical inference."""
import argparse,json
from pathlib import Path
import torch
from report_visible_correspondence import exact
from report_geometry_supervision_v41 import read,reduce


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.root/'status.json').read_text())['completed']
    starts=[torch.load(a.root/arm/'runs/seed42/initial.pt',map_location='cpu',weights_only=False) for arm in ('assisted','direct')]
    identity={k:exact(starts[0][k],starts[1][k]) for k in ('model','optimizer','scheduler','rng','sampler_position')};assert all(identity.values())
    for i,arm in enumerate(('assisted','direct')):
        root=a.root/arm;end=torch.load(root/'runs/seed42/last.pt',map_location='cpu',weights_only=False);assert end['step']==100
        names=[n for n in end['model'] if n=='query' or n.startswith(('head.','object_attn.','object_norm.','geometry_readout.','core.feature_','cad_transport.'))]
        assert all(torch.equal(end['model'][n],starts[i]['model'][n]) for n in names)
        assert json.loads((root/'runs/seed42/resume2.json').read_text())['complete_state_verified']
        del end
    del starts
    rows={}
    for arm in ('assisted','direct'):
        for step in (0,100):
            for clean in (False,True):rows[f'{arm}_{step}_{"clean" if clean else "corrupted"}']=read(a.root/arm,step,clean)
    reference=rows['assisted_0_corrupted']
    for label,records in rows.items():
        assert records.keys()==reference.keys()
        for seed,r in records.items():
            other=reference[seed];assert all(r[k]==other[k] for k in ('stream','heavy','natural','window'))
            for kind in ('real','proxy'):
                x,y=r['metrics'][kind],other['metrics'][kind];assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    result=dict(completed=True,updates_per_arm=100,matched_initial_state=identity,old_pose_and_flow_frozen=True,original_eval_masks_unchanged=True,
                metrics={k:reduce(v) for k,v in rows.items()},goal_complete=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V43 matched direct correspondence supervision','','|Arm/step|Input|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---|---:|---:|---:|---:|']
    for tag in ('assisted_0','assisted_100','direct_100'):
        for setting in ('corrupted','clean'):
            m=result['metrics'][tag+'_'+setting];r,s=m['real'],m['proxy']
            lines.append(f"|{tag}|{setting}|{r['canonical_xyz_mm']:.3f}|{r['depth_mm']:.3f}|{s['canonical_xyz_mm']:.3f}|{s['depth_mm']:.3f}|")
    lines+=['','Same inference and original labels; only coordinate-prior term in correspondence CE differs. No pose/native/history claim. Frozen learned-only intervention remains necessary to assess the intended matching mechanism.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

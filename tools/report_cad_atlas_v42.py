"""Compare full-CAD hard retrieval with source/V41 on unchanged fixed probes."""
import argparse,json
from pathlib import Path
import torch
from report_visible_correspondence import exact
from report_geometry_supervision_v41 import read,reduce


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--aborted',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.root/'status.json').read_text())['completed']
    first=torch.load(a.root/'runs/seed42/initial.pt',map_location='cpu',weights_only=False)
    failed=torch.load(a.aborted/'runs/seed42/initial.pt',map_location='cpu',weights_only=False)
    identity={k:exact(first[k],failed[k]) for k in ('model','optimizer','scheduler','rng','sampler_position')};assert all(identity.values())
    del failed
    parent=torch.load(a.reference/'runs/seed42/initial.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(first['model'][n],value) for n,value in parent['model'].items())
    extra=set(first['model'])-set(parent['model']);assert extra and all(n.startswith('cad_atlas_decoder.') for n in extra)
    del parent
    terminal=torch.load(a.root/'runs/seed42/last.pt',map_location='cpu',weights_only=False);assert terminal['step']==100
    frozen=[n for n in terminal['model'] if n=='query' or n.startswith(('head.','object_attn.','object_norm.','geometry_readout.','core.feature_','cad_transport.'))]
    assert all(torch.equal(first['model'][n],terminal['model'][n]) for n in frozen)
    assert json.loads((a.root/'runs/seed42/resume2.json').read_text())['complete_state_verified']
    del first,terminal
    data={}
    for arm,root,steps in [('source',a.reference,(0,)),('v41',a.reference,(100,)),('v42',a.root,(0,100))]:
        for step in steps:
            for clean in (False,True):data[f'{arm}_{step}_{"clean" if clean else "corrupted"}']=read(root,step,clean)
    ref=data['source_0_corrupted']
    for rows in data.values():
        assert rows.keys()==ref.keys()
        for seed,row in rows.items():
            assert all(row[k]==ref[seed][k] for k in ('stream','window','heavy','natural'))
            for kind in ('real','proxy'):
                x,y=row['metrics'][kind],ref[seed]['metrics'][kind]
                assert (x is None)==(y is None)
                if x is not None:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    result=dict(completed=True,updates=100,identical_restart=identity,shared_parent_tensors_exact=True,new_tensors=len(extra),
                frozen_pose_feature_flow_tensors=len(frozen),original_probe_masks_unchanged=True,metrics={k:reduce(v) for k,v in data.items()},
                history=False,pose_loss=False,normal_loss=False,goal_complete=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V42 complete CAD hard-point decoder','','|Model|Input|Real CAD XYZ mm|Real sensor XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---|---:|---:|---:|---:|---:|']
    for tag in ('source_0','v41_100','v42_0','v42_100'):
        for setting in ('corrupted','clean'):
            d=result['metrics'][tag+'_'+setting];r,s=d['real'],d['proxy']
            lines.append(f"|{tag}|{setting}|{r['canonical_xyz_mm']:.3f}|{r['xyz_mm']:.3f}|{r['depth_mm']:.3f}|{s['canonical_xyz_mm']:.3f}|{s['depth_mm']:.3f}|")
    lines+=['','All64 records and original evaluation masks match. Same parent tensors; V42 adds a random full-CAD query/key head. Interrupted normal-loss attempt preserved separately; stable restart has identical initial model/optimizer/RNG and temporarily disables finite-difference normals. Empty-CAD fallback is DPT; no FP/pose shortcut. Hard membership in a CAD point cloud is not proof of correct correspondence.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

"""Matched comparison against V40, retaining every original probe label."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from report_visible_correspondence import exact


def read(root,step,clean=False):
    rows={}
    for rank in range(8):
        p=root/'probe'/f"step{step}{'_clean' if clean else ''}"/f'rank{rank}'
        assert json.loads((p/'receipt.json').read_text())['completed']
        for line in (p/'frames.jsonl').read_text().splitlines():
            r=json.loads(line);assert r['seed'] not in rows;rows[r['seed']]=r
    assert len(rows)==64
    return rows


def reduce(rows):
    groups={}
    for r in rows.values():
        if not r['heavy']:continue
        seq='/'.join(r['stream'].split('|')[0].split('/')[:2])
        for kind in ('real','proxy'):
            value=r['metrics'][kind]
            if value is None:continue
            for metric in ('xyz_mm','canonical_xyz_mm','depth_mm','cad_flow_epe','cad_zero_flow_epe','cad_lookup_coverage'):
                v=value.get(metric)
                if v is not None:groups.setdefault(kind,{}).setdefault(metric,{}).setdefault(seq,[]).append(v)
    return {k:{m:statistics.mean(statistics.mean(v) for v in sequences.values()) for m,sequences in metrics.items()} for k,metrics in groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.root/'status.json').read_text())['completed']
    starts=[torch.load(r/'runs/seed42/initial.pt',map_location='cpu',weights_only=False) for r in (a.reference,a.root)]
    identity={k:exact(starts[0][k],starts[1][k]) for k in ('model','optimizer','scheduler','rng','sampler_position')}
    assert all(identity.values()),identity
    end=torch.load(a.root/'runs/seed42/last.pt',map_location='cpu',weights_only=False);assert end['step']==100
    names=[n for n in end['model'] if n=='query' or n.startswith(('head.','object_attn.','object_norm.','geometry_readout.','core.feature_'))]
    assert all(torch.equal(end['model'][n],starts[1]['model'][n]) for n in names)
    assert json.loads((a.root/'runs/seed42/resume2.json').read_text())['complete_state_verified']
    del starts,end
    rows={}
    for arm,root in [('v40',a.reference),('v41',a.root)]:
        for step in (0,100):
            for clean in (False,True):rows[f'{arm}_{step}_{"clean" if clean else "corrupted"}']=read(root,step,clean)
    ref=rows['v40_0_corrupted']
    for key,items in rows.items():
        assert items.keys()==ref.keys()
        for seed,row in items.items():
            r=ref[seed]
            assert all(row[k]==r[k] for k in ('stream','heavy','natural','window'))
            for kind in ('real','proxy'):
                x,y=row['metrics'][kind],r['metrics'][kind];assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    result=dict(completed=True,updates=100,initial_state_exact=identity,strict_resume_verified=True,
                frozen_pose_feature_tensors=len(names),original_eval_masks_unchanged=True,metrics={k:reduce(v) for k,v in rows.items()},
                reduction='Heavy eligible records averaged within physical sequence, equal sequence mass',goal_complete=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V41 audited supervision — matched100 updates','','|Model|Input|Real CAD XYZ mm|Real sensor XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---|---:|---:|---:|---:|---:|']
    for model in ('v41_0','v40_100','v41_100'):
        for setting in ('corrupted','clean'):
            d=result['metrics'][model+'_'+setting];r,s=d['real'],d['proxy']
            lines.append(f"|{model}|{setting}|{r['canonical_xyz_mm']:.3f}|{r['xyz_mm']:.3f}|{r['depth_mm']:.3f}|{s['canonical_xyz_mm']:.3f}|{s['depth_mm']:.3f}|")
    lines+=['','Original evaluation masks retained; no evaluation filtering. Same initialization, optimizer, scheduler, eight rank RNG states, sampler, observations and100-update budget. Changed training supervision quality and FP32 geometry operations. Clean input is a privileged availability diagnostic, not heavy-occlusion deployment.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

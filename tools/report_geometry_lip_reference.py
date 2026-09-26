"""Input-availability diagnostic, not a native LIP/JEPA ranking."""
import argparse,json,statistics
from pathlib import Path


def aggregate(rows,fn):
    values={}
    for row in rows:
        d=fn(row)
        if d is None:continue
        physical='/'.join(row['stream'].split('|')[0].split('/')[:2])
        for k,v in d.items():
            if isinstance(v,(int,float)):values.setdefault(k,{}).setdefault(physical,[]).append(v)
    return {k:statistics.mean(statistics.mean(v) for v in groups.values()) for k,groups in values.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    tables={};records={}
    for mode,folder in [('corrupted',root),('clean_control',root/'clean')]:
        data=[]
        for rank in range(8):
            d=folder/f'rank{rank}';receipt=json.loads((d/'receipt.json').read_text())
            assert receipt['completed'] and receipt['lip_baseline_sha256']=='89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868'
            data += [json.loads(s) for s in (d/'frames.jsonl').read_text().splitlines()]
        records[mode]={r['seed']:r for r in data};tables[mode]={}
        for group in ('all','heavy','natural'):
            rows=[r for r in data if group=='all' or r[group]]
            tables[mode][group]=dict(lip_pose=aggregate(rows,lambda r:r['lip_pose']),geometry={tag:{kind:aggregate(rows,lambda r:r['geometry_baselines'][tag][kind]) for kind in ('real','proxy')} for tag in ('base_cad','lip_cad','jepa')})
    assert records['corrupted'].keys()==records['clean_control'].keys()
    for seed,row in records['corrupted'].items():
        other=records['clean_control'][seed]
        assert all(row[k]==other[k] for k in ('stream','heavy','natural','window'))
        assert row['geometry_baselines']['base_cad']==other['geometry_baselines']['base_cad']
    result=dict(completed=True,paired_records=len(records['corrupted']),paired_base_render_and_targets_verified=True,
                reduction='Equal physical-sequence mass; heavy records within sequence',metrics=tables,training=False,production_model_changed=False,
                scope='Same controlled base/crop/current RGB-D; both models history off. Clean control removes only artificial input occlusion and retains target masks. Not native accuracy or proof of information-theoretic impossibility.')
    (root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V39 clean-input availability versus corrupted input','','|Input|LIP rotation after10deg|LIP real surface XYZ mm*|JEPA real surface XYZ mm*|LIP proxy XYZ mm*|JEPA proxy XYZ mm*|','|---|---:|---:|---:|---:|---:|']
    for mode,t in tables.items():
        x=t['heavy'];g=x['geometry']
        lines.append(f"|{mode}|{x['lip_pose']['rotation_after_deg']:.3f}|{g['lip_cad']['real']['covered_canonical_xyz_mm']:.3f}|{g['jepa']['real']['covered_canonical_xyz_mm']:.3f}|{g['lip_cad']['proxy']['covered_canonical_xyz_mm']:.3f}|{g['jepa']['proxy']['covered_canonical_xyz_mm']:.3f}|")
    lines+=['','*Conditional covered-region errors have DIFFERENT coverage across methods. They are not a whole-region ranking. `outcome.json` also reports coverage and the fraction of ALL target pixels within10mm XYZ AND5mm depth, counting missing outputs as failures.',
            'Clean current RGB-D is a privileged input-availability control; its results must not be called heavy-occlusion deployment accuracy.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

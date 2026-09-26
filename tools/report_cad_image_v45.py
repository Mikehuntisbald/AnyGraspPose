"""Equal physical-sequence means; keep controlled probes separate from native."""
import argparse,json,statistics
from pathlib import Path


def read(path):
    rows={}
    for p in sorted(path.glob('rank*/frames.jsonl')):
        assert json.loads((p.parent/'receipt.json').read_text())['completed']
        for line in p.read_text().splitlines():
            r=json.loads(line);assert r['seed'] not in rows;rows[r['seed']]=r
    assert len(rows)==64
    return rows


def flatten(x,prefix=''):
    result={}
    if isinstance(x,dict):
        for key,v in x.items():result.update(flatten(v,prefix+'/'+key))
    elif isinstance(x,(int,float,bool)):result[prefix]=float(x)
    return result


def reduce(rows,heavy=True):
    groups={}
    for r in rows.values():
        if heavy and not r['heavy']:continue
        sequence='/'.join(r['stream'].split('|')[0].split('/')[:2])
        for key,value in flatten(dict(geometry=r['metrics'],correspondence=r['correspondence'])).items():
            groups.setdefault(key,{}).setdefault(sequence,[]).append(value)
    return {k:statistics.mean(statistics.mean(v) for v in sequences.values()) for k,sequences in groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--steps',type=int,default=100);a=p.parse_args()
    data={str(step):read(a.root/'probe'/f'step{step}') for step in (0,a.steps)}
    if (a.root/'frozen/status.json').exists() and json.loads((a.root/'frozen/status.json').read_text()).get('completed'):
        data.update({arm:read(a.root/'frozen'/arm) for arm in ('source','initial','trained')})
    reference=data['0']
    for rows in data.values():
        assert rows.keys()==reference.keys()
        for seed,r in rows.items():
            other=reference[seed]
            assert all(r[k]==other[k] for k in ('stream','heavy','natural','window'))
            for name in ('real','proxy'):
                x,y=r['metrics'][name],other['metrics'][name]
                assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    result=dict(completed=True,records=64,reduction='frames within physical sequence, then equal sequence means',
                scope='training-partition physical holdout, 10-degree rotation-only initial error; not native pose evaluation',
                original_eval_masks_unchanged=True,heavy={k:reduce(v) for k,v in data.items()},all={k:reduce(v,False) for k,v in data.items()},default_model_changed=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2))
    lines=['# Explicit CAD-to-image outcome: '+a.root.name,'','All numbers below use the heavy subset and equal physical-sequence means. Controlled64 probes, not native tracking. Original target masks are unchanged.','',
           '|Checkpoint|Real CAD XYZ mm|Real depth mm|Proxy XYZ mm|Proxy depth mm|','|---|---:|---:|---:|---:|']
    for name in data:
        m=result['heavy'][name];keys=['/geometry/real/canonical_xyz_mm','/geometry/real/depth_mm','/geometry/proxy/canonical_xyz_mm','/geometry/proxy/depth_mm']
        lines.append('|'+name+'|'+'|'.join(f'{m[k]:.3f}' for k in keys)+'|')
    lines+=['','|Checkpoint/readout|Rotation deg|Translation mm|ADD-S mm|ADD-S<0.05d %|Accepted %|','|---|---:|---:|---:|---:|---:|']
    for name,m in result['heavy'].items():
        methods=sorted({k.split('/')[3] for k in m if k.startswith('/correspondence/pose/')})
        for method in methods:
            prefix='/correspondence/pose/'+method
            values=[m[prefix+'/'+k] for k in ('rotation_deg','translation_mm','adds_mm','adds_005d')];values[-1]*=100
            accept=m.get('/correspondence/solvers/'+method+'/accepted')
            lines.append('|'+name+'/'+method+'|'+'|'.join(f'{v:.3f}' for v in values)+'|'+(f'{accept*100:.1f}' if accept is not None else '—')+'|')
    lines+=['','Poses rejected by prediction-only checks retain the base pose and remain in metrics. Base translation is exactly correct in this protocol; ADD-S success is already saturated. Endpoint, continuous pose error, acceptance, and wider-initialization/native tests are needed.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

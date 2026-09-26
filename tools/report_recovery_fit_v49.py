"""Separate fixed-training-set fit from independent reconstruction accuracy."""
import argparse,json,statistics
from pathlib import Path
from report_cad_image_v45 import read


def geometry(rows):
    groups={}
    for row in rows.values():
        if not row['heavy']:continue
        sequence='/'.join(row['stream'].split('|')[0].split('/')[:2])
        for region in ('real','proxy'):
            value=row['metrics'][region]
            if value is None:continue
            for metric in ('canonical_xyz_mm','depth_mm'):
                groups.setdefault(region+'/'+metric,{}).setdefault(sequence,[]).append(value[metric])
    return {key:statistics.mean(statistics.mean(v) for v in sequences.values()) for key,sequences in groups.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run=a.root/'runs/seed42'
    assert json.loads((a.root/'status.json').read_text())['completed']
    logs=[]
    for rank in range(8):
        rows=[json.loads(line) for line in (run/f'rank{rank}.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,201));logs.append(rows)
        before=json.loads((run/f'fixed_batch_rank{rank}_start0.json').read_text())
        after=json.loads((run/f'fixed_batch_rank{rank}_start2.json').read_text())
        assert before==after
    assert json.loads((run/'resume2.json').read_text())['complete_state_verified']
    keys=['real_cad_xyz_mm','real_depth_mm','proxy_cad_xyz_mm','proxy_depth_mm','atlas_real_top8','atlas_proxy_top8']
    fit={name:{key:statistics.mean(rows[index]['metrics'][key] for rows in logs) for key in keys} for name,index in [('first_forward',0),('last_forward',-1)]}
    probes={str(step):read(a.root/'probe'/f'step{step}') for step in (0,200)}
    assert probes['0'].keys()==probes['200'].keys()
    for seed,row in probes['0'].items():
        other=probes['200'][seed]
        assert all(row[k]==other[k] for k in ('stream','heavy','natural','window'))
        assert 'correspondence' not in row and 'correspondence' not in other
        for name in ('real','proxy'):
            x,y=row['metrics'][name],other['metrics'][name];assert (x is None)==(y is None)
            if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    result=dict(completed=True,updates=200,training_observations=32,paired_hypotheses=64,input_and_target_hashes_match_across_resume=True,
        train_fit=fit,independent_heavy={k:geometry(v) for k,v in probes.items()},pose_training=False,pose_evaluation=False,
        scope='Training metrics are equal-rank means on fixed audited masks; independent64 probe uses original masks and equal physical-sequence means. Do not compare absolute levels across the two populations.',
        default_model_changed=False,accurate_recovery_achieved=False)
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2))
    lines=['# JEPA-only fixed32 reconstruction fit','','No pose, PnP, flow-readout training or pose evaluation.200 geometry updates;32 fixed training observations/64 hypotheses. Every step checked identical inputs/targets. Strict2→200 resume verified.','',
        '|Fixed TRAINING set|CAD identity XYZ real mm|Real sensor depth mm|CAD identity XYZ proxy mm|Proxy depth mm|','|---|---:|---:|---:|---:|']
    for name,m in fit.items():lines.append('|'+name+'|'+'|'.join(f'{m[k]:.3f}' for k in keys[:4])+'|')
    lines+=['','These are logged pre-update forward metrics (first versus last training step), equal-rank means. They establish fitting ability, not held-out accuracy.','',
        '|Independent HEAVY probe|CAD identity XYZ real mm|Real sensor depth mm|CAD identity XYZ proxy mm|Proxy depth mm|','|---|---:|---:|---:|---:|']
    for name,m in result['independent_heavy'].items():lines.append('|'+name+'|'+'|'.join(f"{m[k]:.3f}" for k in ('real/canonical_xyz_mm','real/depth_mm','proxy/canonical_xyz_mm','proxy/depth_mm'))+'|')
    lines+=['','Object-coordinate XYZ measures CAD surface identity correspondence. Camera-space recovered geometry must be assessed separately through depth and calibrated camera rays. Real-depth supervision remains original sensor depth; naturally hidden proxy remains GT CAD. No learned decoder was promoted from this memorization diagnostic.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

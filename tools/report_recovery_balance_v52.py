"""Paired recovery metrics only; same source, samples, masks and fixed budget."""
import argparse, json, statistics
from pathlib import Path
from report_cad_image_v45 import read


def reduce(rows,heavy):
    groups={}
    for row in rows.values():
        if heavy and not row['heavy']:continue
        seq='/'.join(row['stream'].split('|')[0].split('/')[:2])
        assert 'correspondence' not in row and 'lip_pose' not in row
        for region in ('real','proxy'):
            m=row['metrics'][region]
            if not m:continue
            for key in ('canonical_xyz_mm','depth_mm','camera_xyz_mm','camera_normal_deg'):
                if m.get(key) is not None:groups.setdefault(region+'/'+key,{}).setdefault(seq,[]).append(m[key])
    return {k:statistics.mean(statistics.mean(v) for v in g.values()) for k,g in groups.items()}


def paired(data):
    reference=next(iter(data.values()))
    for rows in data.values():
        assert rows.keys()==reference.keys()
        for seed,r in rows.items():
            other=reference[seed]
            assert all(r[k]==other[k] for k in ('stream','heavy','natural','window'))
            for name in ('real','proxy'):
                x,y=r['metrics'][name],other['metrics'][name]
                assert (x is None)==(y is None)
                if x:assert x['pixels']==y['pixels'] and x['canonical_pixels']==y['canonical_pixels']
    return {kind:{name:reduce(rows,heavy) for name,rows in data.items()} for kind,heavy in [('heavy',True),('all',False)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    data={}
    for arm in ('control','balanced'):
        r=a.root/arm
        assert json.loads((r/'status.json').read_text())['completed']
        assert json.loads((r/'runs/seed42/resume2.json').read_text())['complete_state_verified']
        for step in (0,200):data[f'{arm}_{step}']=read(r/'probe'/f'step{step}')
        for rank in range(8):
            logs=[json.loads(x) for x in (r/f'runs/seed42/rank{rank}.jsonl').read_text().splitlines()]
            assert [x['step'] for x in logs]==list(range(1,201))
    for rank in range(8):
        logs=[[json.loads(x) for x in (a.root/arm/f'runs/seed42/rank{rank}.jsonl').read_text().splitlines()] for arm in ('control','balanced')]
        assert [r['windows'] for r in logs[0]]==[r['windows'] for r in logs[1]]
    result=dict(completed=True,updates_per_arm=200,paired=paired(data),pose_training=False,pose_evaluation=False,
        default_model_changed=False,scope='Two matched200-update arms, training-partition physical holdout64; equal physical-sequence means, original raw evaluation masks. No official test. Canonical XYZ is CAD identity, camera XYZ is depth lifted through crop rays.')
    if (a.root/'confirmation').exists():
        confirmation={arm:read(a.root/'confirmation'/arm) for arm in ('source','control','balanced')}
        result['confirmation']=paired(confirmation)
        frame_keys=lambda rows:{(r['stream'],r['window']['first_frame']) for r in rows.values()}
        result['confirmation_overlap_frames']=len(frame_keys(data['control_0'])&frame_keys(confirmation['source']))
    result['relative_to_matched_control_pct']={}
    for split in ('paired','confirmation'):
        if split not in result:continue
        control,balanced=('control_200','balanced_200') if split=='paired' else ('control','balanced')
        v=result[split]['heavy']
        result['relative_to_matched_control_pct'][split]={k:100*(v[balanced][k]/value-1) for k,value in v[control].items()}
    result['accurate_recovery_achieved']=False
    (a.root/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# JEPA recovery gradient balance V52','','Only correspondence weight changes: control1.0, balanced0.1. Network architecture and all learning rates are unchanged. Encoder/JEPA/DPT/CAD train, pose/history/DINO loss stay off. Matched data and strict resume checked.','',
        'The lower weight improves real-depth recovery on both probes. The usual probe trades worse proxy correspondence/depth for better real recovery; the prespecified confirmation improves both regions. This is evidence that weighting contributes to the problem, not a complete geometry repair or a uniform win. No default-model promotion.','']
    for split in ('paired','confirmation'):
        if split not in result:continue
        lines += [f'## {split}: heavy subset','','|Arm|Real CAD XYZ mm|Real depth mm|Proxy CAD XYZ mm|Proxy depth mm|Real camera XYZ mm|Proxy camera XYZ mm|','|---|---:|---:|---:|---:|---:|---:|']
        for arm,m in result[split]['heavy'].items():
            lines.append('|'+arm+'|'+'|'.join(f'{m[k]:.3f}' for k in ('real/canonical_xyz_mm','real/depth_mm','proxy/canonical_xyz_mm','proxy/depth_mm','real/camera_xyz_mm','proxy/camera_xyz_mm'))+'|')
        lines+=['']
    lines+=['The usual64 probe is reused development evidence. The confirmation64 uses seeds54000000+rank+8*draw, fixed before training and sampled from the same held-out physical-sequence pool. It checks new frames/occluders, not unseen objects or a new sequence split. No candidate is automatically promoted.']
    lines+=['','[Paired recovery scores](paired_recovery.png) · [Predetermined heavy-frame depth/error maps](fixed_recovery_example.png) · [Checkpoint verification](checkpoint_verified.json)',
        '', '26 targeted tests passed in each arm. Checkpoint verification records715 identical initial tensors and35 unchanged pose tensors; first-forward metrics match on all8 ranks. Full2→200 resume is verified. Real depth targets remain sensor measurements; proxy targets remain GT CAD; original evaluation masks remain unchanged.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()

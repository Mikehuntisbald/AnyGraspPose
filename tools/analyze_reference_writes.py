"""Descriptive write fractions and conditional initial-center mixing weights."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def trace_stream(rows,limit):
    if not math.isfinite(limit) or not 0<limit<=1:raise ValueError('Invalid write limit')
    rows=sorted(rows,key=lambda r:r['frame_index'])
    if not rows or not rows[0]['initialization'] or any(r['initialization'] for r in rows[1:]):raise ValueError('Exactly one leading initializer required')
    if len({r['frame_index'] for r in rows})!=len(rows):raise ValueError('Duplicate frame')
    weight=1.;output=[]
    for position,row in enumerate(rows[1:],1):
        before=weight;rotation=center=None
        if row['status']=='ok':
            rotation=float(row['reference_write_rotation_coefficient']);center=float(row['reference_write_center_coefficient'])
            if not all(math.isfinite(v) and 0<=v<=limit+1e-7 for v in (rotation,center)):raise ValueError('Invalid write fraction')
            if not row['reference_write_valid_target'] and (rotation!=0 or center!=0):raise ValueError('Invalid target was written')
            weight*=1-center
        elif 'reference_write_center_coefficient' in row:raise ValueError('Uncommitted write telemetry')
        output.append(dict(stream_id=row['stream_id'],frame_index=row['frame_index'],position=position,object_id=row['object_id'],
            initial_good=bool(rows[0]['adds_005']),visibility=row['visibility'],status=row['status'],
            rotation_write=rotation,center_write=center,initial_center_weight_before=before,initial_center_weight_after=weight))
    return output


def summarize(rows,limit):
    valid=[r for r in rows if r['center_write'] is not None]
    result=dict(frames=len(rows),usable=len(valid),objects=len({r['object_id'] for r in rows}),usable_objects=len({r['object_id'] for r in valid}),streams=len({r['stream_id'] for r in rows}))
    for key in ('rotation_write','center_write','initial_center_weight_before'):
        groups=defaultdict(list)
        for row in valid:groups[row['object_id']].append(row[key])
        values=[r[key] for r in valid]
        result[key]=dict(object_macro_mean=statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None,
            frame_median=statistics.median(values) if values else None)
        if key.endswith('_write'):
            result[key].update(zero_fraction=sum(v==0 for v in values)/len(values) if values else None,
                saturated_fraction=sum(v>=limit-1e-7 for v in values)/len(values) if values else None)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);m=json.loads((a.evaluation/'manifest.json').read_text())
    assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
    assert m['architecture_id']=='stream_rk_adaptive_reference' and m['split']=='val' and m['fp_calls']==m['critic_calls']==0
    raw=(a.evaluation/'predictions.jsonl').read_bytes();streams=defaultdict(list)
    for row in map(json.loads,raw.splitlines()):streams[row['stream_id']].append(row)
    assert len(streams)==320;limit=m['config']['reference_write_limit']
    traced=[r for sid,rows in sorted(streams.items()) for r in trace_stream(rows,limit)];assert len(traced)==22880
    masks={'all':lambda r:True,'first8':lambda r:r['position']<=8,
        'first8_initial_good':lambda r:r['position']<=8 and r['initial_good'],
        'first8_initial_bad':lambda r:r['position']<=8 and not r['initial_good'],
        'visibility_ge05':lambda r:r['visibility'] is not None and r['visibility']>=.5,
        'visibility_lt05':lambda r:r['visibility'] is not None and r['visibility']<.5,
        'visibility_lt03':lambda r:r['visibility'] is not None and r['visibility']<.3}
    populations={name:summarize([r for r in traced if select(r)],limit) for name,select in masks.items()}
    prefixes={str(n):summarize([r for r in traced if r['position']==n],limit) for n in (1,8,16,32,64)}
    path=a.out/'write_trace.csv'
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(traced[0]));writer.writeheader();writer.writerows(traced)
    report=dict(completed=True,checkpoint_sha256=m['checkpoint_sha256'],source_sha256=m['source_sha256'],
        prediction_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        write_limit=limit,scope='Posthoc descriptive diagnostics. Fractions are not calibrated probabilities. The product of (1-center_write) is the explicit initial-center coefficient in the real-arithmetic recurrence conditional on realized gates and proposals. It ignores floating-point rounding and indirect dependence through proposals, readout, gates, and visual history; it is not total initial-pose sensitivity or a rotation mixing coefficient. GT groups are unavailable as deployment switches; groups have different object sets.',
        populations=populations,prefixes=prefixes,csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (a.out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(completed=True,populations=populations),indent=2))


if __name__=='__main__':main()

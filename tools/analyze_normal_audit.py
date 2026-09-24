"""Aggregate frozen normal audit with equal frame/case/sequence mass and counts."""
import argparse,json,hashlib,statistics
from pathlib import Path
from collections import defaultdict


def metrics(c):
    e=c['eligible'];v=c['valid_unit']
    if not e:return {}
    result={k+'_pct':100*c[k]/e for k in ['degenerate','nonfinite','zero_tangent','small_tangent','near_collinear','legacy_attenuated']}
    result['legacy_angle_deg']=c['legacy_angle_sum']/e
    if v:
        result.update(unit_angle_deg=c['unit_angle_sum']/v,unoriented_angle_deg=c['unoriented_angle_sum']/v,
            negative_dot_pct=100*c['negative_dot']/v,near_opposite_pct=100*c['near_opposite']/v,near_aligned_pct=100*c['near_aligned']/v)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--world',type=int,default=8);a=p.parse_args()
    pooled=defaultdict(lambda:defaultdict(float));frames=defaultdict(lambda:defaultdict(list))
    sequences=set();identity=None;total_rows=0
    for rank in range(a.world):
        folder=a.run/f'rank{rank}';m=json.loads((folder/'manifest.json').read_text())
        assert m['completed'] and m['normal_audit'] and not m['smoke']
        if identity is None:identity={k:m[k] for k in ['checkpoint_sha256','reference_sha256','normal_contract','expected_physical_sequences']}
        assert all(m[k]==v for k,v in identity.items())
        assert not sequences.intersection(m['physical_sequences']);sequences.update(m['physical_sequences'])
        with (folder/'frames.jsonl').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['frames_sha256']
        with (folder/'frames.jsonl').open() as f:
            for line in f:
                row=json.loads(line);total_rows+=1
                assert row['history']=='off'
                if row['phase']!='occlusion':continue
                for source,data in row['normal_audit'].items():
                    for spatial in ['all','within_patch','cross_patch']:
                        combined={k:data['s1_'+spatial][k]+data['s2_'+spatial][k] for k in data['s1_'+spatial]}
                        groups=dict(data);groups['both_'+spatial]=combined
                        for stencil in ['s1_'+spatial,'s2_'+spatial,'both_'+spatial]:
                            counts=groups[stencil];key=(row['case'],source,stencil)
                            for k,v in counts.items():pooled[key][k]+=v
                            for k,v in metrics(counts).items():frames[(row['physical_sequence'],*key)][k].append(v)
    assert len(sequences)==40 and sequences==set(identity['expected_physical_sequences']) and total_rows==8000
    seq={key:{k:statistics.mean(v) for k,v in values.items()} for key,values in frames.items()}
    summary={}
    for case in ['natural','light','heavy8','heavy16','heavy32','heavy_pooled']:
        summary[case]={}
        cases=['heavy8','heavy16','heavy32'] if case=='heavy_pooled' else [case]
        for source in ['real','proxy']:
            summary[case][source]={}
            for spatial in ['all','within_patch','cross_patch']:
                for scale in ['s1','s2','both']:
                    stencil=scale+'_'+spatial;counts=defaultdict(float);by_sequence=defaultdict(lambda:defaultdict(list))
                    for kind in cases:
                        for k,v in pooled[(kind,source,stencil)].items():counts[k]+=v
                    for (sequence,kind,src,st),values in seq.items():
                        if kind in cases and src==source and st==stencil:
                            for k,v in values.items():by_sequence[sequence][k].append(v)
                    values=defaultdict(list)
                    for s,ms in by_sequence.items():
                        for k,v in ms.items():values[k].append(statistics.mean(v))
                    summary[case][source][stencil]=dict(counts=dict(counts),pooled_metrics=metrics(counts) if counts else {},
                        sequence_mean={k:statistics.mean(v) for k,v in values.items()},metric_sequences={k:len(v) for k,v in values.items()})
    result=dict(completed=True,optimizer_updates=0,physical_sequences=40,rows=total_rows,identity=identity,tables=summary,
        reduction='frame means within case/sequence, heavy cases averaged within sequence, equal sequence mass; pooled counts also provided',
        validity='target eligibility unchanged; report prediction degeneracy separately, never hide it in angle denominator',
        pure_sign_warning='negative dot means opposite hemisphere, not proof of an otherwise correct flipped surface')
    (a.run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Normal degeneracy and orientation audit','',result['reduction'],'',
        '| Case | Source | Degenerate % | Unit angle deg | Negative dot % | Unoriented angle deg | Near opposite % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case in ['natural','light','heavy_pooled']:
        for source in ['real','proxy']:
            m=summary[case][source]['both_all']['sequence_mean']
            if not m:continue
            names=['degenerate_pct','unit_angle_deg','negative_dot_pct','unoriented_angle_deg','near_opposite_pct']
            lines.append('| '+case+' | '+source+' | '+' | '.join(f'{m[k]:.5f}' if k in m else 'N/A' for k in names)+' |')
    (a.run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=True,rows=total_rows,checkpoint_sha256=identity['checkpoint_sha256'])))


if __name__=='__main__':main()

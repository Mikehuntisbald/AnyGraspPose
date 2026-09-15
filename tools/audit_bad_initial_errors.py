"""Separate later threshold crossings from continuous error changes after bad starts."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_pose_robustness import initial_groups,stable_success_position
from compare_rk_ablation import paired_summary


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('reference-eval','candidate-eval','index-root','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);data={};manifests={};hashes={}
    for name,folder in [('reference',a.reference_eval),('candidate',a.candidate_eval)]:
        m=json.loads((folder/'manifest.json').read_text());assert m['completed'] and m['population_verified'] and m['frames']==23200 and m['split']=='val' and m['fp_calls']==0
        raw=(folder/'predictions.jsonl').read_bytes();data[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,raw.splitlines())};hashes[name]=hashlib.sha256(raw).hexdigest();manifests[name]=m
    assert set(data['reference'])==set(data['candidate'])
    assert all(manifests['reference'][k]==manifests['candidate'][k] for k in ('initial_poses_sha256','mesh_hash','split_hash'))
    audit=json.loads((a.index_root/'audit.json').read_text());assert all(audit[k]==manifests['reference'][k] for k in ('split_hash','mesh_hash'))
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    groups=defaultdict(list)
    for row in data['reference'].values():groups[row['stream_id']].append(row)
    initial=initial_groups(list(data['reference'].values()));records=[];changes=[]
    for sid,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['frame_index'])
        assert rows[0]['pose_centered']==data['candidate'][(sid,rows[0]['frame_index'])]['pose_centered']
        if initial[sid]['good']:continue
        candidate=[data['candidate'][(sid,r['frame_index'])] for r in rows]
        recovered=[stable_success_position(rows),stable_success_position(candidate)]
        both=max(recovered) if all(v is not None for v in recovered) else None
        d=float(streams[sid]['mesh_diameter']);assert d>0
        for position,(r,c) in enumerate(zip(rows[1:],candidate[1:]),1):
            record=dict(stream_id=sid,frame=r['frame_index'],position=position,object_id=r['object_id'],physical_sequence='/'.join(sid.split('/')[:2]),
                both_previously_stable=both is not None and position>both,visibility=r['visibility'],
                reference_adds005=bool(r['adds_005']),candidate_adds005=bool(c['adds_005']))
            for name,x in [('reference',r),('candidate',c)]:
                record[name]=dict(adds_percent_d=x['adds_m']/d*100,add_percent_d=x['add_m']/d*100,center_mm=x['center_mm'],center_percent_d=x['center_mm']/10/d,rotation_deg=x['rotation_deg'])
            records.append(record)
            if record['reference_adds005']!=record['candidate_adds005']:changes.append(record)
    all_sequences=sorted({'/'.join(s.split('/')[:2]) for s in groups});weights=np.random.default_rng(20260914).multinomial(len(all_sequences),np.full(len(all_sequences),1/len(all_sequences)),size=2000)
    populations={'all_bad_initial':records,'first8':[r for r in records if r['position']<=8],'after8':[r for r in records if r['position']>8],
        'after_both_stable':[r for r in records if r['both_previously_stable']]};summary={}
    for name,rs in populations.items():
        obj=np.array([r['object_id'] for r in rs]);seq=np.array([r['physical_sequence'] for r in rs]);summary[name]=dict(frames=len(rs),objects=len(set(obj.tolist())),metrics={})
        if not rs:continue
        w=weights[:,np.isin(all_sequences,np.unique(seq))]
        for metric in records[0]['reference']:
            values=np.array([[r[arm][metric] for arm in ('reference','candidate')] for r in rs]);point,boot=paired_summary(values,obj,seq,w)
            delta=boot[:,1]-boot[:,0];finite=delta[np.isfinite(delta)]
            summary[name]['metrics'][metric]=dict(reference=float(point[0]),candidate=float(point[1]),delta=float(point[1]-point[0]),ci95=np.quantile(finite,[.025,.975]).tolist())
    counts=Counter(('regression' if r['reference_adds005'] else 'improvement','first8' if r['position']<=8 else 'later') for r in changes)
    margins={kind:[r for r in changes if r['reference_adds005']==(kind=='regression')] for kind in ('regression','improvement')}
    for kind,rs in margins.items():
        margins[kind]=dict(frames=len(rs),after_both_stable=sum(r['both_previously_stable'] for r in rs),
            median_reference_adds_percent_d=float(np.median([r['reference']['adds_percent_d'] for r in rs])) if rs else None,
            median_candidate_adds_percent_d=float(np.median([r['candidate']['adds_percent_d'] for r in rs])) if rs else None)
    interventions=[m.get('inference_intervention',{}) for m in manifests.values()]
    output_intervention=(manifests['reference']['checkpoint_sha256']==manifests['candidate']['checkpoint_sha256'] and
        {i.get('mode') for i in interventions}=={'learned','zero'} and
        all(i.get('completed') and i.get('weights_bitwise_unchanged') and i.get('source_and_intervention_bound') for i in interventions))
    components={i.get('component','feedback') for i in interventions}
    output_intervention=output_intervention and len(components)==1
    component=next(iter(components)) if output_intervention else None
    zero_channels=next((i.get('zero_channels','both') for i in interventions if i.get('mode')=='zero'),None) if output_intervention else None
    description=component if zero_channels in (None,'both') else f'{component} {zero_channels} channel'
    interpretation=(f'Same frozen weights with learned versus zero reference {description} output. The difference includes the downstream causal trajectory after the intervention; it is not a claim about generalization or the latent mechanism.' if output_intervention else
        'These compare paired input streams with separately evolved prediction trajectories, not proof that the reference feedback caused the changes.')
    result=dict(completed=True,scope='Completed controlled-val posthoc diagnostic, shared noisy initializer and zero FP. Positive continuous error delta is worse. Stable recovery is three consecutive usable strict-ADD-S successes. '+interpretation,
        same_weights_output_intervention=output_intervention,
        intervention_component=component,
        intervention_zero_channels=zero_channels,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prediction_sha256=hashes,
        checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},populations=summary,
        threshold_changes={kind:dict(first8=counts[kind,'first8'],later=counts[kind,'later']) for kind in ('regression','improvement')},crossing_margins=margins)
    (a.out/'analysis.json').write_text(json.dumps(result,indent=2));(a.out/'changed_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changes));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Re-score rotation/center hybrids from two completed, frozen trajectories.

Hybrids are posthoc geometric diagnostics, never inferred tracking outputs.
The two replacement orders share the non-additive metric effect equally.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

NAMES = ('on', 'rotation_off', 'center_off', 'off')
CONTRASTS = {
    'off_minus_on': [-1., 0., 0., 1.],
    'rotation_contribution': [-.5, .5, -.5, .5],
    'center_contribution': [-.5, -.5, .5, .5],
}


def component_errors(vertices, target, on, off):
    vertices, target, on, off = [np.asarray(v, dtype='f4') for v in (vertices, target, on, off)]
    truth = vertices @ target[:3, :3].T + target[:3, 3]
    rotation_on = vertices @ on[:3, :3].T
    rotation_off = vertices @ off[:3, :3].T
    clouds = (rotation_on + on[:3, 3], rotation_off + on[:3, 3],
              rotation_on + off[:3, 3], rotation_off + off[:3, 3])
    return np.array([float(np.linalg.norm(cloud-truth, axis=1).mean()) for cloud in clouds])


def contributions(values):
    values = np.asarray(values)
    result = {key: values @ coefficient for key, coefficient in CONTRASTS.items()}
    assert np.allclose(result['rotation_contribution'] + result['center_contribution'],
                       result['off_minus_on'], rtol=0, atol=1e-10)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for key in ('runtime', 'on-eval', 'off-eval', 'index-root', 'out'):
        parser.add_argument('--'+key, required=True, type=Path)
    a = parser.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.runtime/'src'))
    import torch
    from lip.geometry.so3 import center_pose
    from lip.engine.stream_checkpoint import source_hash
    from compare_rk_ablation import paired_summary
    torch.set_num_threads(2)
    manifests={}; data={}; hashes={}
    for name, folder in (('on',a.on_eval), ('off',a.off_eval)):
        m=json.loads((folder/'manifest.json').read_text()); manifests[name]=m
        assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
        assert m['split']=='val' and m['fp_calls']==m['critic_calls']==0
        intervention=m['inference_intervention']
        assert intervention['component']=='writer' and intervention['weights_bitwise_unchanged']
        assert intervention['mode']==('learned' if name=='on' else 'zero')
        raw=(folder/'predictions.jsonl').read_bytes(); hashes[name]=hashlib.sha256(raw).hexdigest()
        rows=list(map(json.loads,raw.splitlines()));data[name]={(r['stream_id'],r['frame_index']):r for r in rows}
        assert len(rows)==len(data[name])==23200
    on=data['on']; off=data['off']; assert on.keys()==off.keys()
    for key in ('checkpoint_sha256','source_sha256','split_hash','mesh_hash','initial_poses_sha256'):
        assert manifests['on'][key]==manifests['off'][key]
    assert source_hash()==manifests['on']['source_sha256']
    assert all(on[k][f]==off[k][f] for k in on for f in ('initialization','object_id','visibility','moving','timestamp'))
    audit=json.loads((a.index_root/'audit.json').read_text())
    assert all(audit[k]==manifests['on'][k] for k in ('split_hash','mesh_hash'))
    streams={s['stream_id']:s for s in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines())}
    groups=defaultdict(list)
    for row in on.values():groups[row['stream_id']].append(row)
    records=[]; mesh_hashes={}; pose_hashes={}; maximum_error=0.; changed_factual_scores=0
    with (a.out/'frames.jsonl').open('w') as writer:
        for number,(sid,rows) in enumerate(sorted(groups.items()),1):
            rows.sort(key=lambda r:r['frame_index']);s=streams[sid];assert s['split']=='val'
            assert rows[0]['initialization'] and not any(r['initialization'] for r in rows[1:])
            assert rows[0]['pose_centered']==off[sid,rows[0]['frame_index']]['pose_centered']
            with np.load(a.index_root/s['mesh_cache']) as z:mesh={k:z[k].copy() for k in z.files}
            with np.load(a.index_root/s['pose_cache']) as z:
                frames=z['frames'].copy();target=center_pose(torch.from_numpy(z['poses'].copy()),torch.from_numpy(mesh['center'])).numpy()
            assert frames.tolist()==[r['frame_index'] for r in rows]
            pose_sha=hashlib.sha256((a.index_root/s['pose_cache']).read_bytes()).hexdigest()
            assert pose_sha==s['pose_cache_sha256'];pose_hashes[sid]=pose_sha
            mesh_hashes[s['mesh_cache']]=hashlib.sha256((a.index_root/s['mesh_cache']).read_bytes()).hexdigest()
            diameter=float(mesh['diameter'])
            for position,(row,truth) in enumerate(zip(rows,target)):
                other=off[sid,row['frame_index']]
                if not position:continue
                assert row['status']==other['status']=='ok', 'This diagnostic requires usable factual endpoints'
                errors=component_errors(mesh['vertices'],truth,row['pose_centered'],other['pose_centered'])
                assert np.isfinite(errors).all()
                success=100*(errors<.1*diameter)
                for index,factual in ((0,row),(3,other)):
                    difference=abs(errors[index]-factual['add_m']);maximum_error=max(maximum_error,difference)
                    assert difference<=1e-7
                    changed_factual_scores+=int(bool(success[index])!=bool(factual['add_01']))
                normalized=errors/diameter*100
                assert not changed_factual_scores, 'Factual score reproduction failed'
                record=dict(stream_id=sid,frame_index=row['frame_index'],position=position,object_id=s['object_id'],
                    object_name=Path(s['mesh_path']).parent.name,physical_sequence='/'.join(sid.split('/')[:2]),
                    visibility=row['visibility'],moving=row['moving'],initial_good=bool(rows[0]['adds_005']),
                    add_percent_d=dict(zip(NAMES,normalized.tolist())),add_01=dict(zip(NAMES,success.tolist())),
                    error_contributions={k:float(v) for k,v in contributions(normalized).items()},
                    success_contributions={k:float(v) for k,v in contributions(success).items()},
                    factual={name:dict(rotation_deg=x['rotation_deg'],center_mm=x['center_mm'],adds_percent_d=x['adds_m']/diameter*100,
                                       adds_005=bool(x['adds_005'])) for name,x in (('on',row),('off',other))},
                    rotation_write=row['reference_write_rotation_coefficient'],center_write=row['reference_write_center_coefficient'])
                writer.write(json.dumps(record)+'\n');records.append(record)
            writer.flush()
            if number%20==0:print(json.dumps(dict(completed_streams=number,tracked_frames=len(records))),flush=True)
    assert len(records)==22880
    objects=np.array([r['object_id'] for r in records]);sequences=np.array([r['physical_sequence'] for r in records])
    clusters=np.unique(sequences);assert len(clusters)==40
    weights=np.random.default_rng(20260914).multinomial(40,np.full(40,1/40),size=2000)
    masks={'all':lambda r:True,'initial_good':lambda r:r['initial_good'],'initial_bad':lambda r:not r['initial_good'],
           'bad_first8':lambda r:not r['initial_good'] and r['position']<=8,
           'bad_after8':lambda r:not r['initial_good'] and r['position']>8,
           'visibility_ge05':lambda r:r['visibility'] is not None and r['visibility']>=.5,
           'visibility_lt05':lambda r:r['visibility'] is not None and r['visibility']<.5,
           'visibility_lt03':lambda r:r['visibility'] is not None and r['visibility']<.3,
           'severe_moving':lambda r:r['visibility'] is not None and r['visibility']<.3 and r['moving']}
    summary={};per_object=[]
    for name,choose in masks.items():
        mask=np.array([choose(r) for r in records]);present=np.unique(objects[mask])
        info=dict(frames=int(mask.sum()),objects=len(present),physical_sequences=len(np.unique(sequences[mask])),metrics={})
        if not mask.any():
            summary[name]=info
            continue
        subweights=weights[:,np.isin(clusters,np.unique(sequences[mask]))]
        for metric in ('add_percent_d','add_01'):
            values=np.array([[r[metric][n] for n in NAMES] for r in records])
            point,draws=paired_summary(values[mask],objects[mask],sequences[mask],subweights)
            effects={}
            for effect,coefficient in CONTRASTS.items():
                sample=draws@coefficient;finite=sample[np.isfinite(sample)]
                effects[effect]=dict(delta=float(point@coefficient),
                    ci95=np.quantile(finite,[.025,.975]).tolist() if info['physical_sequences']>1 and len(finite) else None,
                    finite_draws=len(finite))
            info['metrics'][metric]=dict(values=dict(zip(NAMES,point.tolist())),effects=effects)
            for oid in present:
                selected=mask&(objects==oid);means=values[selected].mean(0)
                per_object.append(dict(population=name,metric=metric,object_id=int(oid),frames=int(selected.sum()),
                    object_name=next(r['object_name'] for r in records if r['object_id']==oid),
                    values=dict(zip(NAMES,means.tolist())),effects={k:float(v) for k,v in contributions(means).items()}))
        summary[name]=info
    result=dict(completed=True,frames=23200,tracked_frames=22880,streams=320,
        checkpoint_sha256=manifests['on']['checkpoint_sha256'],source_sha256=manifests['on']['source_sha256'],
        prediction_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        aggregation_script_sha256=hashlib.sha256((Path(__file__).parent/'compare_rk_ablation.py').read_bytes()).hexdigest(),
        factual_max_add_m_error=maximum_error,factual_threshold_mismatches=changed_factual_scores,
        mesh_cache_sha256=mesh_hashes,pose_cache_sha256=pose_hashes,hybrid_adds_not_computed=True,
        scope='Posthoc geometric decomposition of frozen writer-on/off trajectories. Hybrids combine one realized rotation with the other realized mesh-center translation at the same frame; they are not inferred trajectories or a new model benchmark. Contributions average the two replacement orders and sum to the factual change. Positive ADD/d error is worse; positive success difference is better. This does not isolate causal writer-rotation versus writer-center channels. No new predictor or FP calls; GT and motion/visibility only for scoring and grouping. One training seed with shared physical-sequence bootstrap.',
        populations=summary,per_object=per_object,
        frames_sha256=hashlib.sha256((a.out/'frames.jsonl').read_bytes()).hexdigest(),
        bootstrap=dict(draws=2000,seed=20260914,physical_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()))
    (a.out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Rotation / mesh-center geometric decomposition','',result['scope'],'',
           '| Population / metric | Off - on | Rotation contribution | Center contribution |',
           '|---|---:|---:|---:|']
    for name,info in summary.items():
        for metric,data_metric in info['metrics'].items():
            cells=[]
            for key in CONTRASTS:
                d=data_metric['effects'][key]
                interval='NA' if d['ci95'] is None else f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                cells.append(f"{d['delta']:+.4f} {interval}")
            lines.append(f'| {name} / {metric} | '+' | '.join(cells)+' |')
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=True,factual_max_add_m_error=maximum_error,tracked_frames=22880)))


if __name__=='__main__':main()

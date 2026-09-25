"""CPU-only audit of fixed episode offsets versus full chronological validation."""
import argparse,json,hashlib,collections,statistics
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def physical(sid):return '/'.join(sid.split('/')[:2])
def heldout(sid):return int(hashlib.sha256(physical(sid).encode()).hexdigest()[:8],16)%5==0
def aggregate(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['object_id']].append(r)
    return dict(frames=len(rows),objects=len(groups),adds_005=100*statistics.mean(statistics.mean(float(r['adds_005']) for r in g) for g in groups.values()) if groups else None,
        heavy_fraction=statistics.mean(r['visibility'] is not None and r['visibility']<.5 for r in rows) if rows else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.project
    init_path=root/'posecnn_train_20260915/runs/full/initializers.json';stream_path=root/'cache/dexycb_s0/streams.jsonl'
    init=json.loads(init_path.read_text())['initializers'];streams={r['stream_id']:r for r in map(json.loads,stream_path.read_text().splitlines()) if r['split']=='train'}
    options=[(k.split('|')[0],v) for k,v in init.items() if v is not None and k.split('|')[0] in streams and streams[k.split('|')[0]]['num_frames']-v['frame_index']>=40 and not heldout(k.split('|')[0])]
    covered={(sid,v['frame_index']+offset) for sid,v in options for offset in (0,1,8,9)};eligible={sid for sid,v in options}
    upper=max(f for sid,f in covered);total=sum(streams[sid]['num_frames'] for sid in eligible)
    result=dict(scope='Descriptive coverage and time-stratified errors, not proof of a causal generalization failure',train=dict(options=len(options),streams=len(eligible),supervised_offsets=[0,1,8,9],unique_reachable_frames=len(covered),total_frames_in_eligible_streams=total,
        reachable_fraction=len(covered)/total,max_absolute_frame=upper,physical_holdout_excluded=True),sources={str(p):sha(p) for p in (init_path,stream_path)},validation={})
    paths=dict(pure_lip=root/'smooth_val_evaluation_20260915/runs/full/residual/scored/predictions.jsonl',jepa_v26=root/'unified_jepa_20260921/decoded_features_v26/validation/step1200_scored/predictions.jsonl')
    identity=None
    for name,path in paths.items():
        rows=list(map(json.loads,path.read_text().splitlines()));keys={(r['stream_id'],r['frame_index']) for r in rows};assert len(keys)==len(rows)==23200
        if identity is None:identity=keys
        assert keys==identity
        groups=dict(all=rows,early=[r for r in rows if r['frame_index']<=upper],late=[r for r in rows if r['frame_index']>upper])
        groups.update({name+'_heavy':[r for r in part if r['visibility'] is not None and r['visibility']<.5] for name,part in list(groups.items()) if name!='all'})
        result['validation'][name]={k:aggregate(v) for k,v in groups.items()};result['sources'][str(path)]=sha(path)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

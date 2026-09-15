"""Read-only initial/prefix/supervised natural-visibility coverage audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def classify(value):
    if value is None: return 'unknown'
    if not 0 <= value <= 1: raise ValueError('Visibility must be a fraction or None')
    return 'severe' if value < .3 else ('occluded' if value < .5 else 'clear')


def fragments(values, start, burn, unroll):
    if start < 0 or burn < 0 or unroll < 1 or start+burn+unroll >= len(values):
        raise ValueError('Insufficient initialization plus observed frames')
    return {'initializer': values[start:start+1], 'first_update': values[start+1:start+2],
        'prefix': values[start+1:start+burn+1], 'supervised': values[start+burn+1:start+burn+unroll+1]}


def summarize(values):
    counts=Counter(classify(v) for v in values);n=sum(counts.values())
    return dict(frames=n, counts=dict(counts), severe_fraction=counts['severe']/n if n else None,
        visibility_lt05_fraction=(counts['severe']+counts['occluded'])/n if n else None,
        unknown_fraction=counts['unknown']/n if n else None)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('cache','experiment','evaluation','index-root','hold-frames','out'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    e=json.loads((a.experiment/'experiment.json').read_text());cache=json.loads((a.cache/'completed.json').read_text())
    raw=(a.cache/'train_visibility.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==cache['output_sha256']
    assert cache['completed'] and cache['split']=='train' and cache['all_frame_positions_covered']
    visible=sorted(map(json.loads,raw.splitlines()),key=lambda r:r['stream_id'])
    streams=sorted([r for r in map(json.loads,(a.index_root/'streams.jsonl').read_text().splitlines()) if r['split']=='train'],key=lambda r:r['stream_id'])
    assert [r['stream_id'] for r in visible]==[r['stream_id'] for r in streams]
    for key in ('split_hash','mesh_hash'):assert cache[key]==e[key]
    train={part:[] for part in ('initializer','first_update','prefix','supervised')}
    manifest_raw=Path(e['training_manifest']).read_bytes();assert hashlib.sha256(manifest_raw).hexdigest()==e['training_manifest_sha256']
    manifest=json.loads(manifest_raw);starts=Counter();per_sample=[]
    for item in manifest:
        s=visible[item['stream']];start=item['start'];starts[start]+=1
        parts=fragments(s['values'],start,8,48)
        for key,values in parts.items():train[key].extend(values)
        per_sample.append(dict(stream=item['stream'],start=start,first_update_class=classify(parts['first_update'][0]),
            prefix_severe=sum(classify(v)=='severe' for v in parts['prefix']),supervised_severe=sum(classify(v)=='severe' for v in parts['supervised'])))
    all_train={part:[] for part in ('initializer','first_update','prefix','later')}
    for s in visible:
        v=s['values'];all_train['initializer'].append(v[0]);all_train['first_update'].append(v[1]);all_train['prefix'].extend(v[1:9]);all_train['later'].extend(v[9:])
    vm=json.loads((a.evaluation/'manifest.json').read_text());assert vm['completed'] and vm['split']=='val' and vm['fp_calls']==0
    assert vm['initial_poses_sha256']==e['initial_poses_sha256']
    vr=(a.evaluation/'predictions.jsonl').read_bytes();vg={}
    for r in map(json.loads,vr.splitlines()):vg.setdefault(r['stream_id'],[]).append(r)
    val={part:[] for part in ('initializer','first_update','prefix','later')};startup_failures=[]
    for sid,rr in sorted(vg.items()):
        rr=sorted(rr,key=lambda r:r['frame_index']);assert rr[0]['initialization']
        val['initializer'].append(rr[0]['visibility']);val['first_update'].append(rr[1]['visibility'])
        val['prefix'].extend(r['visibility'] for r in rr[1:9]);val['later'].extend(r['visibility'] for r in rr[9:])
        failed=next((i for i,r in enumerate(rr) if i and (not r['adds_005'] or r['status']!='ok')),None)
        startup_failures.append(dict(stream_id=sid,initial_success=bool(rr[0]['adds_005']),first_failure_position=failed,first_update_class=classify(rr[1]['visibility'])))
    hold_raw=a.hold_frames.read_bytes();hold=list(map(json.loads,hold_raw.splitlines()));beneficial=[r for r in hold if r['visibility']<.3 and r['initial_stationary_prefix'] and r['hold_initial_success'] and not r['lip_success']]
    onset={r['stream_id']:r['first_failure_position'] for r in startup_failures}
    receipt=dict(completed=True,scope='Read-only natural visibility and saved failure-onset counts. Synthetic occlusion begins after burn+2 under the active configuration. No new training, inference, calibrated reliability, or causal claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_manifest_sha256=e['training_manifest_sha256'],
        train_visibility_sha256=cache['output_sha256'],visibility_provenance_limit=cache['provenance_limit'],
        val_prediction_sha256=hashlib.sha256(vr).hexdigest(),val_checkpoint_sha256=vm['checkpoint_sha256'],hold_frames_sha256=hashlib.sha256(hold_raw).hexdigest(),
        settings=dict(burn_in_frames=8,supervised_unroll_frames=48,synthetic_earliest_observation_index=10),
        actual_training_samples={key:summarize(v) for key,v in train.items()},all_train_streams={key:summarize(v) for key,v in all_train.items()},
        all_val_streams={key:summarize(v) for key,v in val.items()},sample_start_counts=dict(sorted(starts.items())),
        initially_accurate_val_streams=sum(r['initial_success'] for r in startup_failures),
        accurate_init_failure_in_first8=sum(r['initial_success'] and r['first_failure_position'] is not None and r['first_failure_position']<=8 for r in startup_failures),
        static_hold_benefit_frames=len(beneficial),static_hold_benefit_frames_from_first8_failure=sum(onset[r['stream_id']] is not None and onset[r['stream_id']]<=8 for r in beneficial),
        static_hold_benefit_frames_from_first1_failure=sum(onset[r['stream_id']]==1 for r in beneficial))
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2));(a.out/'val_onsets.json').write_text(json.dumps(startup_failures,indent=2))
    (a.out/'sample_visibility.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in per_sample))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

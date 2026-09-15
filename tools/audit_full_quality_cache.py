"""Check complete-stream coverage, own-state feedback, and label prevalence."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(1);root=a.cache;c=json.loads((root/'collection.json').read_text());manifest=json.loads((root/'samples.json').read_text())
    counts={part:dict(observations=0,harmful=0,early_observations=0,early_harmful=0,late_observations=0,late_harmful=0) for part in c['physical_sequences']}
    seen=set();opportunities=unusable=rejected=0
    for rank in range(c['world_size']):
        folder=root/f'rank{rank}';done=json.loads((folder/'completed.json').read_text());assert done['completed'] and not done['smoke']
        opportunities+=done['frame_opportunities'];unusable+=done['unusable_frames'];rejected+=done['rejected_frames']
        for row in map(json.loads,(folder/'progress.jsonl').read_text().splitlines()):
            item=manifest[row['manifest_index']];assert row['manifest_index'] not in seen;seen.add(row['manifest_index'])
            assert row['stream_id']==item['stream_id'] and row['frames']==item['expected_frames']
            if 'file' not in row:assert row['unusable_frames']==row['frames'];continue
            path=folder/row['file'];assert sha(path)==row['sha256'];x=torch.load(path,map_location='cpu',weights_only=False)
            n=len(x['observation']);assert n+row['unusable_frames']==row['frames']
            assert set(x['partition'])=={item['partition']} and set(x['physical_sequence'])=={item['physical_sequence']}
            assert all(x['frame'][i+1]>x['frame'][i] for i in range(n-1));assert torch.all(x['advantage'][:,1]==0)
            assert torch.isfinite(x['observation']).all() and torch.isfinite(x['candidate_pose']).all()
            # Check exact feedback where there are consecutive, accepted proposals.
            for i in range(1,n):
                if x['stream_position'][i]==x['stream_position'][i-1]+1 and x['proposal_status'][i-1]=='ok':
                    assert torch.equal(x['base_pose'][i],x['candidate_pose'][i-1,3])
            expected_advantage=x['adds_d'][:,1:2]-x['adds_d'];assert torch.equal(expected_advantage,x['advantage'])
            assert torch.equal(x['harmful'],x['adds_d']>x['adds_d'][:,1:2]+.005)
            h=x['harmful'][:,3];early=torch.tensor(x['stream_position'])<=8;late=torch.tensor(x['stream_position'])>40
            totals=counts[item['partition']];totals['observations']+=n;totals['harmful']+=int(h.sum())
            totals['early_observations']+=int(early.sum());totals['early_harmful']+=int(h[early].sum())
            totals['late_observations']+=int(late.sum());totals['late_harmful']+=int(h[late].sum())
    assert seen==set(range(len(manifest))) and opportunities==c['expected_observations']
    assert sum(v['observations'] for v in counts.values())+unusable==opportunities
    for v in counts.values():
        for prefix in ('','early_','late_'):v[prefix+'harmful_fraction']=v[prefix+'harmful']/max(1,v[prefix+'observations'])
    result=dict(passed=True,streams=len(seen),frame_opportunities=opportunities,unusable_frames=unusable,rejected_frames=rejected,partitions=counts,
        zero_candidate_checked=True,own_pose_feedback_checked=True,labels_checked=True,raw_data_modified=False,teacher_sha256=sha(c['teacher_checkpoint']))
    assert result['teacher_sha256']==c['teacher_sha256']
    (root/'coverage_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

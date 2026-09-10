import json
from pathlib import Path
from lip.data.motion import motion_thresholds


def merge_sampling(index,length=8,world=1,test_finalized=False):
    index=Path(index);audit=json.loads((index/'audit.json').read_text());summaries={}
    for split in ['train','val']+(['test'] if test_finalized else []):
        pool=[]
        for rank in range(world):
            shard=json.loads((index/'sampling_shards'/f'{split}_L{length}_rank{rank}.json').read_text())
            if shard['world']!=world or shard['split_hash']!=audit['split_hash'] or shard['rank']!=rank:
                raise ValueError('Stale or mismatched sampling shard')
            if any(c['stream']%world!=rank for c in shard['clips']):raise ValueError('Sampling shard ownership mismatch')
            pool.extend(shard['clips'])
        pool.sort(key=lambda c:(c['stream'],c['stride'],c['start']))
        if len({(c['stream'],c['stride'],c['start']) for c in pool})!=len(pool):raise ValueError('Duplicate clips across shards')
        (index/f'{split}_clips_L{length}.json').write_text(json.dumps(pool))
        report=dict(total=len(pool),moving=sum(x['moving'] for x in pool),occluded=sum(x['occluded'] for x in pool),
                    unknown_visibility=sum(x['visibility'] is None for x in pool),moving_center_d=.05,moving_rotation_deg=5,heavy_visibility=.3)
        (index/f'{split}_sampling_L{length}.json').write_text(json.dumps(report,indent=2));summaries[split]=report
    summaries['motion']=motion_thresholds(index)
    return summaries

"""Export sequence/object sums that reproduce a completed paired comparison."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from compare_rk_ablation import episode_populations


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',required=True,type=Path);a=p.parse_args()
    root=a.experiment;e=json.loads((root/'experiment.json').read_text());comparison=json.loads((root/'comparison.json').read_text())
    assert comparison['completed']
    reader=e.get('reader_arm','spatial')
    folders={'parent':Path(e['reference_evaluation']),'control':root/'control/s0_val',reader:root/reader/'s0_val'}
    metrics=('add_01','adds_005','center_mm','rotation_deg');data={};hashes={}
    for name,folder in folders.items():
        manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['completed'] and manifest['checkpoint_sha256']==comparison['checkpoints'][name]
        assert manifest['initial_poses_sha256']==e['initial_poses_sha256'] and manifest['fp_calls']==0
        raw=(folder/'predictions.jsonl').read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        data[name]={(r['stream_id'],r['frame_index']):r for r in map(json.loads,raw.splitlines()) if not r['initialization']}
    keys=set(data['parent']);assert all(set(v)==keys for v in data.values())
    long,_=episode_populations(list(data['parent'].values()))
    grouped={};rows=[]
    for pop in comparison['populations']:
        selected=[key for key in sorted(keys) if pop=='all' or
            pop=='long_occlusion_gt_8_frames' and key in long or
            pop in ('visibility_lt_05','visibility_lt_03') and data['parent'][key]['visibility'] is not None and
            data['parent'][key]['visibility']<(.5 if pop.endswith('05') else .3)]
        assert len(selected)==comparison['populations'][pop]['frames']
        for name,records in data.items():
            groups=defaultdict(lambda:dict(frames=0,**{m:0. for m in metrics}))
            for key in selected:
                row=records[key];group=groups[('/'.join(key[0].split('/')[:2]),row['object_id'])]
                group['frames']+=1
                for metric in metrics:group[metric]+=row[metric]*(100 if metric in ('add_01','adds_005') else 1)
            for (physical,obj),totals in sorted(groups.items()):
                rows.append(dict(population=pop,arm=name,physical_sequence=physical,object_id=obj,**totals))
            # Reaggregate the exported sequence sums to the same object macro.
            objects=defaultdict(lambda:dict(frames=0,**{m:0. for m in metrics}))
            for (_,obj),totals in groups.items():
                for field in totals:objects[obj][field]+=totals[field]
            for metric in metrics:
                point=sum(v[metric]/v['frames'] for v in objects.values())/len(objects)
                expected=comparison['populations'][pop]['metrics'][metric]['values'][name]
                assert abs(point-expected)<1e-9,(pop,name,metric,point,expected)
    path=root/'paired_physical_sequences.csv'
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt=dict(completed=True,rows=len(rows),prediction_sha256=hashes,csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        all_four_metrics_reaggregate_to_report=True,scope='One training seed; sequence/object sufficient statistics. Success fields are sums of percentage-scaled indicators, not per-sequence percentages.')
    (root/'paired_export_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()

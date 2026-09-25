"""Paired controlled geometry interventions; never a native validation ranking."""
import argparse,json,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();rows=[]
    receipts=[]
    for rank in range(8):
        rec=json.loads((a.root/f'rank{rank}/receipt.json').read_text())
        assert rec['completed'] and rec['optimizer_updates']==0
        receipts.append(rec)
        part=list(map(json.loads,(a.root/f'rank{rank}/frames.jsonl').read_text().splitlines()))
        assert len(part)==rec['rows'];rows+=part
    assert len({r['checkpoint_sha256'] for r in receipts})==1
    result=dict(checkpoint_sha256=receipts[0]['checkpoint_sha256'],rows=len(rows),groups={},
                production_model_changed=False,scope='Controlled positive-axis 10-degree perturbations; interventions include unavailable GT; rigid fit is a diagnostic only')
    for label,selected in [('nonsym_rotation',[x for x in rows if not x['symmetric'] and x['condition']!='zero']),
            ('nonsym_zero',[x for x in rows if not x['symmetric'] and x['condition']=='zero']),
            ('nonsym_heavy_rotation',[x for x in rows if not x['symmetric'] and x['condition']!='zero' and x['visibility'] is not None and x['visibility']<.5])]:
        result['groups'][label]=dict(cases=len(selected),physical_sequences=len({x['physical'] for x in selected}),
            rotation_deg={k:statistics.mean(x['metrics'][k]['rotation_deg'] for x in selected) for k in selected[0]['metrics']},
            contamination={k:statistics.mean(x['contamination'][k] for x in selected) for k in selected[0]['contamination']})
    (a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

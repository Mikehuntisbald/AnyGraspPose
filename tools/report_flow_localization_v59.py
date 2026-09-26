"""Equal-frame summaries; preserve all source-owned correspondence regions."""
import argparse,json
from pathlib import Path
from collections import defaultdict


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    rows=[]
    for rank in range(8):
        folder=a.root/f'rank{rank}'
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['completed'] and receipt['weights_frozen']
        rows.extend(json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines())
    assert len(rows)==256
    grouped=defaultdict(list)
    for row in rows:
        for region,value in row['regions'].items():
            grouped[f"{row['base_kind']}/{'heavy' if row['heavy'] else 'nonheavy'}/{region}"].append(value)
    summary={}
    for name,values in sorted(grouped.items()):
        summary[name]=dict(frames=len(values),points=sum(v['points'] for v in values),
            **{key:sum(v[key] for v in values)/len(values) for key in values[0] if key!='points'})
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# V59 frozen flow localization audit','',
        '256 controlled training-partition physical-holdout frames. Frozen V56 weights; no training or pose evaluation.',
        'All means below weight each eligible frame equally. Heavy denotes requested augmentation, not a measured visibility threshold.',
        'Oracle box EPE is an unattainable-or-equal diagnostic lower bound for the existing ±14px endpoint correction; it is not a model result.','',
        '| Case | Frames / points | Zero EPE | Round 0 EPE | Round 1 coarse | Round 1 EPE | No feedback EPE | Outside correction box | Oracle box EPE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in summary.items():
        lines.append(f"| {name} | {v['frames']} / {v['points']} | {v['zero_epe']:.3f} | {v['round0_final_epe']:.3f} | {v['round1_coarse_epe']:.3f} | {v['round1_final_epe']:.3f} | {v['no_feedback_epe']:.3f} | {v['round1_unreachable_fraction']:.1%} | {v['round1_oracle_box_epe']:.3f} |")
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()

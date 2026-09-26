"""Fixed-feature heldout curves; not full-forward reconstruction metrics."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    points=sorted((int(f.stem[4:]),json.loads(f.read_text())) for f in (a.root/'training').glob('eval*.json'))
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    output={}
    for axis,(angle,region) in zip(axes.flat,[(10,'real'),(10,'proxy'),(60,'real'),(60,'proxy')]):
        output[f'{angle}/{region}']={}
        for arm in ('control','extra'):
            values=[]
            for step,data in points:
                v=[r['regions'][region]['epe'] for r in data[arm] if r['heavy'] and r['base_kind']==f'controlled_{angle}' and region in r['regions']]
                values.append(statistics.mean(v))
            output[f'{angle}/{region}'][arm]=dict(steps=[p[0] for p in points],epe=values)
            axis.plot([p[0] for p in points],values,'o-',label=arm)
        axis.set_title(f'{angle} deg / heavy / {region}');axis.set_xlabel('Head updates');axis.set_ylabel('EPE (crop pixels)');axis.grid(alpha=.25);axis.legend()
    fig.suptitle('Frozen-feature holdout: both rounds pooled, equal eligible frame weight')
    fig.savefig(a.root/'cached_flow_curves.png',dpi=160)
    (a.root/'cached_summary.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':main()

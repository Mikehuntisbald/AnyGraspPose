"""Report a fixed-target source-MLP vs DPT/RoPE/normal experiment, no ranking gate."""
import argparse,csv,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    names=('source_mlp','update500','update1000')
    reports={n:json.loads((a.root/'diagnostics'/n/'summary.json').read_text()) for n in names}
    assert all(r['completed'] and r['physical_sequences']==40 for r in reports.values())
    for key in ('dino_layers','feature_layer_weights','fixed_feature_teacher','history_branch_disabled'):
        assert all(r[key]==reports[names[0]][key] for r in reports.values()),key
    folder=a.root/'comparison';folder.mkdir(exist_ok=True)
    selections=[('geometry_focus_real','xyz_mm'),('geometry_focus_real','depth_mm'),('normal_real','angle_deg'),
                ('geometry_focus_proxy','xyz_mm'),('geometry_focus_proxy','depth_mm'),('normal_proxy','angle_deg'),
                ('spatial_hidden_real','retrieval_top1'),('spatial_cad_proxy','retrieval_top1')]
    rows=[];text=['# DPT + 3D RoPE + XYZ-derived normal supervision','',
        'Same fixed40 protocol and fixed step11000 EMA feature teacher. Pose is frozen; this is recovery evaluation. Three changes are combined, so differences cannot identify their individual effects.','',
        '| Case | Target / metric | Source MLP | +500 | +1000 |','|---|---|---:|---:|---:|']
    for case in ('natural','light','heavy_pooled'):
        for region,metric in selections:
            values=[reports[n]['tables'][case][region]['arms'].get('off',{}).get(metric,{}).get('mean') for n in names]
            if any(v is None for v in values):continue
            rows.append(dict(case=case,region=region,metric=metric,**dict(zip(names,values))))
            text.append('| '+case+' | '+region+'/'+metric+' | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    (folder/'REPORT.md').write_text('\n'.join(text)+'\n')
    with (folder/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for ax,(region,metric) in zip(axes.flat,selections[:6]):
        row=next(r for r in rows if r['case']=='heavy_pooled' and r['region']==region and r['metric']==metric)
        ax.bar(['MLP','DPT +500','DPT +1000'],[row[n] for n in names],color=['#999999','#407ba7','#198b78'])
        ax.set_title(region+' / '+metric,fontsize=10)
    fig.suptitle('Controlled heavy occlusion: lower geometry errors are better')
    fig.tight_layout();fig.savefig(folder/'geometry_normals.png',dpi=180);fig.savefig(folder/'geometry_normals.pdf');plt.close(fig)
    (folder/'receipt.json').write_text(json.dumps(dict(completed=True,updates=1000,physical_sequences=40,
        checkpoints={n:r['checkpoint_sha256'] for n,r in reports.items()},fixed_targets_verified=True,
        individual_component_causality_established=False,pose_training=False),indent=2)+'\n')


if __name__=='__main__':main()

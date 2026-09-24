"""Same-checkpoint, same-input RoPE intervention; no fitting or training."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def analyze(root,world):
    records=[];sequences=set();manifests=[]
    for rank in range(world):
        folder=root/f'rank{rank}';m=json.loads((folder/'manifest.json').read_text())
        assert m['completed'] and m['rope_ablation'] and not m['smoke'] and m['world']==world
        assert m['history_branch_disabled'] and m['optimizer_updates']==0
        assert m['intervention_receipt']['model_state_unchanged']
        with (folder/'frames.jsonl').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['frames_sha256']
        assert not sequences.intersection(m['physical_sequences']);sequences.update(m['physical_sequences'])
        records.extend(json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines());manifests.append(m)
    assert len(sequences)==40 and len(records)==16000
    for key in ('checkpoint_sha256','reference_sha256','config_sha256','fixed_feature_teacher','intervention_receipt'):
        assert all(m[key]==manifests[0][key] for m in manifests),key
    pairs=defaultdict(dict)
    for row in records:
        assert row['history']=='off' and row['read_memory_tokens']==0
        key=(row['physical_sequence'],row['case'],row['relative_frame'])
        assert row['rope'] not in pairs[key];pairs[key][row['rope']]=row
    regions=('geometry_focus_real','geometry_focus_proxy','normal_real','normal_proxy',
             'spatial_hidden_real','spatial_cad_proxy','cad_match_real','cad_match_proxy')
    for pair in pairs.values():
        assert set(pair)=={'on','off'};on,off=pair['on'],pair['off']
        for key in ('stream_id','frame_index','phase','target_base_silhouette_fraction','duration','occluder_kinds','remaining_originally_visible_pixel_fraction'):
            assert on[key]==off[key],key
        for region in ('hidden_real','visible_real','cad_proxy'):
            a,b=on[region],off[region];assert (a is None)==(b is None)
            if a is not None:
                for key in ('patches','raw_feature_loss','raw_cosine','raw_retrieval'):assert a[key]==b[key],key
        for region in regions:
            a,b=on[region],off[region];assert (a is None)==(b is None)
            if a is not None:
                for key in ('pixels','patches','stencils','zero_xyz_mm','teacher_entropy'):
                    if key in a:assert a[key]==b[key],(region,key)
    grouped=defaultdict(list);changes=defaultdict(list)
    for row in records:
        if row['phase']!='occlusion':continue
        for region in regions:
            if row[region] is not None:grouped[(row['physical_sequence'],row['case'],region,row['rope'])].append(row[region])
        if row['rope']=='off':changes[(row['physical_sequence'],row['case'])].append(row['output_change'])
    reduced={k:{m:float(np.mean([x[m] for x in v])) for m in v[0]} for k,v in grouped.items()}
    tables={};metric_rows=[];rng=np.random.default_rng(42)
    for case in ('natural','light','heavy8','heavy16','heavy32','heavy_pooled'):
        tables[case]={}
        for region in regions:
            arms={}
            for policy in ('on','off'):
                per_seq=defaultdict(list)
                for (seq,kind,reg,arm),value in reduced.items():
                    if reg==region and arm==policy and (kind.startswith('heavy') if case=='heavy_pooled' else kind==case):per_seq[seq].append(value)
                arms[policy]={seq:{m:float(np.mean([x[m] for x in vals])) for m in vals[0]} for seq,vals in per_seq.items()}
            assert set(arms['on'])==set(arms['off']);common=sorted(arms['on'])
            if not common:continue
            result=dict(sequences=len(common),metrics={})
            samples=rng.integers(len(common),size=(10000,len(common)))
            for metric in arms['on'][common[0]]:
                on=np.array([arms['on'][s][metric] for s in common]);off=np.array([arms['off'][s][metric] for s in common]);delta=on-off
                result['metrics'][metric]=dict(on=float(on.mean()),off=float(off.mean()),on_minus_off=float(delta.mean()),ci95=np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),max_abs_sequence_delta=float(np.abs(delta).max()))
                for seq,a,b,d in zip(common,on,off,delta):metric_rows.append(dict(sequence=seq,case=case,region=region,metric=metric,on=a,off=b,on_minus_off=d))
            tables[case][region]=result
    diagnostic={}
    for case in ('natural','light','heavy_pooled'):
        vals=[v for (seq,kind),rows in changes.items() if (kind.startswith('heavy') if case=='heavy_pooled' else kind==case) for v in rows]
        diagnostic[case]={k:dict(mean=float(np.mean([v[k] for v in vals])),max=float(max(v[k] for v in vals))) for k in vals[0]}
    result=dict(completed=True,checkpoint_sha256=manifests[0]['checkpoint_sha256'],step=manifests[0]['source_step'],
        physical_sequences=40,paired_frames=len(pairs),rows=len(records),optimizer_updates=0,
        same_encoded_observation_and_teacher=True,raw_targets_and_masks_verified=True,model_state_unchanged=True,
        intervention=manifests[0]['rope_intervention'],effective_gains=manifests[0]['intervention_receipt']['effective_gains'],
        reduction='equal sequence mass; heavy durations averaged within sequence; paired sequence bootstrap10000',
        scope='inference-time marginal dependence of this checkpoint; not a from-scratch training ablation; frozen pose, history disabled',
        tables=tables,output_change=diagnostic)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'paired_sequence_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(metric_rows)
    selections=[('geometry_focus_real','xyz_mm'),('geometry_focus_real','depth_mm'),('normal_real','angle_deg'),
                ('geometry_focus_proxy','xyz_mm'),('geometry_focus_proxy','depth_mm'),('normal_proxy','angle_deg'),
                ('spatial_hidden_real','retrieval_top1'),('spatial_cad_proxy','retrieval_top1'),
                ('cad_match_real','cross_entropy'),('cad_match_proxy','cross_entropy'),
                ('cad_match_real','top1_supported'),('cad_match_proxy','top1_supported')]
    text=['# Same-checkpoint RoPE on/off','',f'Checkpoint step{result["step"]}, SHA256 `{result["checkpoint_sha256"]}`.',
          '', 'Fixed40, identical encoded observation/teacher per pair. Off changes only four RoPE gains to0; all model tensors restored and audited. No optimizer updates. History disabled and pose weights frozen.',
          '', 'Differences below are on minus off. Geometry/angle/cross-entropy: negative favors RoPE. Retrieval/support: positive favors RoPE. This cannot establish whether retraining without RoPE would be better.',
          '', '| Case | Target / metric | On | Off | On - off | 95% paired CI |','|---|---|---:|---:|---:|---|']
    for case in ('natural','light','heavy_pooled'):
        for region,metric in selections:
            if region not in tables[case]:continue
            v=tables[case][region]['metrics'][metric]
            text.append(f'| {case} | {region}/{metric} | {v["on"]:.6f} | {v["off"]:.6f} | {v["on_minus_off"]:+.6f} | [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}] |')
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for ax,(region,metric) in zip(axes.flat,selections[:6]):
        v=tables['heavy_pooled'][region]['metrics'][metric]
        ax.bar(['RoPE on','RoPE off'],[v['on'],v['off']],color=['#208b83','#aaa'])
        ax.set_title(region+'/'+metric,fontsize=10)
        ax.text(.5,.97,f'on-off: {v["on_minus_off"]:+.6f}',ha='center',va='top',transform=ax.transAxes,fontsize=9)
    fig.suptitle('Same checkpoint, controlled heavy occlusion');fig.tight_layout()
    fig.savefig(root/'rope_switch.png',dpi=180);fig.savefig(root/'rope_switch.pdf');plt.close(fig)
    print(json.dumps({k:result[k] for k in ('completed','checkpoint_sha256','paired_frames','optimizer_updates','model_state_unchanged')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--world',type=int,default=8)
    a=p.parse_args();analyze(a.run,a.world)

"""Paired sequence-level recovery audit; no fitting or checkpoint selection."""
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def analyze(root, world):
    records=[]; sequences=set(); checkpoint=None; expected=None; metadata=None
    for rank in range(world):
        folder=root/f'rank{rank}'; m=json.loads((folder/'manifest.json').read_text())
        current_metadata={k:m.get(k) for k in ('dino_layers','feature_layer_weights','history_branch_disabled','fixed_feature_teacher','local_structure_diagnostics')}
        if metadata is None:metadata=current_metadata
        assert metadata==current_metadata
        assert m['completed'] and not m['smoke'] and m['world']==world
        assert digest(folder/'frames.jsonl')==m['frames_sha256']
        assert not sequences.intersection(m['physical_sequences']); sequences.update(m['physical_sequences'])
        checkpoint=checkpoint or m['checkpoint_sha256']; expected=expected or set(m['expected_physical_sequences'])
        assert m['checkpoint_sha256']==checkpoint and set(m['expected_physical_sequences'])==expected
        records.extend(map(json.loads,(folder/'frames.jsonl').read_text().splitlines()))
    assert sequences==expected and len(sequences)==40 and len(records)==24000
    pairs=defaultdict(dict)
    for row in records:
        key=(row['physical_sequence'],row['case'],row['relative_frame'])
        assert row['history'] not in pairs[key]; pairs[key][row['history']]=row
    for pair in pairs.values():
        assert set(pair)=={'on','off','scrambled'}
        for field in ('hidden_real','visible_real','cad_proxy'):
            vals=[pair[p][field] for p in ('on','off','scrambled')]
            assert all((v is None)==(vals[0] is None) for v in vals)
            if vals[0] is not None:
                for metric in ('patches','raw_feature_loss','raw_cosine','raw_retrieval'):
                    assert len({v[metric] for v in vals})==1
    regions=('spatial_hidden_real','spatial_visible_real','spatial_cad_proxy','geometry_focus_real','geometry_focus_proxy')
    if any('geometry_canonical_real' in r for r in records):regions+=('geometry_canonical_real','geometry_canonical_proxy')
    if any('cad_match_real' in r for r in records):regions+=('cad_match_real','cad_match_proxy')
    if any('spatial_hidden_real_mid' in r for r in records):regions+=('spatial_hidden_real_mid','spatial_cad_proxy_mid')
    if any('local_real_mid' in r for r in records):regions+=('local_real_mid','local_real_last','local_proxy_mid','local_proxy_last')
    if any('normal_real' in r for r in records):regions+=('normal_real','normal_proxy')
    if any('coarse_geometry_real' in r for r in records):regions+=('coarse_geometry_real','coarse_geometry_proxy','rope_routing_real','rope_routing_proxy')
    grouped=defaultdict(list)
    for row in records:
        if row['phase']!='occlusion': continue
        for region in regions:
            if row[region] is not None:
                grouped[(row['physical_sequence'],row['case'],region,row['history'])].append(row[region])
    reduced={key:{m:float(np.mean([v[m] for v in rows])) for m in rows[0]} for key,rows in grouped.items()}
    metric_rows=[]; tables={}; rng=np.random.default_rng(42)
    for case in ('natural','light','heavy8','heavy16','heavy32','heavy_pooled'):
        tables[case]={}
        for region in regions:
            arms={}
            for policy in ('on','off','scrambled'):
                by_seq=defaultdict(list)
                for (seq,kind,reg,arm),val in reduced.items():
                    if reg==region and arm==policy and (kind.startswith('heavy') if case=='heavy_pooled' else kind==case): by_seq[seq].append(val)
                arms[policy]={seq:{m:float(np.mean([v[m] for v in vals])) for m in vals[0]} for seq,vals in by_seq.items()}
                for seq,vals in arms[policy].items():
                    for metric,value in vals.items(): metric_rows.append(dict(sequence=seq,case=case,region=region,history=policy,metric=metric,value=value))
            common=sorted(set(arms['on'])&set(arms['off'])&set(arms['scrambled']))
            assert set(arms['on'])==set(arms['off'])==set(arms['scrambled'])
            result=dict(sequences=len(common),arms={},paired={})
            if common:
                metrics=list(arms['on'][common[0]])
                for policy in arms:
                    result['arms'][policy]={m:dict(mean=float(np.mean([arms[policy][s][m] for s in common])),median=float(np.median([arms[policy][s][m] for s in common]))) for m in metrics}
                selections=rng.integers(len(common),size=(10000,len(common)))
                for contrast in ('off','scrambled'):
                    result['paired']['on_minus_'+contrast]={}
                    for m in metrics:
                        delta=np.array([arms['on'][s][m]-arms[contrast][s][m] for s in common])
                        ci=np.quantile(delta[selections].mean(1),[.025,.975])
                        result['paired']['on_minus_'+contrast][m]=dict(mean=float(delta.mean()),ci95=ci.tolist())
                if region.startswith('geometry'):
                    delta=np.array([arms['on'][s]['xyz_mm']-arms['on'][s]['zero_xyz_mm'] for s in common])
                    result['on_xyz_minus_zero_mm']=dict(mean=float(delta.mean()),ci95=np.quantile(delta[selections].mean(1),[.025,.975]).tolist())
            tables[case][region]=result
    result=dict(completed=True,checkpoint_sha256=checkpoint,physical_sequences=40,rows=len(records),
        optimizer_updates=0,training_stopped=True,paired_observation_and_target_verified=True,
        reduction='equal sequence mass; heavy cases averaged within each sequence; paired bootstrap 10000',
        scope='shared baseline-conditioned crops; hidden real and CAD proxy separate; XYZ not symmetry reduced; exact patch retrieval not symmetry equivalent correspondence',
        history_scope='same incoming causal memory; scrambled permutes valid-slot content, preserves locations and age; not unrelated-object rejection',tables=tables)
    result['dino_layers']=metadata['dino_layers']
    result['feature_layer_weights']=metadata['feature_layer_weights']
    result['history_branch_disabled']=metadata['history_branch_disabled']
    result['fixed_feature_teacher']=metadata['fixed_feature_teacher']
    result['local_structure_diagnostics']=metadata['local_structure_diagnostics']
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'sequence_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    selections=[('spatial_hidden_real','centered_cosine','Spatial centered cosine (higher)'),
                ('spatial_hidden_real','retrieval_top1','Exact patch retrieval (higher)'),
                ('spatial_hidden_real','spatial_variance_ratio','Spatial variance / teacher'),
                ('geometry_focus_real','xyz_mm','Canonical XYZ mm (lower)'),
                ('geometry_focus_real','xyz_depth_inconsistency_mm','XYZ / depth inconsistency mm (lower)'),
                ('geometry_focus_real','xyz_reprojection_px','Reprojection pixels (lower)')]
    for ax,(region,metric,title) in zip(axes.flat,selections):
        data=tables['heavy_pooled'][region]['arms']
        ax.bar(['history on','off','scrambled'],[data[p][metric]['mean'] for p in ('on','off','scrambled')],color=['#007f86','#aaa','#dc863c'])
        if metric=='xyz_mm': ax.axhline(data['on']['zero_xyz_mm']['mean'],color='black',ls='--',label='constant object center'); ax.legend(fontsize=8)
        ax.set_title(title,fontsize=10); ax.tick_params(axis='x',labelsize=8)
    fig.suptitle(root.name+': controlled heavy occlusion, real hidden targets')
    fig.tight_layout(); fig.savefig(root/'recovery_focus.png',dpi=180); fig.savefig(root/'recovery_focus.pdf'); plt.close(fig)
    print(json.dumps(dict(completed=True,rows=len(records),checkpoint_sha256=checkpoint)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--world',type=int,default=8)
    a=p.parse_args();analyze(a.run,a.world)

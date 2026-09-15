"""Predeclared conditional effects for spatial supervision x direct latent route."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha
from compare_fp_scorecard import load_evaluations,episodes
from compare_rk_ablation import paired_summary
from diagnose_occlusion_events import stable_recovery,successful

ARMS=('s0_l1','s0_l0','s1_l1','s1_l0')
CONTRASTS={
    'spatial_with_latent':{'s1_l1':1,'s0_l1':-1},
    'spatial_without_latent':{'s1_l0':1,'s0_l0':-1},
    'remove_latent_without_spatial':{'s0_l0':1,'s0_l1':-1},
    'remove_latent_with_spatial':{'s1_l0':1,'s1_l1':-1},
    'interaction':{'s1_l0':1,'s0_l0':-1,'s1_l1':-1,'s0_l1':1},
}


def estimate(values,objects,sequences,clusters,weights,names):
    point,draws=paired_summary(values,objects,sequences,weights[:,np.isin(clusters,np.unique(sequences))])
    definitions=dict(CONTRASTS)
    for arm in ARMS:
        for reference in ('residual','fp'): definitions[arm+'_vs_'+reference]={arm:1,reference:-1}
    effects={}
    for name,mapping in definitions.items():
        vector=np.array([mapping.get(n,0) for n in names]);delta=draws@vector;finite=delta[np.isfinite(delta)]
        effects[name]=dict(delta=float(point@vector),ci95=np.quantile(finite,[.025,.975]).tolist(),ci99=np.quantile(finite,[.005,.995]).tolist(),valid_draws=len(finite))
    return dict(frames=len(values),values=dict(zip(names,point.tolist())),effects=effects)


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    root=Path(__file__).resolve().parents[1];r=root/'runs/factorial';e=json.loads((r/'experiment.json').read_text());out=r/'analysis';out.mkdir(exist_ok=False)
    folders={n:r/'evaluation'/n/'scored' for n in ('residual',)+ARMS}
    folders['fp']=Path('/mnt/why/dexycb_lip/fp_val_20260915/runs/full_v2/scored')
    data,manifests=load_evaluations(folders);names=list(data);keys=sorted(data['residual']);rows=[data['residual'][k] for k in keys]
    objects=np.array([v['object_id'] for v in rows]);sequences=np.array([v['physical_sequence'] for v in rows]);clusters=np.unique(sequences)
    weights=np.random.default_rng(20260915).multinomial(len(clusters),np.ones(len(clusters))/len(clusters),size=10000)
    updates=np.array([v['updates_after_initialization'] is not None and v['updates_after_initialization']>0 for v in rows]);bad=np.array([v['initial_bad'] is True for v in rows])
    first8=np.array([v['updates_after_initialization'] is not None and 0<v['updates_after_initialization']<=8 for v in rows])
    symmetric=np.isin(objects,e['symmetric_object_ids']);es=episodes(rows);long_keys={(v['stream_id'],f) for v in es if v['length']>8 for f in v['frames']}
    masks=dict(all=np.ones(len(rows),bool),updates=updates,bad_initial_first8=bad&first8,
        asymmetric_bad_initial_first8=bad&first8&~symmetric,symmetric_bad_initial_first8=bad&first8&symmetric,
        initial_good=updates&~bad,initial_bad=updates&bad,asymmetric=np.ones(len(rows),bool)&~symmetric,symmetric=symmetric,
        visibility_lt_03=np.array([v['visibility'] is not None and v['visibility']<.3 for v in rows]),
        visibility_lt_05=np.array([v['visibility'] is not None and v['visibility']<.5 for v in rows]),
        long_occlusion_gt8=np.array([k in long_keys for k in keys]))
    metrics=('add_01','adds_005','adds_01','rotation_deg','center_mm');populations={};flat=[];paired=[];cluster_rows=[]
    for pop,mask in masks.items():
        populations[pop]={}
        for metric in metrics:
            use=mask&np.array([all(data[n][k][metric] is not None for n in names) for k in keys])
            selected=[k for k,take in zip(keys,use) if take];scale=100 if metric.startswith(('add_','adds_')) else 1
            values=np.array([[data[n][k][metric]*scale for n in names] for k in selected])
            stats=estimate(values,objects[use],sequences[use],clusters,weights,names);populations[pop][metric]=stats
            for effect,v in stats['effects'].items():flat.append(dict(population=pop,metric=metric,contrast=effect,delta=v['delta'],lower95=v['ci95'][0],upper95=v['ci95'][1],lower99=v['ci99'][0],upper99=v['ci99'][1]))
            for s in np.unique(sequences[use]):
                for o in np.unique(objects[use][sequences[use]==s]):
                    take=(sequences[use]==s)&(objects[use]==o)
                    for i,n in enumerate(names):cluster_rows.append(dict(population=pop,metric=metric,physical_sequence=s,object_id=int(o),method=n,frames=int(take.sum()),sum=float(values[take,i].sum())))
    for key,row in zip(keys,rows):
        v={k:row[k] for k in ('stream_id','frame_index','object_id','physical_sequence','visibility','initial_bad','updates_after_initialization')}
        for n in names:
            for metric in metrics:v[n+'_'+metric]=data[n][key][metric]
        paired.append(v)
    event_rows=[];fixed=[]
    for event in es:
        sid=event['stream_id'];key=sid,event['frames'][-1];reference_failed=not successful(data['residual'][key])
        for n in names:
            post=[data[n][sid,frame] for frame in event['post_frames']]
            event_rows.append(dict(method=n,stream_id=sid,object_id=event['object_id'],physical_sequence=event['physical_sequence'],start=event['frames'][0],end=event['frames'][-1],
                length=event['length'],post_complete=event['post_complete'],censor_reason=event['censor_reason'],reference_failed=reference_failed,stable_offset=stable_recovery(post)))
        if event['post_complete'] and reference_failed:fixed.append(event)
    recovery=None
    if fixed:
        values=np.array([[100.*(stable_recovery([data[n][v['stream_id'],frame] for frame in v['post_frames']]) is not None) for n in names] for v in fixed])
        recovery=estimate(values,np.array([v['object_id'] for v in fixed]),np.array([v['physical_sequence'] for v in fixed]),clusters,weights,names)
    write_csv(out/'contrasts.csv',flat);write_csv(out/'paired_frames.csv',paired);write_csv(out/'sequence_object_sums.csv',cluster_rows);write_csv(out/'occlusion_events.csv',event_rows)
    report=dict(completed=True,frames=23200,streams=320,scope='Full native s0 val, shared first-legal PoseCNN, zero FP LIP, every frame and missing initializer retained. Pure FP baseline uses frozen two-iteration track_one. Not official BOP AR.',
        checkpoints={n:m['checkpoint_sha256'] for n,m in manifests.items()},predictions_sha256={n:m['predictions_sha256'] for n,m in manifests.items()},
        source_sha256=e['source_sha256'],experiment_sha256=sha(r/'experiment.json'),initializers_sha256=e['val_initializers_sha256'],
        contrasts=CONTRASTS,primary='Bad-initialization first-eight canonical rotation error, five predeclared factorial contrasts. Lower errors are favorable. Overall/strict/severe accuracy and center are protections, not substitutes.',
        bootstrap=dict(seed=20260915,draws=10000,physical_sequences=len(clusters),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
            ci95='Unadjusted exploratory population interval',ci99='Bonferroni 5-contrast family coverage for the primary rotation metric only; not all populations/metrics or training seeds.'),
        populations=populations,fixed_parent_failed_recovery=recovery,symmetric_object_ids=e['symmetric_object_ids'],models_info_sha256=e['models_info_sha256'],
        recovery_definition='Fixed retained-parent failures at occlusion end; complete ten-clear-frame windows; success requires three consecutive accurate frames. Gaps/missing visibility/init censor. Shared denominator; no method-dependent event selection.',
        candidate_promoted=False,official_test_launched=False,entrypoint_sha256=sha(Path(__file__)))
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({p:populations[p] for p in ('all','bad_initial_first8','visibility_lt_03')},indent=2))


if __name__=='__main__':main()

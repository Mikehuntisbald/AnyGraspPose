"""Keep seed variation distinct from paired validation-population uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_startup_factorial import ARMS, contrasts

NAMES = ('parent', *ARMS)
METRICS = ('add_01', 'adds_005', 'center_mm', 'rotation_deg')


def aggregate(sums, counts, weights):
    numerator = np.einsum('bs,soa->boa', weights, sums)
    denominator = weights @ counts
    ratios = np.divide(numerator, denominator[..., None],
                       out=np.full_like(numerator, np.nan), where=denominator[..., None] > 0)
    with np.errstate(invalid='ignore'):
        return np.nanmean(ratios, axis=1)


def replicate_contrast(points, draws, coefficient, seed_weights):
    """draws: seed x shared physical draw x arm; seeds are whole paired runs."""
    points = np.asarray(points); draws = np.asarray(draws); coefficient = np.asarray(coefficient)
    assert points.shape == (draws.shape[0], draws.shape[2])
    assert seed_weights.shape == (draws.shape[1], draws.shape[0])
    assert np.all(seed_weights.sum(axis=1) == draws.shape[0])
    values = points @ coefficient
    population = draws.mean(axis=0) @ coefficient
    joint = np.einsum('bs,sba->ba', seed_weights / draws.shape[0], draws) @ coefficient
    intervals = {}
    for name, values_drawn in (('paired_population_ci95', population), ('exploratory_seed_and_population_ci95', joint)):
        finite = values_drawn[np.isfinite(values_drawn)]
        if not len(finite): raise ValueError('No usable uncertainty draws')
        intervals[name] = np.quantile(finite, [.025, .975]).tolist()
    return dict(per_seed=values.tolist(), mean=float(values.mean()), seed_std=float(values.std(ddof=1)),
                seed_range=[float(values.min()), float(values.max())], positive_seeds=int((values > 0).sum()),
                negative_seeds=int((values < 0).sum()), **intervals)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--experiment', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); assert len(a.experiment) == 3
    experiments=[]; comparisons=[]; tables=[]; provenance=[]
    for folder in a.experiment:
        e=json.loads((folder/'experiment.json').read_text())
        assert json.loads((folder/'status.json').read_text())['phase']=='completed'
        c=json.loads((folder/'comparison.json').read_text());assert c['completed']
        raw=(folder/'paired_physical_sequences.csv').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==c['paired_csv']['sha256']
        with (folder/'paired_physical_sequences.csv').open() as handle:
            rows=[row for row in csv.DictReader(handle) if row['arm'] in NAMES]
        for arm in ARMS:
            m=json.loads((folder/arm/'s0_val/manifest.json').read_text())
            t=json.loads((folder/arm/'training_receipt.json').read_text())
            assert m['completed'] and m['population_verified'] and m['frames']==23200 and len(m['streams'])==320
            assert m['checkpoint_sha256']==t['checkpoint_sha256']==c['checkpoints'][arm]
            assert m['fp_calls']==m['critic_calls']==0 and m['source_sha256']==e['source_sha256']
            assert m['config']['seed']==e['seed'] and m['initial_poses_sha256']==e['initial_poses_sha256']
        provenance.append(dict(seed=e['seed'], experiment=str(folder), checkpoints=c['checkpoints'],
                               comparison_sha256=hashlib.sha256((folder/'comparison.json').read_bytes()).hexdigest(),
                               paired_csv_sha256=c['paired_csv']['sha256'], training_manifest_sha256=e['training_manifest_sha256']))
        experiments.append(e);comparisons.append(c);tables.append(rows)
    assert [e['seed'] for e in experiments]==[42,1000003,2000003]
    for e in experiments:
        assert all(e[k]==experiments[0][k] for k in ('parent_sha256','source_sha256','split_hash','mesh_hash','initial_poses_sha256','factors','steps'))
    clusters=sorted({r['physical_sequence'] for r in tables[0] if r['population']=='all'})
    assert len(clusters)==40
    rng=np.random.default_rng(20260915)
    weights=rng.multinomial(40,np.full(40,1/40),size=2000)
    seed_weights=rng.multinomial(3,np.full(3,1/3),size=2000)
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(completed=True, seeds=[42,1000003,2000003], provenance=provenance,
                source_sha256=experiments[0]['source_sha256'],initial_poses_sha256=experiments[0]['initial_poses_sha256'],
                scope='Three adaptation-stage seeds share one frozen M1A1 ancestor and the same controlled noisy-GT validation streams. No FP or official non-GT test. Paired population intervals keep these seeds fixed; the second interval also resamples three whole paired runs and is exploratory with only three seeds. Cameras stay together; physical draws are shared across seeds, arms, populations and metrics. No independent-ancestor or disjoint-image claim.',
                bootstrap=dict(draws=2000,seed=20260915,physical_sequences=40,
                               physical_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                               seed_weights_sha256=hashlib.sha256(seed_weights.tobytes()).hexdigest()),populations={})
    exported=[]
    for pop in comparisons[0]['populations']:
        objects=sorted({int(r['object_id']) for r in tables[0] if r['population']==pop})
        arrays=[];reference_counts=None
        for rows in tables:
            sums=np.zeros((len(clusters),len(objects),len(NAMES),len(METRICS)))
            counts=np.full((len(clusters),len(objects),len(NAMES)),-1.)
            for row in rows:
                if row['population']!=pop:continue
                si=clusters.index(row['physical_sequence']);oi=objects.index(int(row['object_id']));ai=NAMES.index(row['arm'])
                assert counts[si,oi,ai]==-1, 'Duplicate grouped row'
                counts[si,oi,ai]=int(row['frames'])
                sums[si,oi,ai]=[float(row[m]) for m in METRICS]
            counts[counts<0]=0
            assert np.all(counts==counts[..., :1])
            counts=counts[...,0]
            if reference_counts is None:reference_counts=counts
            else:assert np.array_equal(counts,reference_counts)
            arrays.append((sums,counts))
        result=dict(frames=int(reference_counts.sum()),objects=len(objects),metrics={})
        for mi,metric in enumerate(METRICS):
            points=[];draws=[]
            for si,(sums,counts) in enumerate(arrays):
                point=aggregate(sums[...,mi],counts,np.ones((1,40)))[0]
                expected=[comparisons[si]['populations'][pop]['metrics'][metric]['values'][name] for name in NAMES]
                assert np.allclose(point,expected,rtol=0,atol=1e-9)
                points.append(point);draws.append(aggregate(sums[...,mi],counts,weights))
            effects={}
            for name,coefficient in contrasts().items():
                effects[name]=replicate_contrast(points,draws,coefficient,seed_weights)
                for seed,value in zip(report['seeds'],effects[name]['per_seed']):
                    exported.append(dict(seed=seed,population=pop,metric=metric,contrast=name,delta=value))
            result['metrics'][metric]=dict(arm_values_by_seed={str(seed):dict(zip(NAMES,point.tolist())) for seed,point in zip(report['seeds'],points)},contrasts=effects)
        report['populations'][pop]=result
    path=a.out/'per_seed_contrasts.csv'
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)
    report['csv_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Startup factorial: three adaptation seeds','',report['scope'],'',
           '| Population / metric | Contrast | Seed 42 / 1000003 / 2000003 | Mean | Paired population 95% CI |',
           '|---|---|---:|---:|---:|']
    for pop,result in report['populations'].items():
        for metric in ('add_01','adds_005'):
            for name in ('S_main','O_main','interaction','S1O1_vs_S0O0'):
                d=result['metrics'][metric]['contrasts'][name];lo,hi=d['paired_population_ci95']
                lines.append(f"| {pop} / {metric} | {name} | "+' / '.join(f'{v:+.3f}' for v in d['per_seed'])+f" | {d['mean']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=True,seeds=report['seeds'],csv_rows=len(exported))))


if __name__=='__main__':main()

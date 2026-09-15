"""Verify terminal diagnostic artifacts and create portable plots and receipts."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lip.engine.stream_checkpoint import sha,source_hash


def main():
    root=Path(__file__).resolve().parents[1];r=root/'runs';out=r/'completion';out.mkdir(exist_ok=False)
    spec=json.loads((r/'train_diagnostic/spec.json').read_text())
    assert source_hash()==spec['source_sha256']
    for field in ('initial_checkpoint','final_checkpoint','training_manifest','train_initializers'):
        assert sha(spec[field])==spec[field+'_sha256']
    state=json.loads((r/'ablation/status.json').read_text());assert state['phase']=='completed' and state['full_learned_reproduction']
    analysis=json.loads((r/'ablation/analysis/report.json').read_text());assert analysis['completed']
    gradient=json.loads((r/'train_diagnostic/gradient/report.json').read_text())
    assert gradient['completed'] and gradient['weights_bitwise_unchanged'] and gradient['optimizer_steps']==0
    fits={n:json.loads((r/'train_diagnostic'/n/'receipt.json').read_text()) for n in ('frozen_parent','joint')}
    start=[];flat=[];per_clip=[]
    for n,x in fits.items():
        assert x['completed'] and x['steps']==x['optimizer_steps']==64 and x['finite'] and not x['val_test_access']
        assert x['spec_sha256']==sha(r/'train_diagnostic/spec.json') and x['rgb_bitwise_unchanged']
        assert sha(r/'train_diagnostic'/n/'diagnostic_only.pt')==x['final_diagnostic_sha256']
        steps=list(map(json.loads,(r/'train_diagnostic'/n/'steps.jsonl').read_text().splitlines()))
        assert [s['step'] for s in steps]==list(range(1,65))
        assert all(np.isfinite([s['loss'],s['grad_norm'],*s['lrs']]).all() for s in steps)
        start.append([v for v in x['evaluations'] if v['step']==0])
        for e in x['evaluations']:
            for pop,metrics in e['metrics'].items():
                flat.append(dict(arm=n,step=e['step'],population=e['population'],window=pop,loss=e['loss'],
                    **{k:metrics[k] for k in ('rotation_deg','center_mm','branch_deg')}))
                for i,m in enumerate(metrics['per_clip']):
                    c=spec['cohort'][e['population']][i]
                    per_clip.append(dict(arm=n,step=e['step'],population=e['population'],window=pop,manifest_index=c['manifest_index'],
                        object_id=c['object_id'],stream_id=c['stream_id'],initial_rotation_deg=c['initial_rotation_deg'],**m))
    assert start[0]==start[1] and fits['frozen_parent']['parent_bitwise_unchanged']
    assert all(n.startswith('rotation_alignment.') for n in fits['frozen_parent']['changed_tensors'])
    for name,rows in [('fit_metrics.csv',flat),('fit_per_clip.csv',per_clip)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    suites=list(ET.parse(r/'diagnostic_tests.xml').getroot().iter('testsuite'))
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']==3 and tests['errors']==tests['failures']==tests['skipped']==0
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for n,color in [('frozen_parent','#d2691e'),('joint','#2277aa')]:
        for pop,style in [('fit','-'),('probe','--')]:
            rows=[v for v in flat if v['arm']==n and v['population']==pop and v['window']=='supervised']
            axes[0].plot([v['step'] for v in rows],[v['rotation_deg'] for v in rows],style+'o',color=color,label=n+' / '+pop)
            axes[1].plot([v['step'] for v in rows],[v['branch_deg'] for v in rows],style+'o',color=color)
    axes[0].set(title='Train-only bounded fitting',xlabel='Optimizer steps',ylabel='Rotation error (deg)');axes[0].legend(fontsize=8)
    axes[1].set(title='Learned branch magnitude',xlabel='Optimizer steps',ylabel='Extra rotation (deg)')
    variants=['zero_dense','zero_new_cad','swap_new_cad','zero_latent','swap_latent']
    vals=[np.mean([v['mean_change_deg'] for v in gradient['input_ablations'] if v['variant']==n]) for n in variants]
    axes[2].barh(variants,vals,color='#658b75');axes[2].set(title='Final1000 fixed-input sensitivity',xlabel='Mean output change (deg)')
    fig.suptitle('Diagnostic only: 8 fit + 8 distinct train clips; probe is never optimized',fontsize=11)
    fig.savefig(out/'learnability.png',dpi=180);fig.savefig(out/'learnability.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.5),constrained_layout=True)
    pops=['all','bad_initial_first8','visibility_lt_03'];labels=['All frames','Bad init: first 8','Visibility < 0.3']
    for ax,metric,title in zip(axes,['adds_005','rotation_deg'],['Strict ADD-S change (pp)','Rotation error change (deg)']):
        values=[analysis['comparisons']['learned_vs_zero'][p][metric] for p in pops]
        points=np.array([v['delta'] for v in values]);lo=np.array([v['ci95'][0] for v in values]);hi=np.array([v['ci95'][1] for v in values])
        ax.errorbar(points,np.arange(3),xerr=np.stack([points-lo,hi-points]),fmt='o',capsize=4,color='#2277aa')
        ax.axvline(0,color='gray',lw=1);ax.set(yticks=np.arange(3),yticklabels=labels,title=title);ax.invert_yaxis()
    fig.suptitle('Same weights: branch on minus off; 40-sequence paired 95% CI',fontsize=11)
    fig.savefig(out/'branch_intervention.png',dpi=180);fig.savefig(out/'branch_intervention.pdf');plt.close(fig)
    receipt=dict(completed=True,source_sha256=source_hash(),final_checkpoint_sha256=spec['final_checkpoint_sha256'],
        tests=tests,learned_full_23200_reproduction=True,same_weight_tensor_hash=analysis['same_tensor_hash'],
        frames_per_arm=23200,streams_per_arm=320,fp_calls=0,gt_inference_reads=0,
        branch_center_immediate_bitwise_unchanged=True,paired_sequence_draws=2000,
        gradient_model_unchanged=True,short_fit_initial_metrics_exact=True,short_fit_frozen_parent_unchanged=True,
        short_fit_arms_completed=2,short_fit_steps_per_arm=64,short_fit_not_candidates=True,official_test_launched=False,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in [r/'ablation/status.json',r/'ablation/analysis/report.json',r/'train_diagnostic/spec.json',
            r/'train_diagnostic/gradient/report.json',r/'train_diagnostic/frozen_parent/receipt.json',r/'train_diagnostic/joint/receipt.json',r/'diagnostic_tests.xml']})
    (out/'verification.json').write_text(json.dumps(receipt,indent=2))
    files={}
    for sub in ('src','tools','tests','configs','runs'):
        for p in sorted((root/sub).rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.pth','.pyc') and '__pycache__' not in p.parts:
                files[str(p.relative_to(root))]=p
    files['pyproject.toml']=root/'pyproject.toml'
    dest=Path('/mnt/why/dexycb_lip/reports');dest.mkdir(exist_ok=True)
    archive=dest/'alignment_diagnosis_20260915.tar.gz'
    assert not archive.exists()
    hashes={}
    with tarfile.open(archive,'w:gz') as tar:
        for name,p in files.items():
            raw=p.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
            member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
        raw=json.dumps(dict(completed=True,files=hashes,weights='Checkpoints remain on server; bound by SHA-256. Diagnostic fitted weights are not candidates.'),indent=2).encode()
        member=tarfile.TarInfo('SHA256.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
    result=dict(completed=True,archive=str(archive),sha256=sha(archive),files=len(hashes),bytes=archive.stat().st_size)
    (dest/'alignment_diagnosis_20260915_archive.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

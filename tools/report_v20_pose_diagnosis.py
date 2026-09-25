"""Consolidated frozen diagnostics with paired sequence bootstrap and exportable plots."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
 d=json.loads((r/'pose_path_summary.json').read_text());old=json.loads((r/'stage40400/pose_path_summary.json').read_text());latent=json.loads((r/'latent/nested_latent_summary.json').read_text());native=json.loads((r/'native_decomposition.json').read_text());train=json.loads((r/'training_bases/summary.json').read_text())
 rows=[]
 for rank in range(8):rows+=list(map(json.loads,(r/f'rank{rank}/frames.jsonl').read_text().splitlines()))
 sub=[x for x in rows if not x['symmetry'] and x['base_kind'].startswith(('axis0','axis1','axis2'))]
 groups={}
 for x in sub:
  groups.setdefault(x['physical_sequence'],[]).append(x['metrics']['v20']['rotation_deg']-x['metrics']['lip_no_history']['rotation_deg'])
 sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()]);idx=np.random.default_rng(42).integers(0,len(sums),(10000,len(sums)));boot=sums[idx].sum(1)/counts[idx].sum(1)
 ci=dict(metric='V20 minus LIP-no-history rotation error',mean=float(sums.sum()/counts.sum()),paired_physical_sequence_ci95=np.quantile(boot,[.025,.975]).tolist(),sequences=len(sums));(r/'paired_ci.json').write_text(json.dumps(ci,indent=2))
 gains={m:[d['gains'][f'nonsymmetric/axis{i}'][m]*100 for i in range(3)] for m in ['v20','lip_no_history']}
 fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 x=np.arange(3);ax=axs[0,0];lip=[83.66415386848323,53.46829312331135,29.278210863985638];v20=[31.334204830338525,10.196868065541121,2.6432903930493135]
 ax.bar(x-.18,lip,.36,label='Old LIP (history on)',color='#3377aa');ax.bar(x+.18,v20,.36,label='V20 (history off)',color='#dd8844');ax.set_xticks(x,['All','Visibility <50%','Visibility <30%']);ax.set_ylabel('ADD-S@0.05d (%)');ax.set_title('Native tracking: same 23,200 frames');ax.legend(fontsize=8)
 ax=axs[0,1];ax.bar(x-.18,gains['lip_no_history'],.36,label='LIP without history',color='#3377aa');ax.bar(x+.18,gains['v20'],.36,label='V20',color='#dd8844');ax.axhline(100,color='gray',ls='--',label='Ideal response');ax.set_xticks(x,['Rotation X','Rotation Y','Rotation Z']);ax.set_ylabel('Signed correction gain (% of ideal)');ax.set_title('Same observation, +/-10 deg; nonsymmetric');ax.legend(fontsize=8)
 ax=axs[1,0];names=['fused','patch','object','lip_object'];values=[latent['results'][n]['nonsymmetric']['rotation10_error_deg'] for n in names];ax.bar(np.arange(4),values,color=['#dd8844']*3+['#3377aa']);ax.axhline(10,color='gray',ls='--');ax.set_xticks(np.arange(4),['V20 fused','V20 patch','V20 query','LIP query']);ax.set_ylabel('Residual rotation error (deg)');ax.set_title('Frozen-feature ridge: held-out physical sequences')
 for i,v in enumerate(values):ax.text(i,v+.08,f'{v:.2f}',ha='center',fontsize=9)
 ax=axs[1,1];names=['completion_off','oracle_surface_and_validity','relation_off'];values=[latent['intervention_output_changes'][n]['rotation_deg']['mean'] for n in names];ax.bar(np.arange(3),values,color='#dd8844');ax.set_yscale('log');ax.set_xticks(np.arange(3),['Completion off','GT surface + validity','All relations off']);ax.set_ylabel('Mean output rotation change (deg)');ax.set_title('Same latent/readout; frozen intervention')
 for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.savefig(r/'diagnosis.png',dpi=180);fig.savefig(r/'diagnosis.pdf');plt.close(fig)
 # An exact geometric illustration of the missing explicit camera XY residual.
 theta=np.pi/18;x=np.array([.2,.1,.3]);rz=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
 illustration=dict(rotation_deg=10.,normalized_xyz=x.tolist(),diameter_m=.15,z_consistency_error_m=float(((rz@x)[2]-x[2])*.15),camera_xy_error_mm=float(np.linalg.norm((rz@x-x)[:2])*.15*1000),scope='Analytical illustration for perfect recovered geometry; does not claim the full network lacks other rotation cues.')
 (r/'camera_relation_example.json').write_text(json.dumps(illustration,indent=2))
 text=f'''# V20 versus old LIP: frozen failure diagnosis

No production model training or optimizer updates. Completed fixed40 physical-sequence diagnostics on119 selected frames spanning20 objects. Non-symmetric rotation comparison uses28 physical sequences and498 paired perturbation cases. Native comparison uses all23,200 val frames. Official test untouched.

## Main findings

1. **Rotation correction response is the primary observed failure.** A separate same-base component swap on119 selected natural frames gives25.21% ADD-S@0.05d with V20,41.18% using the old LIP rotation update plus V20 translation, and30.25% using V20 rotation plus old LIP translation (both old LIP components:46.22%). These are frozen conditional diagnostics, not a new hybrid model or full native score. At a10-degree error, old LIP without history reaches6.818deg; V20 remains10.026deg. V20 signed response gain is0.76%/0.39%/0.62% of ideal across X/Y/Z; LIP without history53.52%/53.35%/56.54%. The paired physical-sequence95% interval for the error gap is{ci['paired_physical_sequence_ci95'][0]:.3f} to{ci['paired_physical_sequence_ci95'][1]:.3f}deg. Translation response exists but is weaker. Old LIP history is not required for this rotation advantage.
2. **Restored geometry has negligible influence on the current pose readout.** Turning completion off changes output by0.00649deg and0.0423mm on average; replacing predicted surface values and validity with GT changes it by0.00619deg and0.0378mm. These are paired output changes, not pose improvements. Oracle supplies geometry only at the readout, not JEPA's earlier staged RoPE pass. All weights stayed frozen. The learned relation gain is{d['relation_gain']:.5f}; mean completion weight across all crop patches is{d['average_completion_weight']:.5f} (includes background, not an object-coverage statistic).
3. **A different final linear readout alone is not supported as the fix.** Five outer folds split by physical sequence, with regularization chosen inside each training fold, produce residual rotation errors: V20 fused9.897deg, patch9.945deg, object query9.927deg; LIP query6.125deg. Model weights never change. This is an exploratory fitted diagnostic within val, not independent downstream evaluation. It bounds simple readout capability; it does not prove that no information exists or that a nonlinear readout cannot help.
4. **The problem predates removal of DINO supervision.** At step40400 (before V20), the same rotation probe gives{old['groups']['nonsymmetric/rotation10']['models']['v20']['rotation_deg']:.5f}deg, versus10.02577deg at45400. Both have under1% signed correction gain. The rotation rows of the head changed during training (`head_weight_audit.json`); this is not an accidentally frozen rotation head.
5. **Training/inference trajectory mismatch amplifies the failure.** `frame_batch_training.py` uses the frozen crop-reference output to build all subsequent student inputs; student outputs never update those training bases. Old LIP's `stream_training.py` updates `accepted` from its own predicted pose.32 actual train episodes replayed with reference versus student feedback give post-anchor median canonical rotation errors19.82deg versus37.13deg and median center error0.0637d versus0.1013d. This replay retains synthetic occlusion, and differing crops alter its realization; it is supporting distribution evidence, not an isolated proof of causality or a symmetry-reduced score.

## What is and is not entering the latent

Matched old/new crops are exactly equal. Changing the estimated pose does change V20 representations: the mean relative L2 change for +/-10deg is10.18% before JEPA,12.76% at patch output,20.88% at object query. The corresponding LIP query change is73.14%. Therefore the geometric input is not completely disconnected; variation is not organized into a reliably readable rotation correction.

The existing recovered-relation vector contains canonical XYZ, nearest canonical CAD residual, relative depth, matched depth residual, and `depth - (R_base X)_z`. It has no explicit same-pixel camera-XYZ or reprojection-XY residual. In the optical-axis example, a10-degree rotation produces exactly zero Z-consistency error but{illustration['camera_xy_error_mm']:.3f}mm XY error for a150mm object. Other network paths can still encode in-plane rotation; this is a concrete missing geometric cue, not a theorem that the entire model cannot estimate it.

A suitable explicit relation to test within the existing unified readout is:

`r_camera = R_base (diameter * X_recovered) + t_base - depth_recovered * K_crop^-1 [u,v,1]`

Use observed depth on trusted visible pixels and confidence-weighted predictions only where needed. Preserve source/confidence distinctions.

## Training protocol gap

Current `pose_pair_frames` is empty. Joint unfreezing restored pose gradients but retained the JEPA-only fixed-reference training layout; it did not restore same-observation/different-estimate paired correction and correct-pose zero-update training. Old LIP's saved config selects GT-centered/noisy starts in the retained trainer, which feeds back its own rollout; this JEPA factory starts from native PoseCNN and optional extra noise. The new training replay is not dominated by tiny errors: reference rotation median16.97deg, and34.13% exceed30deg. Do not explain this run as only having easy near-correct training examples.

## Repair priority supported by these results

First train the original pose head to respond correctly to signed pose perturbations and preserve a correct pose; require measurable rotation gain on held-out physical sequences. Add explicit camera/projection relations to the shared patch/object-query path if readability remains weak. Then introduce student-feedback trajectories gradually. Keep this inside the unified JEPA path, with no FP bypass. Additional DINO/DPT reconstruction training or learning-rate changes alone have not been shown to address the measured failure.

## Limits and audit

- Pure-LIP native baseline SHA89d5a66bc72d8afc5eb57d20b4a4ba52964dc7706dc7058af8bb9a01cde15868; V20 terminal SHAb74ccb3d5de69386fb440aec08f4a8fa9cd00781e63bd16759f16cab5694f29c. Split/mesh/initializers match.
- Native ADD-S@0.05d: LIP83.6642/53.4683/29.2782%, V2031.3342/10.1969/2.6433% for all/<50%/<30% visibility. Native LIP has history; controlled probes include no-history LIP.
- Old LIP's replay is bitwise identical to stored poses (max difference0). V20 eager diagnostic versus compiled native max pose-component difference{d['max_v20_native_replay_difference']:.7f}; therefore small threshold-level diagnostic changes should not be overinterpreted.
- GT bases, GT surfaces/validity and `patch_minus_gt_ORACLE` use unavailable GT information and are not deployable metrics. Predicted-geometry rigid fits also use GT support and include unsupervised visible regions; they are not a standalone geometry-quality verdict.
- Full model-state hashes match before/after every val probe. The train replay uses no optimizer and no backward. Ridge readout fits are isolated diagnostic models and never replace production parameters.
- These experiments establish functional bottlenecks and support repair priorities; they do not establish that any proposed repair will recover the52.33pp native gap.
'''
 (r/'REPORT.md').write_text(text)
 receipt=dict(completed=True,model_training=False,model_optimizer_updates=0,official_test_access=False,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'REPORT.md',r/'diagnosis.png',r/'diagnosis.pdf',r/'paired_ci.json']})
 (r/'report_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(ci))
if __name__=='__main__':main()

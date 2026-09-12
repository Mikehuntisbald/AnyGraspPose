"""Summarize completed paired diagnostics and saved full-validation FP transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve()
    def read(name):return json.loads((out/name).read_text())
    mv=read('multiview_depth.json')['rows'];oracle=read('fp_depth_intervention.json')['rows']
    tables=read('table_consistency.json')['paired'];geometry=read('mesh_depth_hypotheses.json')['rows']
    clear=[r for r in mv if r['image_fraction']>=.98 and r['visible_fraction_in_image']>=.9]
    stages=list(map(json.loads,(root/'runs/fp_transition_analysis_21530_31000/frames_31000.jsonl').read_text().splitlines()))
    stages=[r for r in stages if r['fp_update']];assert len(stages)==22880
    cameras=sorted(set(r['camera'] for r in mv));camera_rows=[]
    for c in cameras:
        cr=[r for r in clear if r['camera']==c];val=[r for r in stages if r['stream_id'].endswith(c)]
        assert len(val)==2860
        # Full-validation camera results use object macro averages, excluding initial GT frames.
        macro={}
        for side in ['before','after']:
            macro[side]={k:float(np.mean([np.mean([r[side][k] for r in val if r['object_id']==oid]) for oid in sorted(set(r['object_id'] for r in val))])) for k in ['center_mm','rotation_deg','add_01','adds_01']}
        pair=[r['b_minus_a_height_mm'] for r in tables if r['camera_a']=='839512060362' and r['camera_b']==c]
        camera_rows.append(dict(camera=c,clear_observations=len(cr),clear_depth_gap_median_mm=float(np.median([r['depth_gap_median_mm'] for r in cr])),
            table_height_vs_0362_median_mm=float(np.median(pair)) if pair else None,
            val_updates=len(val),before=macro['before'],after=macro['after'],mean_FP_z_shift_mm=float(np.mean([r['shift_xyz_mm'][2] for r in val]))))
    def oracle_stats(rs):
        return {mode:dict(center_mm=float(np.mean([r[mode]['errors']['center_mm'] for r in rs])),
            rotation_deg=float(np.mean([r[mode]['errors']['rotation_deg'] for r in rs])),
            signed_z_mm=float(np.mean([r[mode]['signed_xyz_mm'][2] for r in rs]))) for mode in ['observed','oracle_target_depth']}
    untruncated=[];selection_coverage=[]
    for r in oracle:
        q=next(m for m in mv if m['stream_id']==r['stream_id'] and m['frame_index']==r['frame_index'] and m['object_id']==r['object_id'])
        selection_coverage.append(dict(object_id=r['object_id'],image_fraction=q['image_fraction'],visibility_in_image=q['visible_fraction_in_image']))
        if q['image_fraction']>=.98:untruncated.append(r)
    summary=dict(completed=True,objects=20,physical_frame_selections=20,camera_frames=160,object_observations=len(mv),clear_full_image_observations=len(clear),
        oracle_frames=len(oracle),oracle_center_improved=sum(r['oracle_target_depth']['errors']['center_mm']<r['observed']['errors']['center_mm'] for r in oracle),
        oracle=oracle_stats(oracle),oracle_untruncated_frames=len(untruncated),oracle_untruncated=oracle_stats(untruncated),
        repeated_real_FP_max_abs=max(r['repeated_real_FP_pose_max_abs'] for r in oracle),camera_rows=camera_rows,
        renderer_comparison_pixels=sum(r['renderer_comparison_pixels'] for r in geometry),renderer_diff_over_1mm_pixels=sum(r['renderer_diff_over_1mm_pixels'] for r in geometry),
        renderer_per_frame_p99_diff_max_mm=max(r['renderer_diff_p99_mm'] for r in geometry),
        renderer_max_diff_mm=max(r['renderer_diff_max_mm'] for r in geometry),
        high_res_mesh_max_median_depth_change_mm=max(r.get('high_vs_simple_depth_median_abs_mm',0) for r in geometry),selection_coverage=selection_coverage,
        scope='Diagnostics only. GT-depth replacement is an oracle intervention, never an inference option. Camera depth statistics use 460 clear/untruncated observations. Full-val FP transitions use saved 31k hybrid histories and object macro means excluding first GT frames; camera association is descriptive.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained');xx=np.arange(8);labels=[c[-4:] for c in cameras]
    axes[0].bar(xx,[r['clear_depth_gap_median_mm'] for r in camera_rows],color=['#4477aa']*4+['#cc6677']*4)
    axes[0].axhline(0,color='k',linewidth=.7);axes[0].set_xticks(xx,labels,rotation=45);axes[0].set_ylabel('Observed - GT render (mm)')
    axes[0].set_title('460 clear, untruncated object views\nMedian signed depth gap by camera')
    axes[1].bar(xx,[(r['after']['add_01']-r['before']['add_01'])*100 for r in camera_rows],color='#cc6677')
    axes[1].axhline(0,color='k',linewidth=.7);axes[1].set_xticks(xx,labels,rotation=45);axes[1].set_ylabel('ADD@0.1d change (percentage points)')
    axes[1].set_title('31k full val: current FP update\nSame input history, 2,860 updates/camera')
    for i,r in enumerate(oracle):
        before=r['observed']['errors']['center_mm'];after=r['oracle_target_depth']['errors']['center_mm']
        axes[2].plot([0,1],[before,after],marker='o',ms=3,color='#4477aa',alpha=.5)
    means=[summary['oracle'][mode]['center_mm'] for mode in ['observed','oracle_target_depth']]
    axes[2].plot([0,1],means,color='black',marker='D',linewidth=3,label='Mean')
    axes[2].set_xticks([0,1],['Observed depth','Oracle GT depth']);axes[2].set_ylabel('FP(GT) center error (mm)');axes[2].legend()
    axes[2].set_title('Paired depth-only intervention, n=20\nDiagnostic only; RGB and GT init fixed')
    fig.savefig(out/'audit_summary.png',dpi=150);plt.close(fig)
    # Hash every diagnostic artifact and the six executable scripts. Receipt itself is excluded.
    artifacts={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.iterdir()) if path.is_file() and path.name not in ['receipt.json','summary_stdout.log']}
    for name in ['diagnose_rgbd_alignment.py','diagnose_multiview_depth.py','diagnose_table_consistency.py','probe_fp_depth_intervention.py','check_mesh_depth_hypotheses.py','summarize_rgbd_audit.py']:
        path=root/'tools'/name;artifacts[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'receipt.json').write_text(json.dumps(dict(completed=True,artifacts_sha256=artifacts),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

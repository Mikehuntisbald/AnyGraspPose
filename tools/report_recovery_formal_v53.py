"""Refresh recovery-only monitoring after each scheduled milestone."""
import argparse,json
from pathlib import Path
from report_cad_image_v45 import read
from report_recovery_balance_v52 import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    report={}
    for split in ('usual','confirmation'):
        data={}
        for stage in sorted((a.root/'probe').glob('step*'),key=lambda p:int(p.name[4:])):
            path=stage/split
            if len(list(path.glob('rank*/receipt.json')))==8:data[stage.name]=read(path)
        if data:report[split]=paired(data)
    result=dict(recovery_only=True,source_step=200,target_step=5200,additional_budget=5000,
        metrics=report,pose_training=False,pose_evaluation=False,default_model_changed=False)
    (a.root/'monitoring.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V53 formal JEPA recovery monitoring','','Full-state continuation of V52 balanced200;5000 additional updates. Same loss/architecture/data; CE weight0.1. Pose frozen, history and DINO feature losses off.','',
        'Step numbers continue the V52 clock:200 is the inherited source,700/1200/2700/5200 correspond to500/1000/2500/5000 new updates. Both sets are previously inspected training-partition physical holdouts; use as development monitoring, not an unseen test.','']
    for split,values in report.items():
        lines += [f'## {split}: heavy subset','','|Step|Real CAD XYZ mm|Real depth mm|Proxy CAD XYZ mm|Proxy depth mm|Real surface angle deg|Proxy surface angle deg|','|---|---:|---:|---:|---:|---:|---:|']
        for step,m in values['heavy'].items():
            lines.append('|'+step+'|'+'|'.join(f'{m[k]:.3f}' for k in ('real/canonical_xyz_mm','real/depth_mm','proxy/canonical_xyz_mm','proxy/depth_mm','real/camera_normal_deg','proxy/camera_normal_deg'))+'|')
        lines+=['']
    lines+=['Original real depth and CAD-proxy labels/masks are unchanged. Canonical XYZ measures CAD identity; camera XYZ is depth lifted through rays. Every50 steps saves a full checkpoint; evaluated milestones retain immutable full checkpoints. Automatic stops are runtime/nonfinite failures or the budget boundary; metric changes are reported without selecting or replacing the default model.']
    (a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()

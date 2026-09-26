"""Matched joint-training outcomes; intervention sensitivity is not restoration accuracy."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    finished=json.loads((r/'finish/status.json').read_text());assert finished['completed']
    paired=json.loads((r/'paired_start_identity.json').read_text());assert all(paired.values())
    result=dict(goal_complete=False,paired_start=paired,native={},recovery={},controlled={},terminal_checkpoints={},default_model_changed=False)
    identity=None
    for arm in ('control','shared'):
        result['terminal_checkpoints'][arm]=json.loads((r/arm/'runs/seed42/last.receipt.json').read_text())
        for label in ('step1450','step1700')+ (('step1700_patch_off_verified',) if arm=='shared' else ()):
            m=json.loads((r/arm/'validation'/(label+'_scored')/'metrics.json').read_text());assert m['completed'] and m['frames']==23200
            current={k:m[k] for k in ('frames','streams','initializers_sha256','visibility_reference_sha256')}
            if identity is None:identity=current
            assert current==identity
            result['native'][arm+'/'+label]={k:m['populations'][k]['adds_005'] for k in ('all','visibility_lt_05','visibility_lt_03')}
        m=json.loads((r/arm/'recovery/step1700/summary.json').read_text());assert m['completed'] and m['rows']==24000
        result['recovery'][arm]={region:{k:m['tables']['heavy_pooled'][region]['arms']['off'][k]['mean'] for k in ('xyz_mm','depth_mm','xyz_depth_inconsistency_mm')} for region in ('geometry_focus_real','geometry_focus_proxy')}
        result['controlled'][arm]=json.loads((r/arm/'controlled/step1700/summary.json').read_text())['groups']
    for rank in range(8):
        left=r/f'control/recovery/step1700/rank{rank}';right=r/f'shared/recovery/step1700/rank{rank}'
        x=json.loads((left/'manifest.json').read_text());y=json.loads((right/'manifest.json').read_text())
        for key in ('reference_sha256','physical_sequences','occluder_bank_sha256','fixed_feature_teacher','dino_layers','feature_layer_weights'):assert x[key]==y[key]
        xs=list(map(json.loads,(left/'frames.jsonl').read_text().splitlines()));ys=list(map(json.loads,(right/'frames.jsonl').read_text().splitlines()));assert len(xs)==len(ys)
        for x,y in zip(xs,ys):
            for key in ('physical_sequence','case','relative_frame','history'):assert x[key]==y[key]
            for key in ('geometry_focus_real','geometry_focus_proxy'):
                assert (x[key] is None)==(y[key] is None)
                if x[key] is not None:assert x[key]['pixels']==y[key]['pixels']
        on=json.loads((r/f'shared/validation/step1700/rank{rank}/manifest.json').read_text())
        off=json.loads((r/f'shared/validation/step1700_patch_off_verified/rank{rank}/manifest.json').read_text())
        assert on['checkpoint_sha256']==off['checkpoint_sha256'] and on['config_sha256']==off['config_sha256']
        assert off['disable_shared_patch'] and not off['shared_patch_enabled']
        assert off['gt_pose_reads']==off['gt_mask_reads']==off['gt_resets']==0
    original=json.loads((r/'source_receipt.json').read_text())['files'];finish=json.loads((r/'finish/source_receipt.json').read_text())['files']
    assert all(finish[k]==v for k,v in original.items() if k.startswith(('src/','configs/')))
    result['paired_protocol_verified']=True
    result['candidate_minus_control_pp']={k:result['native']['shared/step1700'][k]-result['native']['control/step1700'][k] for k in result['native']['control/step1700']}
    result['same_weight_patch_on_minus_off_pp']={k:result['native']['shared/step1700'][k]-result['native']['shared/step1700_patch_off_verified'][k] for k in result['native']['control/step1700']}
    result['conclusion']='Shared patch is used but does not beat equal-budget control on heavy pose or improve XYZ restoration. Preserve both; no promotion. Full-window curriculum uses the control parent to isolate data coverage.'
    (r/'outcome.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('native','recovery','candidate_minus_control_pp','same_weight_patch_on_minus_off_pp')},indent=2))

if __name__=='__main__':main()

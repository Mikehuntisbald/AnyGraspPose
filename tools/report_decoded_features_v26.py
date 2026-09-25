"""Matched feature-ownership comparison, not a declaration of task completion."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();r=a.root;control=r.parent/'amp_preview_v25/fixed'
    result=dict(goal_complete=False,paired_start=json.loads((r/'paired_start_audit.json').read_text()),native={},recovery={})
    identity=None
    for name,path in [('observed_visible',control),('decoded',r)]:
        m=json.loads((path/'validation/step1200_scored/metrics.json').read_text());assert m['completed'] and m['frames']==23200
        ident={k:m[k] for k in ('frames','streams','initializers_sha256','visibility_reference_sha256')}
        if identity is None:identity=ident
        assert ident==identity
        result['native'][name]={k:m['populations'][k]['adds_005'] for k in ('all','visibility_lt_05','visibility_lt_03')}
        m=json.loads((path/'recovery/step1200/summary.json').read_text());assert m['completed'] and m['rows']==24000
        result['recovery'][name]={region:{k:m['tables']['heavy_pooled'][region]['arms']['off'][k]['mean'] for k in ('xyz_mm','depth_mm','xyz_depth_inconsistency_mm')} for region in ('geometry_focus_real','geometry_focus_proxy')}
    for rank in range(8):
        left=control/f'recovery/step1200/rank{rank}';right=r/f'recovery/step1200/rank{rank}'
        x=json.loads((left/'manifest.json').read_text());y=json.loads((right/'manifest.json').read_text())
        for key in ('reference_sha256','physical_sequences','occluder_bank_sha256','fixed_feature_teacher','dino_layers','feature_layer_weights'):assert x[key]==y[key]
        xs=list(map(json.loads,(left/'frames.jsonl').read_text().splitlines()));ys=list(map(json.loads,(right/'frames.jsonl').read_text().splitlines()));assert len(xs)==len(ys)
        for x,y in zip(xs,ys):
            for key in ('physical_sequence','case','relative_frame','history'):assert x[key]==y[key]
            for key in ('geometry_focus_real','geometry_focus_proxy'):
                assert (x[key] is None)==(y[key] is None)
                if x[key] is not None:assert x[key]['pixels']==y[key]['pixels']
    result['protocol_verified']=True
    result['delta_pp']={k:result['native']['decoded'][k]-result['native']['observed_visible'][k] for k in result['native']['decoded']}
    result['controlled']=json.loads((r/'controlled/step1200/summary.json').read_text())
    result['conclusion']='Feature ownership now follows JEPA, but heavy native pose and XYZ did not improve. Preserve experiment; do not promote or extend blindly.'
    (r/'outcome.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

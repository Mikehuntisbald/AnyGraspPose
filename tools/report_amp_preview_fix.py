"""Matched software-fix outcome, with explicit starting-state and protocol checks."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;experiments=r.parents[1]
    runs={'parent':(experiments/'serial_completion_v21','step1000'),'old_control':(experiments/'geometry_fidelity_v22/control','step1200'),'fixed':(r,'step1200')}
    result=dict(goal_complete=False,native={},recovery={},paired_start=json.loads((r/'paired_start_audit.json').read_text()),gradient_updates=json.loads((r/'gradient_update_audit.json').read_text()))
    identity=None
    for name,(path,step) in runs.items():
        m=json.loads((path/f'validation/{step}_scored/metrics.json').read_text());assert m['completed'] and m['frames']==23200
        current={k:m[k] for k in ('frames','streams','initializers_sha256','visibility_reference_sha256')}
        if identity is None:identity=current
        assert current==identity
        result['native'][name]={k:m['populations'][k]['adds_005'] for k in ('all','visibility_lt_05','visibility_lt_03')}
        m=json.loads((path/f'recovery/{step}/summary.json').read_text());assert m['completed'] and m['rows']==24000
        result['recovery'][name]={k:{metric:m['tables']['heavy_pooled'][k]['arms']['off'][metric]['mean'] for metric in ('xyz_mm','depth_mm','xyz_depth_inconsistency_mm')} for k in ('geometry_focus_real','geometry_focus_proxy')}
    for rank in range(8):
        old=experiments/f'geometry_fidelity_v22/control/recovery/step1200/rank{rank}';new=r/f'recovery/step1200/rank{rank}'
        x=json.loads((old/'manifest.json').read_text());y=json.loads((new/'manifest.json').read_text())
        for key in ('reference_sha256','physical_sequences','occluder_bank_sha256','fixed_feature_teacher','dino_layers','feature_layer_weights'):assert x[key]==y[key]
        xs=list(map(json.loads,(old/'frames.jsonl').read_text().splitlines()));ys=list(map(json.loads,(new/'frames.jsonl').read_text().splitlines()));assert len(xs)==len(ys)
        for x,y in zip(xs,ys):
            for key in ('physical_sequence','case','relative_frame','history'):assert x[key]==y[key]
            for key in ('geometry_focus_real','geometry_focus_proxy'):
                assert (x[key] is None)==(y[key] is None)
                if x[key] is not None:assert x[key]['pixels']==y[key]['pixels']
    result['protocol_verified']=True
    result['fixed_minus_control_pp']={k:result['native']['fixed'][k]-result['native']['old_control'][k] for k in result['native']['fixed']}
    result['controlled']=json.loads((r/'controlled/step1200/summary.json').read_text())
    result['conclusion']='Keep software gradient fix; this 200-update checkpoint does not demonstrate native improvement and is not promoted.'
    (r/'outcome.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()

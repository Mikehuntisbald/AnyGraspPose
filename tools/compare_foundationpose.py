"""Fail closed on population mismatch before reporting the FP/LIP comparison."""
import argparse,json
from pathlib import Path


def compare(lip,fp):
    lip,fp=Path(lip),Path(fp)
    lm=json.loads((lip/'manifest.json').read_text());fm=json.loads((fp/'manifest.json').read_text())
    for key in ['split','mode','initial_pose_source','split_hash','mesh_hash','streams','full_sequences','quick_subset']:
        if lm[key]!=fm[key]:raise ValueError('Protocol mismatch: '+key)
    if not fm.get('completed'):raise ValueError('FoundationPose evaluation is incomplete')
    def rows(path):
        result={}
        for line in (path/'predictions.jsonl').read_text().splitlines():
            r=json.loads(line);k=(r['stream_id'],r['frame_index'])
            if k in result:raise ValueError('Duplicate frame')
            result[k]=r
        return result
    a,b=rows(lip),rows(fp)
    if a.keys()!=b.keys():raise ValueError('Frame population mismatch')
    for k in a:
        for field in ['object_id','visibility','visibility_bin','moving']:
            if a[k][field]!=b[k][field]:raise ValueError('Stratification mismatch')
    lr=json.loads((lip/'metrics.json').read_text());fr=json.loads((fp/'metrics.json').read_text())
    return dict(matched=True,frames=len(a),streams=len(lm['streams']),lip=lr['macro_object'],foundationpose=fr['macro_object'],
                lip_minus_foundationpose={k:lr['macro_object'][k]-fr['macro_object'][k] for k in lr['macro_object']},
                scope='full val, GT first frame, closed loop; object macro averages; not final test',
                precision=dict(lip=lm['config']['precision'],foundationpose=fm['precision']),
                latency=dict(lip=lr['latency_seconds'],foundationpose=fr['latency_seconds']),
                training_regime='LIP trained on DexYCB s0 train; FoundationPose frozen released refiner weights')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lip',required=True);p.add_argument('--fp',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=compare(a.lip,a.fp);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

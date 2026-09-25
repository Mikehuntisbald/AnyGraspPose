"""Compare same-parent continuation arms without automatically selecting a model."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    parent=root.parent/'serial_completion_v21'
    result=dict(status=json.loads((root/'pilot_controller/status.json').read_text()),native={},recovery={},training={},goal_complete=False)
    runs=[('parent',parent,'step1000')]+[(arm,root/arm,'step1200') for arm in ('control','priority')]
    reference=None
    for name,path,step in runs:
        f=path/'validation'/f'{step}_scored/metrics.json'
        if f.exists():
            m=json.loads(f.read_text());assert m['completed'] and m['frames']==23200
            identity={k:m[k] for k in ('initializers_sha256','frames','streams','visibility_reference_sha256')}
            if reference is None:reference=identity
            assert identity==reference
            result['native'][name]={key:m['populations'][key]['adds_005'] for key in ('all','visibility_lt_05','visibility_lt_03')}
        f=path/'recovery'/step/'summary.json'
        if f.exists():
            m=json.loads(f.read_text());assert m['completed'] and m['rows']==24000
            result['recovery'][name]={}
            for key in ('geometry_focus_real','geometry_focus_proxy'):
                v=m['tables']['heavy_pooled'][key]['arms']['off']
                result['recovery'][name][key]={k:v[k]['mean'] for k in ('xyz_mm','depth_mm','xyz_depth_inconsistency_mm')}
        if name!='parent':
            log=path/'runs/seed42/rank0.jsonl'
            if log.exists():
                rows=list(map(json.loads,log.read_text().splitlines()))
                result['training'][name]=dict(step=rows[-1]['step'],mean_seconds=float(np.mean([x['seconds'] for x in rows[-50:]])))
                if 'gradient_balance' in rows[-1]:
                    result['training'][name]['gradient_balance']={k:float(np.mean([x['gradient_balance'][k] for x in rows[-50:]])) for k in rows[-1]['gradient_balance']}
    verified=[]
    for arm in ('control','priority'):
        path=root/arm/'recovery/step1200'
        if arm not in result['recovery']:continue
        for rank in range(8):
            m=json.loads((path/f'rank{rank}/manifest.json').read_text())
            baseline=json.loads((parent/f'recovery/step1000/rank{rank}/manifest.json').read_text())
            for key in ('reference_sha256','physical_sequences','occluder_bank_sha256','fixed_feature_teacher','dino_layers','feature_layer_weights'):assert m[key]==baseline[key],key
            left=list(map(json.loads,(path/f'rank{rank}/frames.jsonl').read_text().splitlines()))
            right=list(map(json.loads,(parent/f'recovery/step1000/rank{rank}/frames.jsonl').read_text().splitlines()))
            assert len(left)==len(right)
            for x,y in zip(left,right):
                for key in ('physical_sequence','case','relative_frame','history'):assert x[key]==y[key]
                for key in ('geometry_focus_real','geometry_focus_proxy'):
                    assert (x[key] is None)==(y[key] is None)
                    if x[key] is not None:assert x[key]['pixels']==y[key]['pixels']
        verified.append(arm)
    result['paired_recovery_protocol_verified']=verified
    if all(x in result['native'] and x in result['recovery'] for x in ('control','priority')):
        result['priority_vs_control']=dict(native_delta_pp={k:result['native']['priority'][k]-result['native']['control'][k] for k in result['native']['control']},
            geometry_ratio={r:{k:result['recovery']['priority'][r][k]/result['recovery']['control'][r][k] for k in ('xyz_mm','depth_mm')} for r in result['recovery']['control']})
    (root/'pilot_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V22 同起点对照','',
        '目标仍是准确补全并可靠估姿；本表只用于决定下一步，不代表目标完成。两组同seed、同标量损失和学习率，均修正反馈帧运动状态；实验组单独限制恢复参数上的次要梯度。','',
        '| 模型 | 全帧 ADD-S@0.05d | 可见率<50% | 可见率<30% |','|---|---:|---:|---:|']
    for name,v in result['native'].items():lines.append('| '+name+' | '+' | '.join(f'{v[k]:.2f}%' for k in ('all','visibility_lt_05','visibility_lt_03'))+' |')
    lines+=['','| 模型 | 真实 XYZ mm | 真实深度 mm | 代理 XYZ mm | 代理深度 mm |','|---|---:|---:|---:|---:|']
    for name,v in result['recovery'].items():lines.append('| '+name+' | '+' | '.join(f'{v[r][k]:.2f}' for r in ('geometry_focus_real','geometry_focus_proxy') for k in ('xyz_mm','depth_mm'))+' |')
    lines+=['','状态：`'+result['status']['job']+'`。','同crop/固定teacher/遮挡及监督像素数量核对：'+', '.join(verified)+'。','原旧LIP全帧83.66%、重遮挡53.47%的最终要求未被这个小规模对照替代。']
    (root/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()

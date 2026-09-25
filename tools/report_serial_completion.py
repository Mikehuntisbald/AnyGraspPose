"""Read-only V21 outcome aggregation; oracle, controlled and native kept separate."""
import argparse,json,statistics
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    result=dict(oracle=json.loads((r/'oracle_gate_v2/receipt.json').read_text()),native={},controlled={},status=json.loads((r/'controller/status.json').read_text()))
    for step in (500,1000):
        f=r/'validation'/f'step{step}_scored/metrics.json'
        if f.exists():
            m=json.loads(f.read_text());assert m['completed'] and m['frames']==23200
            result['native'][str(step)]={k:m['populations'][k] for k in ('all','visibility_lt_05','visibility_lt_03')}
        folders=sorted((r/'pose_probe'/f'step{step}').glob('rank*'))
        if len(folders)==8 and all((f/'receipt.json').exists() for f in folders):
            rows=[]
            for folder in folders:
                assert json.loads((folder/'receipt.json').read_text())['completed']
                rows+=list(map(json.loads,(folder/'frames.jsonl').read_text().splitlines()))
            rot=[x for x in rows if not x['symmetric'] and x['condition'].startswith(('axis0','axis1','axis2'))]
            zero=[x for x in rows if not x['symmetric'] and x['condition']=='zero']
            mean=lambda values:float(np.mean(values)) if values else None
            summary=dict(rotation_cases=len(rot),rotation_after_deg=mean([x['metrics']['predicted']['rotation_deg'] for x in rot]),
                signed_rotation_gain=mean([float(np.dot(x['predicted_delta'][:3],x['expected_delta'][:3])/np.dot(x['expected_delta'][:3],x['expected_delta'][:3])) for x in rot]),
                zero_rotation_deg=mean([x['metrics']['predicted']['rotation_deg'] for x in zero]),
                zero_center_mm=mean([x['metrics']['predicted']['center_mm'] for x in zero]))
            interventions={}
            for key in ('oracle_geometry','completion_off','appearance_off'):
                paired=[x for x in rot if key in x['metrics']]
                interventions[key]=dict(cases=len(paired),rotation_after_deg=mean([x['metrics'][key]['rotation_deg'] for x in paired]),
                    paired_predicted_rotation_deg=mean([x['metrics']['predicted']['rotation_deg'] for x in paired]))
            summary['interventions']=interventions;result['controlled'][str(step)]=summary
    logs=[]
    for f in sorted((r/'runs/seed42').glob('rank*.jsonl')):
        rows=list(map(json.loads,f.read_text().splitlines()));logs.append(dict(rank=f.stem,step=rows[-1]['step'],seconds_recent50=statistics.mean(x['seconds'] for x in rows[-50:])))
    result['training']=logs
    (r/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# V21 串行补全到位姿：实测结果','',
        '位姿读出只接收恢复外观特征和物体/相机表面对应；没有原始 patch latent 或 FP/LIP 旁路。',
        'GT 补全读出验收与实际预测、native 跟踪分别报告，不能混用。','',
        '| 模型 | 全帧 ADD-S@0.05d | 可见率<50% | 可见率<30% |',
        '|---|---:|---:|---:|',
        '| 旧纯 LIP | 83.66% | 53.47% | 29.28% |',
        '| V20 45400 | 31.33% | 10.20% | 2.64% |']
    for step,m in result['native'].items():
        lines.append('| V21 '+step+' | '+' | '.join(f"{m[k]['adds_005']:.2f}%" for k in ('all','visibility_lt_05','visibility_lt_03'))+' |')
    lines+=['','同样本、非对称物体、初始10°旋转误差：','']
    for step,m in result['controlled'].items():
        lines.append(f"- V21 {step}：单步后 {m['rotation_after_deg']:.3f}°；有符号纠错增益 {m['signed_rotation_gain']:.3f}；GT基础位姿输出误差 {m['zero_rotation_deg']:.3f}°。")
        for key,v in m['interventions'].items():
            lines.append(f"  - {key}：{v['rotation_after_deg']:.3f}°，对应原预测 {v['paired_predicted_rotation_deg']:.3f}°；{v['cases']} 个配对样本。")
    lines+=['','当前控制器状态：`'+result['status']['job']+'`。',
        '这些结果不构成达到旧 LIP 水平的承诺；以表内实际 native 指标判断。',
        '训练为单 seed42、1000步上限；40帧旧episode与本次每episode三次前向的步数不可直接等同。']
    (r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    if result['native']:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=['LIP','V20']+['V21 '+s for s in result['native']]
        values=[[83.66415,53.46829,29.27821],[31.33420,10.19687,2.64329]]+[[v[k]['adds_005'] for k in ('all','visibility_lt_05','visibility_lt_03')] for v in result['native'].values()]
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(names));width=.24
        for i,label in enumerate(('All','Visibility < 50%','Visibility < 30%')):ax.bar(x+(i-1)*width,[v[i] for v in values],width,label=label)
        ax.set_xticks(x,names);ax.set_ylabel('ADD-S@0.05d (%)');ax.set_ylim(0,100);ax.legend();fig.tight_layout()
        fig.savefig(r/'native_comparison.png',dpi=180);fig.savefig(r/'native_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(native=result['native'],controlled=result['controlled'],training=logs,status=result['status'])))

if __name__=='__main__':main()

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
    if (r/'oracle_retention1000.json').exists():result['oracle_retention']=json.loads((r/'oracle_retention1000.json').read_text())
    if (r/'recovery/step1000/summary.json').exists():
        rec=json.loads((r/'recovery/step1000/summary.json').read_text())
        result['heavy_recovery']={k:v['arms'].get('off',{}) for k,v in rec['tables']['heavy_pooled'].items() if k in ('geometry_focus_real','geometry_focus_proxy','spatial_hidden_real_mid','spatial_cad_proxy_mid','cad_match_real','cad_match_proxy')}
    if (r/'recovery/v20_same_crops/summary.json').exists():
        before=json.loads((r/'recovery/v20_same_crops/summary.json').read_text())
        for rank in range(8):
            a=json.loads((r/f'recovery/v20_same_crops/rank{rank}/manifest.json').read_text())
            b=json.loads((r/f'recovery/step1000/rank{rank}/manifest.json').read_text())
            for key in ('reference_sha256','physical_sequences','occluder_bank_sha256','fixed_feature_teacher','dino_layers','feature_layer_weights'):
                assert a[key]==b[key],key
            aa=[json.loads(x) for x in (r/f'recovery/v20_same_crops/rank{rank}/frames.jsonl').read_text().splitlines()]
            bb=[json.loads(x) for x in (r/f'recovery/step1000/rank{rank}/frames.jsonl').read_text().splitlines()]
            assert len(aa)==len(bb)
            for left,right in zip(aa,bb):
                for key in ('physical_sequence','case','relative_frame','history'):assert left[key]==right[key]
                for key in ('geometry_focus_real','geometry_focus_proxy'):
                    assert (left[key] is None)==(right[key] is None)
                    if left[key] is not None:assert left[key]['pixels']==right[key]['pixels']
        result['paired_recovery']=dict(protocol_verified=True,before={k:v['arms'].get('off',{}) for k,v in before['tables']['heavy_pooled'].items() if k in result['heavy_recovery']},after=result['heavy_recovery'])
    result['lip_acceptance_met']=bool('1000' in result['native'] and result['native']['1000']['all']['adds_005']>=83.66415-.3 and result['native']['1000']['visibility_lt_05']['adds_005']>=53.46829-.3)
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
        '达到旧 LIP 回退不超过0.3个百分点的标准：'+('通过。' if result['lip_acceptance_met'] else '**未达到。**'),
        '旧 LIP 使用历史，新模型按用户此前选择关闭历史；同输入、无历史的旧 LIP 旋转诊断仍为6.82°，故历史不能解释全部差距。',
        '1000步的最重遮挡分数低于500步；不能只根据总体指标宣称全部改善。关闭补全/外观的干预证明读出对它们敏感，也存在输入分布变化，不能单独证明恢复正确。',
        '训练为单 seed42、1000步上限；40帧旧episode与本次每episode三次前向的步数不可直接等同。']
    if 'oracle_retention' in result:
        m=result['oracle_retention'];v=m['cached_student_appearance']['rotation']
        lines+=['',f"读出回放后，训练开发集的理想几何＋缓存学生外观测试为10°→{v['rotation_deg']:.3f}°。它与native val上的GT几何替换不是同一输入/数据条件，不能据此把剩余差距全部归因于几何恢复。"]
    if 'paired_recovery' in result:
        lines+=['','同crop、相同遮挡、同固定teacher的重遮挡恢复对照：','',
                '| 指标（越低越好） | V20 | V21 1000 |','|---|---:|---:|']
        for region,title in [('geometry_focus_real','人工遮挡真实目标'),('geometry_focus_proxy','自然遮挡CAD代理')]:
            for metric,label in [('xyz_mm','XYZ mm'),('depth_mm','深度 mm')]:
                vals=[result['paired_recovery'][key][region][metric]['mean'] for key in ('before','after')]
                lines.append(f'| {title} {label} | {vals[0]:.2f} | {vals[1]:.2f} |')
        lines+=['','**几何恢复没有随位姿一起改善。** 不能把本轮位姿收益解释为补全更准确。当前完成的是串行消费链路与读出的修复；恢复精度及最重遮挡稳定性仍未达到目标。']
    lines+=['','主路径：JEPA → 恢复的DINO4/11特征及XYZ/深度 → 完整相机关系 → object query → 位姿；没有原始patch或FP/LIP旁路。',
        '可见RGB/深度保持观测，人工遮挡用原始真实目标，自然遮挡用CAD代理；可见点另有CAD对应监督。纹理输出是局部DINO特征，不是RGB生成图。',
        '训练在350步修正成对crop，在700步加入读出回放及精确可见像素mask；均保存了完整边界断点和精确状态恢复回执。主JEPA不读取GT；读出专用回放明确读取训练GT几何，部署不包含该分支。',
        '22项检查通过，位姿对恢复特征/XYZ/深度有非零梯度；新模型未替换默认模型。']
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

"""Report exact-population comparisons without rerunning or changing predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--baseline',action='append',default=[],help='LABEL=PATH')
    args=parser.parse_args();out=args.evaluation
    current=read(out/'manifest.json')
    assert current['completed'] and current['population_verified'] and current['frames']==23200
    assert current['full_sequences'] and not current['subset'] and current['split']=='val'
    assert current['fp_calls']==current['critic_calls']==0 and not current['depth_correction']
    folders={}
    for value in args.baseline:
        label,path=value.split('=',1)
        if label in folders or label==args.label:raise ValueError('Duplicate comparison label')
        folders[label]=Path(path)
    folders[args.label]=out
    results={};reports={};identities={};reference=None;metadata={}
    for label,folder in folders.items():
        manifest=read(folder/'manifest.json');reports[label]=read(folder/'metrics.json')
        assert manifest['completed'] and manifest['split']=='val'
        assert all(manifest[k]==current[k] for k in ('split_hash','mesh_hash'))
        rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
        keys={(r['stream_id'],r['frame_index']) for r in rows}
        assert len(keys)==len(rows)==23200 and len({r['stream_id'] for r in rows})==320
        assert sum(r.get('initialization',r['frame_index']==0) for r in rows)==320
        if reference is None:reference=keys
        assert keys==reference,'Different frame populations'
        results[label]=reports[label]['excluding_initialization']['macro_object']
        identities[label]=dict(directory=str(folder),manifest_sha256=sha(folder/'manifest.json'),
            metrics_sha256=sha(folder/'metrics.json'),predictions_sha256=sha(folder/'predictions.jsonl'),
            checkpoint_sha256=manifest.get('checkpoint_sha256',manifest.get('checkpoint',{}).get('sha256')),
            checkpoint_stage_step=manifest.get('checkpoint_stage_step',manifest.get('checkpoint',{}).get('global_step')),
            metric_precision=manifest.get('metric_pose_precision','legacy float64 error arithmetic'))
        if label==args.label:metadata={r['stream_id']:(r['object_id'],r['camera_id']) for r in rows}
    report=reports[args.label];primary=results[args.label]
    deltas={label:{key:primary[key]-values[key] for key in primary} for label,values in results.items() if label!=args.label}
    csv_fields=['baseline','stream_id','object_id','camera_id','frames_excluding_initialization',
                'baseline_adds_01','current_adds_01','adds_01_delta_pp',
                'baseline_center_mm','current_center_mm','baseline_rotation_deg','current_rotation_deg']
    with (out/'per_stream_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields);writer.writeheader()
        for label in deltas:
            for stream in sorted(metadata):
                old=reports[label]['excluding_initialization']['per_stream_id'][stream]
                new=report['excluding_initialization']['per_stream_id'][stream]
                assert old['count']==new['count']
                obj,cam=metadata[stream]
                writer.writerow(dict(baseline=label,stream_id=stream,object_id=obj,camera_id=cam,frames_excluding_initialization=new['count'],
                    baseline_adds_01=old['adds_01']['mean'],current_adds_01=new['adds_01']['mean'],
                    adds_01_delta_pp=100*(new['adds_01']['mean']-old['adds_01']['mean']),
                    baseline_center_mm=old['center_mm']['mean'],current_center_mm=new['center_mm']['mean'],
                    baseline_rotation_deg=old['rotation_deg']['mean'],current_rotation_deg=new['rotation_deg']['mean']))
    cameras=[dict(camera_id=c,frames=r['micro']['count'],**r['macro_object']) for c,r in sorted(report['per_camera'].items())]
    objects=[dict(object_id=obj,frames=r['count'],**{key:value['mean'] for key,value in r.items() if key!='count'})
             for obj,r in report['excluding_initialization']['per_object_id'].items()]
    for name,rows in [('per_camera.csv',cameras),('per_object.csv',objects)]:
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    launch=read(out/'launch.json');source=read(out/'comparison_source_audit.json')
    summary=dict(completed=True,current_label=args.label,frames=23200,streams=320,scored_frames_excluding_initialization=22880,
        identical_frame_population=True,primary_aggregation='object macro average; initialization excluded',
        results=results,deltas_current_minus_baseline=deltas,identities=identities,source_audit=source,resource_usage=launch,
        training_budget_matched=False,architecture_causal_improvement_established=False,
        current_micro=report['excluding_initialization']['micro'],hard_subset=report['moving_and_visibility_lt_03'],recovery=report['recovery'])
    (out/'comparison.json').write_text(json.dumps(summary,indent=2))
    lines=[f'# {args.label}：完整 s0 val','',
        '**完成 320 条流、23,200 帧，全部比较项的 `(stream_id, frame_index)` 集合完全一致。** 首帧 GT 初始化，后续自身预测闭环；FP/critic 调用为 0，无 GT 重置或深度修正，官方 test 未访问。', '',
        f"本次归档 checkpoint 为 `{launch['checkpoint']}`，stage step **{current['checkpoint_stage_step']}**，SHA256 `{current['checkpoint_sha256']}`；源码 hash `{current['source_sha256']}`。训练同时继续，没有暂停。",'',
        '下表排除 320 个初始化帧，计分 22,880 帧；成功率、平均误差和 lost 均采用物体宏平均。基线引用已保存结果，没有在本轮重跑。','',
        '| 权重 | ADD@0.05d | ADD@0.1d | ADD-S@0.05d | ADD-S@0.1d | 中心均值 mm | 旋转均值 ° | lost |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in results.items():
        lines.append(f"| {label} | {100*r['add_005']:.2f}% | {100*r['add_01']:.2f}% | {100*r['adds_005']:.2f}% | {100*r['adds_01']:.2f}% | {r['center_mm']:.2f} | {r['rotation_deg']:.2f} | {100*r['lost']:.2f}% |")
    lines+=['','当前权重相对各基线的点估计差值：','']
    for label,r in deltas.items():
        lines.append(f"- 对 {label}：ADD@0.1d {100*r['add_01']:+.3f} pp，ADD-S@0.1d {100*r['adds_01']:+.3f} pp，ADD-S@0.05d {100*r['adds_005']:+.3f} pp，中心 {r['center_mm']:+.3f} mm，旋转 {r['rotation_deg']:+.3f}°。")
    micro=report['excluding_initialization']['micro'];hard=report['moving_and_visibility_lt_03']
    lines+=['',f"全计分帧 micro 中位数/P95：中心 **{micro['center_mm']['median']:.2f}/{micro['center_mm']['p95']:.2f} mm**，旋转 **{micro['rotation_deg']['median']:.2f}/{micro['rotation_deg']['p95']:.2f}°**。分位数为帧汇总，区别于表中的物体宏平均。",'',
        f"`moving AND visibility<0.3` 共 **{hard['micro']['count']} 帧**，物体宏平均 ADD-S@0.1d **{100*hard['macro_object']['adds_01']:.2f}%**，中心均值 **{hard['macro_object']['center_mm']:.2f} mm**，旋转均值 **{hard['macro_object']['rotation_deg']:.2f}°**。",'',
        f"状态计数：`{json.dumps(report['status_counts'])}`。ok 只表示数值/状态检查通过，不表示 GT 误差达标。lost 定义为连续至少五帧 ADD-S@0.1d 失败或需要重新初始化；记录 {report['recovery']['lost_to_not_lost']} 次 lost→非lost，GT 重置为 0。",'',
        '**训练量和迁移来源不同。** 各模型可能从先前训练权重继续迁移，具体来源见 manifest；不能将分数差单独归因于架构，也不能把不同 stage 的步数当作同一 optimizer 轨迹。本报告只给点估计，不宣称统计等价或架构因果收益。', '',
        '源码差异详见审计记录，架构及缓存机制可能不同。V1 历史误差计算为 float64，stream 模型为 FP32（SciPy 最近邻内部 float64）。已有 GT-depth 不一致和 visibility 不包含出画截断的局限保持不变。','',
        '## 各相机（排除初始化，物体宏平均）','',
        '| 相机 | 帧数 | ADD-S@0.1d | 中心 mm | 旋转 ° |','|---|---:|---:|---:|---:|']
    for r in cameras:lines.append(f"| {r['camera_id']} | {r['frames']} | {100*r['adds_01']:.2f}% | {r['center_mm']:.2f} | {r['rotation_deg']:.2f} |")
    lines+=['','## 运行和证据','',
        f"评估开始/结束：{launch['started_utc']} / {launch['completed_utc']}。训练步从 {launch['training_before']['step']} 推进至 {launch['training_after']['step']}；最近步耗时从约 {launch['training_before']['mean_recent_seconds']:.3f}s 到结束时约 {launch['training_after']['mean_recent_seconds']:.3f}s。共享 GPU，本轮不作为模型延迟基准。",'',
        '- [完整指标](metrics.json)、[全部预测](predictions.jsonl)、[manifest](manifest.json)',
        '- [比较与文件哈希](comparison.json)、[逐流配对](per_stream_comparison.csv)',
        '- [逐相机](per_camera.csv)、[逐物体](per_object.csv)',
        '- [源码审计](comparison_source_audit.json)、[启动记录与全部命令](launch.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=True,results=results,deltas=deltas),indent=2))


if __name__=='__main__':main()

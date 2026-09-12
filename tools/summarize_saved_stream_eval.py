"""Pair completed s0 validation predictions with retained standalone baselines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True);a=p.parse_args()
    current=a.evaluation;manifest=read(current/'manifest.json');assert manifest['completed'] and manifest['population_verified']
    assert manifest['checkpoint_stage_step']==5300 and manifest['architecture_id']=='stream_single'
    folders={'v1_34700':a.baseline_root/'legacy','single_short_300':a.baseline_root/'stream_single','single_5300':current}
    reports={};populations={};identities={};pairs={}
    for name,folder in folders.items():
        reports[name]=read(folder/'metrics.json');m=read(folder/'manifest.json')
        assert m['completed'] and m['split_hash']==manifest['split_hash'] and m['mesh_hash']==manifest['mesh_hash']
        rows=list(map(json.loads,(folder/'predictions.jsonl').read_text().splitlines()))
        lookup={(r['stream_id'],r['frame_index']):r for r in rows}
        assert len(lookup)==len(rows)==23200 and len({r['stream_id'] for r in rows})==320
        assert sum(r.get('initialization',r['frame_index']==0) for r in rows)==320
        populations[name]=set(lookup);pairs[name]=lookup
        identities[name]=dict(manifest_sha256=sha(folder/'manifest.json'),predictions_sha256=sha(folder/'predictions.jsonl'),
                              metrics_sha256=sha(folder/'metrics.json'),checkpoint_sha256=m.get('checkpoint_sha256',m.get('checkpoint',{}).get('sha256')))
    assert all(keys==populations['single_5300'] for keys in populations.values())
    primary={n:r['excluding_initialization']['macro_object'] for n,r in reports.items()}
    stats=reports['single_5300'];delta={key:primary['single_5300'][key]-primary['v1_34700'][key] for key in primary['single_5300']}
    stream_meta={r['stream_id']:(r['object_id'],r['camera_id']) for r in pairs['single_5300'].values()}
    with (current/'per_stream_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['stream_id','object_id','camera_id','frames_excluding_init','v1_adds_01','single_adds_01','adds_01_delta_pp','v1_center_mm','single_center_mm','v1_rotation_deg','single_rotation_deg']);writer.writeheader()
        for stream in sorted(stream_meta):
            old=reports['v1_34700']['excluding_initialization']['per_stream_id'][stream]
            new=stats['excluding_initialization']['per_stream_id'][stream];assert old['count']==new['count']
            obj,cam=stream_meta[stream]
            writer.writerow(dict(stream_id=stream,object_id=obj,camera_id=cam,frames_excluding_init=new['count'],v1_adds_01=old['adds_01']['mean'],single_adds_01=new['adds_01']['mean'],adds_01_delta_pp=100*(new['adds_01']['mean']-old['adds_01']['mean']),v1_center_mm=old['center_mm']['mean'],single_center_mm=new['center_mm']['mean'],v1_rotation_deg=old['rotation_deg']['mean'],single_rotation_deg=new['rotation_deg']['mean']))
    camera_rows=[]
    for camera,report in sorted(stats['per_camera'].items()):
        camera_rows.append(dict(camera_id=camera,frames=report['micro']['count'],**report['macro_object']))
    for name,rows in [('per_camera.csv',camera_rows),('per_object.csv',[dict(object_id=k,frames=v['count'],**{metric:entry['mean'] for metric,entry in v.items() if metric!='count'}) for k,v in stats['excluding_initialization']['per_object_id'].items()])]:
        with (current/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(completed=True,frames=23200,streams=320,scored_frames_excluding_initialization=22880,primary_aggregation='unweighted macro average of per-object frame means',identical_frame_population=True,
                 results=primary,delta_single_5300_minus_v1=delta,source_audit=read(current/'comparison_source_audit.json'),identities=identities,
                 current_micro_excluding_init=stats['excluding_initialization']['micro'],hard_moving_and_visibility_lt_03=stats['moving_and_visibility_lt_03'],recovery=stats['recovery'],resource_usage=read(current/'launch.json'),statistical_equivalence_established=False)
    (current/'comparison.json').write_text(json.dumps(summary,indent=2))
    text=['# stream_single 5,300 步：完整 s0 val','',
          '**已完成 320 条流、23,200 帧，精确核对三组 `(stream_id, frame_index)` 集合相同。** 只用首帧 GT 初始化，后续自身预测闭环；FP/critic 调用均为 0，无 GT 重置或深度修正。原始数据只读，官方 test 未访问。','',
          '评估权重：`'+manifest['checkpoint_sha256']+'`。本次沿用原 single 运行目录，源码 hash 为 `'+manifest['source_sha256']+'`。与先前验证相比，仅增加 metadata 形状检查和训练 CLI 参数检查；已重建旧版本 hash，确认推理与指标计算实现一致。','',
          '下表排除 320 个初始化帧，计分 22,880 帧；成功率、平均误差及 lost 为**物体宏平均**。旧两行引用已保存结果，本次未重复运行旧模型。','',
          '| 权重 | ADD@0.05d | ADD@0.1d | ADD-S@0.05d | ADD-S@0.1d | 中心均值 mm | 旋转均值 ° | lost |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('v1_34700','V1 34,700 standalone'),('single_short_300','single 固定片段短训 300'),('single_5300','single 正式训练 5,300')]:
        r=primary[name];text.append(f"| {label} | {100*r['add_005']:.2f}% | {100*r['add_01']:.2f}% | {100*r['adds_005']:.2f}% | {100*r['adds_01']:.2f}% | {r['center_mm']:.2f} | {r['rotation_deg']:.2f} | {100*r['lost']:.2f}% |")
    micro=stats['excluding_initialization']['micro'];hard=stats['moving_and_visibility_lt_03']
    text+=['',f"single 5,300 相比 V1 的 ADD-S@0.1d 差值为 **{100*delta['adds_01']:+.3f} 个百分点**；ADD-S@0.05d 为 **{100*delta['adds_005']:+.3f} 个百分点**。中心均值接近，但旋转均值更高。这里只比较点估计，没有宣称统计等价或已经超过 V1。",'',
           f"全计分帧的 micro 中位数/P95：中心 **{micro['center_mm']['median']:.2f}/{micro['center_mm']['p95']:.2f} mm**，旋转 **{micro['rotation_deg']['median']:.2f}/{micro['rotation_deg']['p95']:.2f}°**。这些分位数采用帧汇总，不与表中的物体宏平均混称。",'',
           f"`moving AND visibility<0.3` 交叉子集共 **{hard['micro']['count']} 帧**，物体宏平均 ADD-S@0.1d 为 **{100*hard['macro_object']['adds_01']:.2f}%**，中心均值 **{hard['macro_object']['center_mm']:.2f} mm**，旋转均值 **{hard['macro_object']['rotation_deg']:.2f}°**。",'',
           f"status 为 initialized {stats['status_counts'].get('initialized',0)} 帧、ok {stats['status_counts'].get('ok',0)} 帧；ok 是数值/状态检查通过，不是 GT 位姿成功标签。lost 定义为连续至少 5 帧 ADD-S@0.1d 失败或需要重新初始化，共记录 {stats['recovery']['lost_to_not_lost']} 次 lost→非lost 恢复，GT 重置为 0。",'',
           'single 短训 300 使用固定训练片段，5,300 步权重则来自正式训练；分数上升不能作为匹配训练预算的架构因果消融。已有 GT-depth 不一致和 visibility 不包含出画截断的局限保持不变。','',
           '## 各相机（排除初始化，物体宏平均）','',
           '| 相机 | 帧数 | ADD-S@0.1d | 中心均值 mm | 旋转均值 ° |','|---|---:|---:|---:|---:|']
    for r in camera_rows:text.append(f"| {r['camera_id']} | {r['frames']} | {100*r['adds_01']:.2f}% | {r['center_mm']:.2f} | {r['rotation_deg']:.2f} |")
    launch=summary['resource_usage'];text+=['','## 运行与证据','',
       f"本次评估与 dual 训练共享 GPU，未暂停或修改训练。开始/结束：{launch['started_utc']} / {launch['completed_utc']}。训练步从 {launch['training_before']['step']} 推进到 {launch['training_after']['step']}；最近训练步耗时由评估前约 {launch['training_before']['mean_recent_seconds']:.3f}s 到结束时约 {launch['training_after']['mean_recent_seconds']:.3f}s。本轮不提供模型延迟基准。",'',
       '- [全部指标](metrics.json)、[完整预测](predictions.jsonl)、[评估 manifest](manifest.json)',
       '- [比较与来源哈希](comparison.json)、[逐流配对](per_stream_comparison.csv)',
       '- [逐相机](per_camera.csv)、[逐物体](per_object.csv)',
       '- [启动命令与进程记录](launch.json)、[源码差异审计](comparison_source_audit.json)']
    (current/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(completed=True,results=primary,delta=delta),indent=2))


if __name__=='__main__':main()

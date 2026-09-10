"""Create the continuation report from completed receipts, without inferred results."""
import json
from pathlib import Path
root=Path('.');out=root/'runs/available_20260910'
read=lambda p:json.loads(Path(p).read_text())
r=read(out/'receipt.json');o=r['overfit'];cl=read(out/'val_closed/metrics.json');one=read(out/'val_onestep/metrics.json');sampling=read(out/'sampling_summary.log');manifest=read(out/'overfit32/manifest.json')
objects=len({x['object_id'] for x in manifest['sample_records']});sequences=len({'/'.join(x['stream'].split('/')[:2]) for x in manifest['sample_records']})
text=f'''# DexYCB-LIP v1：2026-09-10 远端已有数据续做

本次已完成三位完整受试者上的实现补齐、真实训练预检和固定验证诊断。**没有启动40,000-step长训练。完整10位受试者的s0预检仍未通过；工程可训练不等于当前checkpoint可用。**

本地源码：`/home/haoyi/Downloads/xd/dexycb_lip`。远端源码与运行：`/mnt/why/dexycb_lip`。本文相对路径在两处可查看；原始数据和checkpoint只保留在远端。旧单序列实验未覆盖，仍在原有 `runs/*subset*` 目录。

## 已有数据与官方split

冻结使用已上传且SHA256验证通过的 subjects 02、03、10 分包，以及 calibration/models。原压缩包只读；解压到工程缓存：

```bash
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_available_20260910
```

每位受试者均核对完整100段排序清单，使用官方全局subject index，未把02/03/10重新编号。共有 **300段录制、2,400个相机流、174,208帧、20个grasp target物体**。s0物理序列划分 **train 240 / val 20 / test 40**，互斥检查通过。全部RGB/depth解码及物体pose检查未发现错误。test仅做索引/文件有效性检查，没有参与训练采样、模型输入或评估。

证据：`archive_manifest.json`、`extraction.log`、`index.log`，以及 `cache/s0_available_20260910/audit.json`、`streams.jsonl`、`sequence_inventory.json`。索引中绑定pose cache SHA256和mesh hash。

## 实际通过的预检

| 项目 | 结果 | 本目录证据 |
|---|---|---|
| 单元测试 | **24 passed, 1 skipped**；skip为未提供依赖/权重的真实FP推理 | `pytest.log` |
| 几何检查 | 32/32有效样本，覆盖20物体；可见深度每样本中位误差再取中位数 **{r['depth_median_mm']:.3f} mm**；最大crop投影误差 **{r['crop_max_px']:.8f} px** | `geometry/geometry.json`、32组overlay/depth图 |
| 采样统计 | train **{sampling['train']['total']:,}**个候选clips；moving **{sampling['train']['moving']:,}**，heavy occlusion **{sampling['train']['occluded']:,}**，unknown visibility **{sampling['train']['unknown_visibility']:,}**；桶可重叠。采用规定的50/25/25混合 | `sampling_summary.log` |
| 真实32-clips过拟合 | 固定混合采样，涉及{objects}物体、{sequences}段录制；500步，关闭噪声/增强/dropout；loss **{o['initial']['loss']:.6f} → {o['final']['loss']:.6f}**，ADD **{o['final']['zero_motion']['add_m']*1000:.3f} → {o['final']['learned']['add_m']*1000:.3f} mm** | `overfit32/report.json`、`manifest.json`、`loss.jsonl` |
| 单卡显存 | batch 4/8/16/32均完成单步和U=4前后向+optimizer step；选每卡32、accum1、有效global batch256 | `probe.log`、`configs/resolved_8gpu.probe.json` |
| 8卡DDP | 50 optimizer steps，全部U=4，无hang/NaN，各rank样本与step一致；最大allocated峰值 **{r['peak_gpu_allocated_bytes']/2**30:.3f} GiB/卡** | `ddp/rank0.jsonl` 至 `rank7.jsonl`、`ddp_launcher.log` |
| 恢复 | 从step50恢复到53，8个rank的global step/scheduler step/采样位置一致；没有声称逐位确定性 | `ddp_resume.log`、`receipt.json` |
| 验证入口 | 官方val固定跨20物体：单步32 clips三方比较；20个完整stream连续闭环共 **{r['validation_frames']}帧**，未GT重置或丢掉lost帧 | `val_onestep/`、`val_closed/` |

配置 `configs/resolved_8gpu.yaml` 已更新到本次数据hash与真实probe/DDP结果，记录 `engineering_preflight_passed: true`、`official_s0_complete: false`、`preflight_approved: false`。这里只通过已就绪受试者的工程预检，不能称为完整s0验收。

## 验证质量：当前模型仍不合格

使用的是 **32-clips/500步过拟合checkpoint**，不是正式训练完的模型。验证样本由固定seed和object-balanced规则选择，所有方法的noisy历史/时间间隔相同，没有按结果筛选初始化。

| 官方val单步，32 clips | 平均center mm | 平均rotation deg | 平均ADD mm |
|---|---:|---:|---:|
| Zero-motion | {one['zero-motion']['micro']['center_mm']['mean']:.3f} | {one['zero-motion']['micro']['rotation_deg']['mean']:.3f} | {one['zero-motion']['micro']['add_m']['mean']*1000:.3f} |
| Constant-velocity | {one['constant-velocity']['micro']['center_mm']['mean']:.3f} | {one['constant-velocity']['micro']['rotation_deg']['mean']:.3f} | {one['constant-velocity']['micro']['add_m']['mean']*1000:.3f} |
| Learned | {one['learned']['micro']['center_mm']['mean']:.3f} | {one['learned']['micro']['rotation_deg']['mean']:.3f} | {one['learned']['micro']['add_m']['mean']*1000:.3f} |

Learned尚未超过恒速基线。连续闭环只在每stream首帧使用GT初始化一次：

- 平均center误差 **{cl['micro']['center_mm']['mean']:.3f} mm**，rotation误差 **{cl['micro']['rotation_deg']['mean']:.3f} deg**。
- micro ADD recall@0.1d **{cl['micro']['add_01']['mean']*100:.3f}%**；micro ADD-S recall@0.1d **{cl['micro']['adds_01']['mean']*100:.3f}%**。
- macro-over-object ADD-S recall@0.1d **{cl['macro_object']['adds_01']*100:.3f}%**。
- lost fraction **{cl['micro']['lost']['mean']*100:.3f}%**，采用任务书连续5帧ADD-S>0.1d标准，全部{r['validation_frames']}帧保留。
- batch1同步端到端均值 **{cl['latency_seconds']['end_to_end']['mean']*1000:.3f} ms**，p95 **{cl['latency_seconds']['end_to_end']['p95']*1000:.3f} ms**。包括读取/解码/crop/render/network，排除GT metrics和可视化，没有FP。

这是部分val的固定工程诊断，不是完整val或test成绩，也未用于选择best checkpoint。不能据此宣称交互建模有效。完整per-object/sequence、visibility、moving子集和首次连续失败记录都在JSON中；可视化与指标使用同一份保存的预测。评估manifest包含checkpoint SHA256、step、config和split/mesh hashes。

## 本次实际代码修正

- 支持完整已就绪受试者的官方split索引，拒绝对缺少序列的单个受试者重新排序；并行文件审计及pose cache hash。
- mesh缓存按物体文件共享，减少多相机/多序列重复加载。
- 采样统计可在8卡分片计算；合并校验rank归属、重复clip和split hash。
- 几何抽查、quick val与单步诊断覆盖不同物体；固定manifest，避免只取目录开头的同一物体。
- 评估记录checkpoint身份；修复best选择状态未保留在last导致resume遗忘best的问题，best/last均atomic写入。
- 主模型架构与监督保持任务书第一版：无MANO/手监督，ImageNet ResNet50 layer3、9通道几何、2层cross-attention、4层causal temporal、256 latent、解耦rotation/center correction。

## 未完成条件与命令

1. 完整10位受试者的数据尚未纳入这次冻结清单，不能把本次300序列标为完整s0。其余上传任务保留运行，未删改已有压缩包。
2. 未进行正式40,000-step训练。当前质量不合格，后续仍需完整数据上的训练与完整val验收。
3. 没有真实FoundationPose依赖/权重，只有独立adapter及对应单元测试。
4. 没有本次`best.pt`：没有用quick val选择best。当前checkpoint位于远端：`{out}/overfit32/last.pt`（500步）与`{out}/ddp/last.pt`（53步）。

恢复当前已就绪数据的短跑（明确仍为短诊断）：

```bash
cd /mnt/why/dexycb_lip
source scripts/env.sh
export DEX_YCB_DIR=/mnt/why/dexycb_lip/cache/raw_available_20260910
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 -m lip.train \\
  --config configs/resolved_8gpu.yaml --data-root "$DEX_YCB_DIR" \\
  --index-root cache/s0_available_20260910 --allow-verified-subset \\
  --output runs/available_20260910/ddp --resume runs/available_20260910/ddp/last.pt \\
  --max-steps 55 --force-rollout 4
```

完整数据根目录准备好后，正式训练/恢复命令：

```bash
# 将DEX_YCB_DIR指向包含10位受试者、calibration和models的完整已解压目录
bash scripts/preflight.sh
bash scripts/train_8gpu.sh
# 或恢复正式训练：
bash scripts/train_8gpu.sh --resume runs/lip_v1_s0/last.pt
```

完整数据门槛尚未通过时，正式启动脚本会拒绝运行。这些长训练命令本次没有执行。最后GPU检查8卡均为0 MiB使用，短训练/评估均已结束（`gpu_final.csv`）。
'''
(out/'STATUS.md').write_text(text)
print('wrote',out/'STATUS.md')

# RGB-D / FoundationPose 一致性排查

已定位到组合模型掉分的一项主要原因：当前数据的观测深度与 GT pose 下的 mesh 深度存在毫米到厘米级不一致，FP 会追随这部分深度偏差。证据已从相关性推进到配对输入干预。具体应归因于深度测量、深度与彩色相机对齐、还是相机外参的哪一部分，尚未唯一确定。

本次使用此前固定的 20 个验证集物体/物理帧，每物体一帧；扩展到相同时间的 8 个视角，共 160 个相机帧、644 个有效物体观测。所有诊断均写入 `runs/rgbd_alignment_audit`，没有改原始数据、训练配置、主模型或 FP 权重。没有访问 MANO 或手姿态数组。物体分割仅用于诊断像素筛选；桌面检查仅读取背景分割。

**最直接的证据：只改变目标深度，FP 的 Z 偏差基本消失。**

两组均从相同 GT pose 开始，保持 RGB、K、FP 权重、两次 refine 及其他深度像素一致。干预组仅把“目标物体分割、GT 渲染有效、实测深度有效”的交集替换为 GT 渲染深度，缺失深度仍保持缺失。

| 20 帧配对诊断 | 原始观测深度 | 目标 GT 渲染深度 |
|---|---:|---:|
| 平均中心误差 | 9.340 mm | 3.055 mm |
| 平均相机 Z 偏移 | +7.632 mm | −0.093 mm |
| 平均旋转误差 | 4.680° | 3.703° |

19/20 帧中心误差降低。原始深度组再次运行所得 FP pose 与此前记录最大元素差为 **0**。排除明显画面截断的 3 帧后，另外 17 帧中心误差仍从 **6.897 → 2.986 mm**，平均 Z 从 **+5.268 → −0.262 mm**。

这属于使用 GT 的 oracle 诊断，不能计入模型验证成绩，也不是可部署的深度修正方法。它证实这 20 帧中的深度输入对后移有实质作用；没有证明消除这部分偏差后完整闭环能提升多少，也没有解释全部旋转误差。

**真实图像已进行人工式视觉检查。**

实际查看了杯子、电钻、布丁盒、泡沫砖、芥末瓶和水壶的 RGB、GT / FP 纹理投影、深度残差和剖面图，并查看了桌面点云高度图及背景像素筛选图。这里是 AI 视觉检查，不代表独立人工 GT 校验。

- 电钻：目标内部深度整体偏远约 19 mm，FP 后移约 19.5 mm。换用 GT 深度后中心误差 **22.05 → 5.09 mm**。
- 杯子：目标内部约 25 mm 深度差，FP 后移约 33.9 mm。干预后中心误差 **35.08 → 1.81 mm**。
- 泡沫砖：完整留在画面内，约 13.3 mm 深度差。干预后中心误差 **12.90 → 2.33 mm**。
- 芥末瓶和水壶对照：该视角深度残差接近零，原始 FP 中心误差约 1.7～1.8 mm。

杯子和电钻的大误差样例分别只有约 **48.0% / 26.1%** 的完整投影留在画面内；木块约 **64.4%**。之前的 `visibility` 是画面内 silhouette 的可见比例，没有惩罚出画，因此不能把 `visibility >= .9` 解释成“完整物体高可见”。本次新记录 `image_fraction`，并单独报告排除截断的结果。既有完整验证指标没有被覆盖或重算口径。

**跨相机结果与实际验证掉分方向一致。**

以下深度偏差仅统计 `image_fraction >= .98` 且画面内可见比例 `>= .9` 的 **460 个观测**，先取物体内部像素的有符号残差中位数，再按相机取中位数。实际验证列是 31k 保存的混合历史上，本帧 LIP 提案到本帧 FP 输出的配对变化；每相机 2,860 次更新，排除首帧 GT，采用物体宏平均。它不是纯 LIP 轨迹与混合轨迹的另一次比较。

| 相机末四位 | 有效观测 | 深度偏差中位数 | 本帧 FP 的 ADD@0.1d 变化 |
|---|---:|---:|---:|
| 0125 | 52 | +1.49 mm | −1.86 pp |
| 0362 | 53 | +0.21 mm | −1.14 pp |
| 0917 | 65 | +0.82 mm | −5.87 pp |
| 0263 | 69 | −0.22 mm | −7.09 pp |
| 0857 | 37 | +4.10 mm | −10.79 pp |
| 0861 | 44 | +9.62 mm | −15.01 pp |
| 1900 | 73 | +10.40 mm | −16.98 pp |
| 2010 | 67 | +8.62 mm | −10.84 pp |

这是相机层面的描述性关联；视角、对象表面和遮挡仍可能影响结果。0263 的中位深度偏差接近零，但掉分仍明显，也说明一个相机级常数无法解释所有误差。

**桌面也存在跨相机几何不一致。**

按官方外参把观测点云变换到共同坐标，比较同一桌面 XY 网格中的高度。高度计算不读取物体 GT pose。筛选规则及掩膜均保存：背景分割腐蚀 9×9、RGB 最大通道 <65、Apriltag XY 范围 `[.15,1.05] × [.05,.55]` 米、`|Z|<.04` 米；20 mm 网格至少 5 点，每对相机至少 20 个共同网格。帧选集沿用前述 20 帧。

相对 0362，相机 0861 在共同网格中重建的桌面中位高度低 **9.69 mm**；0857、1900、2010 分别低约 **6.32 / 7.18 / 6.78 mm**。比较经过逐帧配对再取中位数，且高度图存在空间梯度。桌布并非精密平面，因此不把绝对 Z=0 当作真值；同一区域的相机间差异仍然表明问题并不限于物体 CAD 或物体 GT。

**已排查的实现假设。**

| 检查 | 实测结果与边界 |
|---|---|
| 数据读取约定 | 与官方相同：`aligned_depth_to_color`、color 内参、深度除 1000、`textured_simple.obj` |
| raw pose / centered pose | 本次恢复 original pose 与原始 `pose_y` 最大元素差 3.73e−9；此前真实 FP adapter 往返回环最大差 1.49e−8 |
| 缓存 mesh vs 原始 mesh | 相机坐标下顶点最大差 1.22e−7 m |
| 项目 renderer vs FP 原生 renderer | 原生原始约定相差半像素；对齐像素约定后，逐帧差值 P99 的最大值仅 0.00223 mm。90,851 个内部像素中有 1 个差 >1 mm，最大 14.76 mm；并非逐像素完全相等，但无法解释整个物体的系统性偏差 |
| 简化 mesh | 5 个案例换原始高分辨率 `textured.obj` 后，深度变化的逐帧中位绝对值为 0.008～0.093 mm；主要深度差保留 |
| 多相机 pose 变换 | 变换到公共坐标后的 GT pose 最大元素差 2.18e−7，未发现相机索引或矩阵方向的明显错配 |
| depth-Z / color-Z 假设 | 对 YAML 外参两种方向分别估算，逐观测轴向差中位数最大约 4 mm；仅补这一项不能解释所有主要偏差，仍不能据此排除完整标定或对齐误差 |

官方依据：[DexYCB dataset reader](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py)、[sequence loader](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/sequence_loader.py)。目前 RealSense SDK 的 [align 实现](https://github.com/realsenseai/librealsense/blob/master/src/proc/align.cpp) 会把原深度值写入对齐图；这说明检查两种 Z 定义有必要，但并不能证明当前数据的采集版本或唯一成因。

**对后续训练的含义。**

这解释了为什么 LIP 已接近 GT 时，无条件接受 FP 仍会掉分：GT 位姿目标和观测深度拟合目标并不完全一致。应优先用 train 的独立物理序列或原始标定数据定位并验证相机/空间相关校正，再重新采集 FP-aware / critic 标签，进行原始与校正输入的配对闭环验证。不能用本次 val GT 拟合一组偏移再把同一 val 提升当作泛化结果，也不建议套统一 −Z 常数。

深度对齐后仍需验证 FP 更新的净收益和接受机制。当前 actor 只使用 `q_converge`，而推理没有用 `q_improve` 拒绝有害更新；“FP 后仍在 0.1d 内”不能保证改善。

LIP 从早期 checkpoint 到后期的严重遮挡旋转退化仍是独立未解项。本次没有做 basin loss / rollout / 加速实现的训练消融，不能给这些因素分配因果责任。当前训练保持既有运行状态，本次诊断没有切换其配置。

证据入口：

- [总览图](../runs/rgbd_alignment_audit/audit_summary.png)、[机器可读汇总](../runs/rgbd_alignment_audit/summary.json)
- [电钻 RGB-D 图](../runs/rgbd_alignment_audit/object_15.png)、[杯子图](../runs/rgbd_alignment_audit/object_14.png)、[泡沫砖图](../runs/rgbd_alignment_audit/object_21.png)
- [桌面跨相机图](../runs/rgbd_alignment_audit/table_camera_height.png)
- [原生 renderer 检查](../runs/rgbd_alignment_audit/renderer_visual_audit.json)、[mesh 与像素约定](../runs/rgbd_alignment_audit/mesh_depth_hypotheses.json)
- [644 个物体观测](../runs/rgbd_alignment_audit/multiview_depth.json)、[桌面配对结果](../runs/rgbd_alignment_audit/table_consistency.json)
- [40 次真实 FP 调用](../runs/rgbd_alignment_audit/fp_depth_intervention.json)、[FP 日志](../runs/rgbd_alignment_audit/fp_depth_intervention.log)
- [文件哈希收据](../runs/rgbd_alignment_audit/receipt.json)
- [实际完成与本地同步校验](../runs/rgbd_alignment_audit/verification.json)：70 个文件哈希一致、6 个入口语法解析成功、5 项诊断及汇总均完成。这是实际诊断和文件完整性检查，不是新运行的一套模型单元测试。

六个实际执行入口为 `tools/diagnose_rgbd_alignment.py`、`diagnose_multiview_depth.py`、`diagnose_table_consistency.py`、`check_mesh_depth_hypotheses.py`、`probe_fp_depth_intervention.py` 和 `summarize_rgbd_audit.py`。前四个使用 `.venv-fp/bin/python`、`PYTHONPATH=src`；FP 干预使用 `PYTHONPATH=runs/lip_fp_31000/candidate/src`，以复用实际验证版本。每个入口均支持 `--root . --out runs/rgbd_alignment_audit`。日志依次为 `visual_audit.log`、`multiview_depth.log`、`table_consistency.log`、`mesh_depth_hypotheses.log`、`fp_depth_intervention.log`、`summary_stdout.log`。FP 干预首次使用旧根目录源码时导入失败，已改用验证 snapshot 重跑成功，失败日志保留为 `fp_depth_intervention_wrong_import.log`。桌面汇总图对无观测网格出现一次 All-NaN 提示，相应网格留白，不作为数值样本。
